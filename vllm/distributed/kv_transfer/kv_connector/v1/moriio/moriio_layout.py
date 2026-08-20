# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


class LayerTransferGeometry(NamedTuple):
    num_blocks: int
    block_size: int
    block_len: int
    slot_size_bytes: int
    block_stride: int


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_to_spec.update(
                {
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in group.layer_names
                }
            )
        else:
            layer_to_spec.update(
                {layer_name: group_spec for layer_name in group.layer_names}
            )
    return layer_to_spec


def is_mla_cache_layer(
    layer_to_spec: Mapping[str, KVCacheSpec], layer_name: str
) -> bool:
    try:
        spec = layer_to_spec[layer_name]
    except KeyError as e:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}") from e
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))


def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> LayerTransferGeometry:
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    spec = layer_to_spec[layer_name]
    if not isinstance(spec, AttentionSpec) or len(shape) != 4:
        raise ValueError(
            f"Unsupported MoRIIO cache for layer {layer_name}: expected a "
            f"standardized 4-D attention cache, got {tuple(shape)}"
        )

    _, num_heads, kernel_num_states, content_dim = shape
    if (
        num_heads != spec.num_heads
        or kernel_num_states <= 0
        or spec.num_states % kernel_num_states != 0
        or content_dim * element_size != spec.state_content_size_bytes
        or not is_non_overlapping_and_dense(kv_cache[0])
    ):
        raise ValueError(
            f"Unsupported MoRIIO cache shape or strides for layer "
            f"{layer_name}: {tuple(shape)}, {tuple(stride)}"
        )

    kernel_blocks_per_block = spec.num_states // kernel_num_states
    kernel_num_blocks = shape[0]
    if kernel_num_blocks % kernel_blocks_per_block != 0:
        raise ValueError(
            f"Unsupported MoRIIO cache shape for layer {layer_name}: "
            f"{kernel_num_blocks} kernel blocks are not divisible by "
            f"{kernel_blocks_per_block}"
        )

    num_blocks = kernel_num_blocks // kernel_blocks_per_block
    slot_size_bytes = num_heads * content_dim * element_size
    return LayerTransferGeometry(
        num_blocks=num_blocks,
        block_size=spec.block_size,
        block_len=kernel_blocks_per_block * kernel_num_states * slot_size_bytes,
        slot_size_bytes=slot_size_bytes,
        block_stride=stride[0] * kernel_blocks_per_block,
    )


def iter_layer_registration_regions(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> list[tuple[torch.Tensor, int]]:
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    spec = layer_to_spec[layer_name]
    assert isinstance(spec, AttentionSpec)
    block_stride_bytes = geometry.block_stride * kv_cache.element_size()
    region_len = max(
        geometry.num_blocks * geometry.block_len,
        (geometry.num_blocks - 1) * block_stride_bytes + spec.page_size_bytes,
    )
    return [(kv_cache, region_len)]


def merge_contiguous_offsets(
    offsets_local: list[int],
    offsets_remote: list[int],
    sizes: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if not offsets_local:
        return [], [], []
    if not (len(offsets_local) == len(offsets_remote) == len(sizes)):
        raise ValueError("Input list lengths mismatch")

    rows = sorted(zip(offsets_local, offsets_remote, sizes), key=lambda row: row[0])
    merged: list[list[int]] = []
    for local, remote, size in rows:
        if (
            merged
            and local == merged[-1][0] + merged[-1][2]
            and remote == merged[-1][1] + merged[-1][2]
        ):
            merged[-1][2] += size
        else:
            merged.append([local, remote, size])

    return (
        [row[0] for row in merged],
        [row[1] for row in merged],
        [row[2] for row in merged],
    )


def compute_block_transfer_offsets(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    local_block_ids: list[int],
    remote_block_ids: list[int],
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
) -> tuple[list[int], list[int], list[int]]:
    # A shorter (or empty) local list is the READ-mode "drop the transfer, just
    # free the prefill blocks" case (full-prefix-hit / aborted-before-scheduled):
    # decode pulls fewer blocks than the prefill holds. The zip loop below pairs
    # local[i]<->remote[i] and sizes by len(local), so a short local transfers
    # only what decode allocated and an empty local is a no-op. A longer local
    # list is a genuine bug and still fails loudly.
    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    element_size = kv_cache.element_size()
    transfer_size_byte = geometry.block_len
    total = len(local_block_ids)
    offset_local = [0] * total
    offset_remote = [0] * total
    sizes = [transfer_size_byte] * total

    for i, (lb, rb) in enumerate(zip(local_block_ids, remote_block_ids)):
        offset_local[i] = element_size * lb * geometry.block_stride
        offset_remote[i] = element_size * rb * geometry.block_stride

    return merge_fn(offset_local, offset_remote, sizes)
