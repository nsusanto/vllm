# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import threading
from collections import OrderedDict, defaultdict
from queue import Queue
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    MLAAttentionSpec,
)

mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_engine = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine"
)
moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)
msgpack = importlib.import_module("msgpack")

ROLE = moriio_common.ROLE
MoRIIOError = moriio_common.MoRIIOError
MoRIIOTransferAck = moriio_common.MoRIIOTransferAck
RemoteAllocInfo = moriio_common.RemoteAllocInfo
WriteTask = moriio_common.WriteTask
set_role = moriio_common.set_role
MoRIIOWrapper = moriio_engine.MoRIIOWrapper
MoRIIOWriter = moriio_engine.MoRIIOWriter


def _full_spec(
    block_size: int = 4, num_kv_heads: int = 2, head_size: int = 3
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _mla_spec(block_size: int = 4) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )


def _cache_views(
    spec: FullAttentionSpec | MLAAttentionSpec,
    *,
    num_blocks: int = 8,
    num_layers: int = 1,
    layout: KVCacheLayout = KVCacheLayout.LBNHC,
    kernel_block_size: int | None = None,
) -> list[torch.Tensor]:
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    return dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=kernel_block_size,
    )


def _worker(
    kv_caches: dict[str, torch.Tensor],
    layer_to_spec: dict[str, object],
    num_blocks: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_caches=kv_caches,
        layer_to_spec=layer_to_spec,
        num_blocks=num_blocks,
        block_size=4,
    )


def _remote_meta(num_blocks: int = 16) -> SimpleNamespace:
    return SimpleNamespace(num_blocks=num_blocks)


def _writer_with_fake_worker(fake_worker: Any) -> Any:
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: fake_worker
    writer._write_task_q = Queue()
    writer._write_state_lock = threading.Lock()
    writer._scheduled_writes = defaultdict(int)
    writer._scheduled_layers = defaultdict(set)
    writer._sealed_writes = {}
    writer.ensure_worker_started = lambda: None
    return writer


def _wrapper_for_messages() -> Any:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_req_ids = []
    wrapper.done_write_cache_req_ids = []
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _write_task(layer_name: str, transfer_id: str = "xfer") -> Any:
    return WriteTask(
        request_id="req",
        transfer_id=transfer_id,
        dst_engine_id="remote-engine",
        local_block_ids=[1, 3],
        remote_block_ids_hint=None,
        layer_name=layer_name,
        event=None,
        remote_notify_port=7000,
        remote_ip="127.0.0.1",
    )


@pytest.mark.parametrize(
    ("spec", "expected_geometry", "expected_offsets"),
    [
        pytest.param(
            _full_spec(),
            {
                "num_blocks": 8,
                "block_size": 4,
                "block_len": 96,
                "slot_size_bytes": 24,
                "block_stride": 48,
            },
            ([96, 288], [384, 480], [96, 96]),
            id="full-attention",
        ),
        pytest.param(
            _mla_spec(),
            {
                "num_blocks": 8,
                "block_size": 4,
                "block_len": 24,
                "slot_size_bytes": 6,
                "block_stride": 12,
            },
            ([24, 72], [96, 120], [24, 24]),
            id="mla",
        ),
    ],
)
def test_standardized_caches_compute_expected_geometry_and_offsets(
    spec, expected_geometry, expected_offsets
):
    (cache,) = _cache_views(spec)
    worker = _worker({"layer": cache}, {"layer": spec})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", cache, worker.layer_to_spec
    )
    for field, expected in expected_geometry.items():
        assert getattr(geometry, field) == expected

    assert (
        moriio_layout.compute_block_transfer_offsets(
            "layer",
            cache,
            worker.layer_to_spec,
            [1, 3],
            [4, 5],
        )
        == expected_offsets
    )


def test_block_interleaved_cache_uses_strided_offsets():
    spec = _full_spec()
    _, cache = _cache_views(
        spec,
        num_layers=2,
        layout=KVCacheLayout.BLNHC,
    )
    layer_to_spec = {"layer": spec}

    geometry = moriio_layout.get_layer_transfer_geometry("layer", cache, layer_to_spec)

    assert geometry.block_stride * cache.element_size() == 192
    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, layer_to_spec, [1, 3], [4, 5]
    ) == ([192, 576], [768, 960], [96, 96])


def test_mixed_layers_compute_distinct_offsets_per_layer():
    full_spec = _full_spec()
    mla_spec = _mla_spec()
    kv_caches = {
        "layer_compact": _cache_views(full_spec)[0],
        "block_interleaved": _cache_views(
            full_spec,
            num_layers=2,
            layout=KVCacheLayout.BLNHC,
        )[1],
        "mla": _cache_views(mla_spec)[0],
    }
    worker = _worker(
        kv_caches,
        {
            "layer_compact": full_spec,
            "block_interleaved": full_spec,
            "mla": mla_spec,
        },
    )

    layer_compact = moriio_layout.compute_block_transfer_offsets(
        "layer_compact",
        kv_caches["layer_compact"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
    )
    block_interleaved = moriio_layout.compute_block_transfer_offsets(
        "block_interleaved",
        kv_caches["block_interleaved"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
    )
    mla = moriio_layout.compute_block_transfer_offsets(
        "mla",
        kv_caches["mla"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
    )

    assert layer_compact != block_interleaved
    assert layer_compact != mla
    assert block_interleaved != mla


def test_write_transfer_plan_caches_offsets_per_geometry():
    kv_caches = {
        "dense0": _cache_views(_full_spec())[0],
        "dense1": _cache_views(_full_spec())[0],
        "indexer": _cache_views(_mla_spec())[0],
    }
    calls: list[str] = []

    class FakeWorker:
        kv_caches: dict[str, torch.Tensor]
        layer_name_to_local_kv_cache_metadata: dict[str, list[Any]]

        def _compute_block_transfer_offsets(
            self, layer_name, local_block_ids, remote_block_ids, remote_moriio_meta
        ):
            calls.append(layer_name)
            call_id = len(calls)
            return ([call_id], [call_id + 10], [call_id + 20])

    fake_worker = FakeWorker()
    fake_worker.kv_caches = kv_caches
    fake_worker.layer_name_to_local_kv_cache_metadata = {name: [] for name in kv_caches}
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: fake_worker
    request_info = RemoteAllocInfo(block_ids=[4, 5])
    remote_meta = _remote_meta()

    dense0_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="dense0",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )
    dense1_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="dense1",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )
    indexer_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="indexer",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )

    assert calls == ["dense0", "indexer"]
    assert dense0_plan.transfer_local_offsets == [1]
    assert dense1_plan.transfer_local_offsets == [1]
    assert indexer_plan.transfer_local_offsets == [2]
    assert len(request_info.transfer_offsets) == 2


def test_write_scheduler_deduplicates_layers_and_seals_expected_count():
    request_info = RemoteAllocInfo(block_ids=[4, 5])
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))

    assert writer.schedule_write(_write_task("dense0"))
    assert not writer.schedule_write(_write_task("dense0"))
    assert writer.schedule_write(_write_task("indexer"))

    assert writer._write_task_q.qsize() == 2
    writer.seal_pending_transfers()

    assert request_info.writes_expected == 2
    assert writer._sealed_writes["xfer"] == 2


def test_write_completion_notifies_once_after_all_sealed_writes_finish():
    class FakeWrapper:
        def __init__(self):
            self.done_remote_allocate_req_dict = {}
            self.done_req_ids = []
            self.lock = threading.Lock()
            self.notifications = []
            self.wait_count = 0
            self.waited_statuses = []
            self._terminal_transfer_ids = OrderedDict()

        def waiting_for_transfer_complete(self, transfer_statuses=None):
            self.wait_count += 1
            self.waited_statuses.append(list(transfer_statuses or []))

        def _is_transfer_terminal_locked(self, transfer_id):
            return transfer_id in self._terminal_transfer_ids

        def _mark_transfer_terminal_locked(self, transfer_id):
            self._terminal_transfer_ids[transfer_id] = None

        def send_notify(
            self,
            transfer_id,
            remote_ip,
            remote_port,
            message_type=None,
            message_fields=None,
        ):
            self.notifications.append(
                (transfer_id, remote_ip, remote_port, message_type, message_fields)
            )

    wrapper = FakeWrapper()
    request_info = RemoteAllocInfo(block_ids=[4, 5], writes_expected=2)
    request_info.transfer_statuses.extend(["status-a", "status-b"])
    request_info.completion_request_id = "req"
    request_info.completion_remote_notify_port = 7000
    request_info.completion_remote_ip = "127.0.0.1"
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = _writer_with_fake_worker(
        SimpleNamespace(moriio_wrapper=wrapper, tp_rank=2)
    )
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_write_done("xfer", request_info)
    assert wrapper.notifications == []
    writer._mark_write_done("xfer", request_info)
    writer._finalize_if_complete("xfer", request_info)

    assert wrapper.notifications == [("xfer", "127.0.0.1", 7002, "write_done", None)]
    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper.wait_count == 1
    assert wrapper.waited_statuses == [["status-a", "status-b"]]
    assert request_info.transfer_statuses == []
    assert wrapper._is_transfer_terminal_locked("xfer")


def test_moriio_wrapper_waits_scoped_statuses_without_global_drain():
    class FakeStatus:
        def __init__(self):
            self.checked = 0

        def Succeeded(self):
            self.checked += 1
            return True

        def Failed(self):
            return False

    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper._transfer_timeout = 1
    global_status = FakeStatus()
    scoped_status = FakeStatus()
    wrapper.transfer_status = [global_status]

    wrapper.waiting_for_transfer_complete([scoped_status])

    assert scoped_status.checked == 1
    assert global_status.checked == 0
    assert wrapper.transfer_status == [global_status]


def test_write_failure_marks_terminal_and_clears_scheduled_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[4, 5])
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_request_done("xfer")

    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper._is_transfer_terminal_locked("xfer")
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_schedule_write_rejects_terminal_transfer_without_recreating_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[4, 5])
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 1
    writer._scheduled_layers["xfer"] = {"dense0"}
    writer._sealed_writes["xfer"] = 1

    writer._mark_request_done("xfer")

    assert not writer.schedule_write(_write_task("indexer"))
    assert writer._write_task_q.empty()
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_late_remote_blocks_message_is_ignored_after_transfer_done():
    set_role(ROLE.PRODUCER)
    wrapper = _wrapper_for_messages()
    with wrapper.lock:
        wrapper._mark_transfer_terminal_locked("xfer")

    wrapper._handle_message(
        msgpack.dumps(
            {
                "type": "remote_blocks",
                "req_id": "req",
                "transfer_id": "xfer",
                "block_notify_list": [4, 5],
                "decode_rank": 3,
            }
        )
    )

    assert "xfer" not in wrapper.done_remote_allocate_req_dict


@pytest.mark.parametrize(
    ("role", "payload", "expected"),
    [
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [4, 5],
                    "decode_rank": 3,
                }
            ),
            "remote_blocks",
            id="remote-blocks",
        ),
        pytest.param(
            ROLE.CONSUMER,
            msgpack.dumps({"type": "write_done", "transfer_id": "xfer"}),
            "write_done",
            id="write-done",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps({"type": "release", "transfer_id": "xfer"}),
            MoRIIOTransferAck("xfer"),
            id="release",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "release",
                    "transfer_id": "xfer",
                    "consumer_tp_size": 8,
                }
            ),
            MoRIIOTransferAck("xfer", 8),
            id="release-consumer-tp-size",
        ),
        pytest.param(None, b"xfer", "plain", id="plain-string"),
    ],
)
def test_moriio_wrapper_routes_valid_messages(role, payload, expected):
    wrapper = _wrapper_for_messages()
    completions: list[str] = []
    if role is not None:
        set_role(role)
    if expected == "plain":
        wrapper._handle_completion_message = completions.append

    wrapper._handle_message(payload)

    if expected == "remote_blocks":
        request_info = wrapper.done_remote_allocate_req_dict["xfer"]
        assert request_info.block_ids == [4, 5]
        assert request_info.decode_dp_rank == 3
    elif expected == "write_done":
        assert wrapper.done_write_cache_req_ids == ["xfer"]
    elif expected == "plain":
        assert completions == ["xfer"]
    else:
        assert wrapper.done_req_ids == [expected]
        assert wrapper._is_transfer_terminal_locked("xfer")


@pytest.mark.parametrize(
    ("role", "payload", "match"),
    [
        pytest.param(
            None,
            msgpack.dumps({"type": "unknown", "transfer_id": "xfer"}),
            "Unhandled structured message type",
            id="unknown-structured-type",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [],
                }
            ),
            "block_notify_list cannot be empty",
            id="empty-remote-blocks",
        ),
        pytest.param(
            None,
            b"",
            "Unhandled message format",
            id="empty-completion",
        ),
    ],
)
def test_moriio_wrapper_rejects_invalid_messages(role, payload, match):
    wrapper = _wrapper_for_messages()
    if role is not None:
        set_role(role)
    wrapper._handle_completion_message = lambda msg: None

    with pytest.raises(MoRIIOError, match=match):
        wrapper._handle_message(payload)


def test_local_block_ids_longer_than_remote_raises_value_error():
    spec = _full_spec()
    (cache,) = _cache_views(spec)
    worker = _worker({"layer": cache}, {"layer": spec})

    with pytest.raises(ValueError, match="longer than remote_block_ids"):
        moriio_layout.compute_block_transfer_offsets(
            "layer", cache, worker.layer_to_spec, [1, 3], [4]
        )


def test_empty_local_block_ids_is_free_only_noop():
    spec = _full_spec()
    (cache,) = _cache_views(spec)
    worker = _worker({"layer": cache}, {"layer": spec})

    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, worker.layer_to_spec, [], [4, 5]
    ) == ([], [], [])


def test_registration_regions_cover_each_standardized_layer_span():
    full_spec = _full_spec()
    mla_spec = _mla_spec()
    layer_compact = _cache_views(full_spec)[0]
    block_interleaved = _cache_views(
        full_spec,
        num_layers=2,
        layout=KVCacheLayout.BLNHC,
    )[1]
    mla = _cache_views(mla_spec)[0]
    worker = _worker(
        {
            "layer_compact": layer_compact,
            "block_interleaved": block_interleaved,
            "mla": mla,
        },
        {
            "layer_compact": full_spec,
            "block_interleaved": full_spec,
            "mla": mla_spec,
        },
    )

    layer_compact_regions = moriio_layout.iter_layer_registration_regions(
        "layer_compact", layer_compact, worker.layer_to_spec
    )
    block_interleaved_regions = moriio_layout.iter_layer_registration_regions(
        "block_interleaved", block_interleaved, worker.layer_to_spec
    )
    mla_regions = moriio_layout.iter_layer_registration_regions(
        "mla", mla, worker.layer_to_spec
    )

    assert len(layer_compact_regions) == 1
    assert layer_compact_regions[0][0] is layer_compact
    assert layer_compact_regions[0][1] == 8 * 96
    assert len(block_interleaved_regions) == 1
    assert block_interleaved_regions[0][0] is block_interleaved
    assert block_interleaved_regions[0][1] == 7 * 192 + 96
    assert len(mla_regions) == 1
    assert mla_regions[0][0] is mla
    assert mla_regions[0][1] == 8 * 24


def test_registration_regions_use_layer_num_blocks():
    spec = _full_spec()
    (cache,) = _cache_views(spec, num_blocks=4)
    worker = _worker({"layer": cache}, {"layer": spec}, num_blocks=8)

    regions = moriio_layout.iter_layer_registration_regions(
        "layer", cache, worker.layer_to_spec
    )

    assert len(regions) == 1
    assert regions[0][1] == 4 * spec.page_size_bytes


def test_unsupported_shape_raises_value_error():
    # A 4-D view whose head/state/content dims all disagree with the spec (a
    # standardized [B, H, N, C] view for _full_spec would be (8, 2, 4, 6)).
    cache = torch.empty((8, 3, 5, 7), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    with pytest.raises(ValueError, match="Unsupported MoRIIO cache shape or strides"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, worker.layer_to_spec)


def test_non_block_dense_layout_raises_value_error():
    spec = _full_spec()
    (cache,) = _cache_views(spec, layout=KVCacheLayout.LHBNC)

    with pytest.raises(ValueError, match="Unsupported MoRIIO cache shape or strides"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, {"layer": spec})


def test_standardized_view_geometry_and_padded_registration():
    """Production views come from ``create_kv_cache_views``: geometry must track
    the [B, H, N, C] shape, and padded pages must register the full strided span
    (not just the meaningful block_len)."""
    num_blocks = 8
    spec = _full_spec()
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    (view,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBNHC)
    worker = _worker({"layer": view}, {"layer": spec})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", view, worker.layer_to_spec
    )
    assert geometry.num_blocks == num_blocks
    assert geometry.block_len == spec.page_size_bytes
    assert geometry.block_stride * view.element_size() == spec.page_size_bytes

    # MLA is just H == 1.
    mla = _mla_spec()
    mla_raw = torch.zeros(num_blocks * mla.page_size_bytes, dtype=torch.int8)
    (mla_view,) = dense_kv_cache_views(mla_raw, mla, num_blocks, 1, KVCacheLayout.LBNHC)
    mla_geometry = moriio_layout.get_layer_transfer_geometry(
        "mla", mla_view, {"mla": mla}
    )
    assert mla_geometry.block_len == mla.page_size_bytes

    # Alignment-padded page: registration must cover the padded stride.
    padded = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
        page_size_padded=64,
    )
    padded_raw = torch.zeros(num_blocks * padded.page_size_bytes, dtype=torch.int8)
    (padded_view,) = dense_kv_cache_views(
        padded_raw, padded, num_blocks, 1, KVCacheLayout.LBNHC
    )
    regions = moriio_layout.iter_layer_registration_regions(
        "padded", padded_view, {"padded": padded}
    )
    assert len(regions) == 1
    assert regions[0][1] == num_blocks * padded.page_size_bytes
