# SPDX-License-Identifier: OpenMDW-1.1
"""CPU simulation verifies captured-buffer ownership and cache admission."""

import gc
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.causal_action import npu_replay_graph as graph_module
from cosmos_framework.inference.causal_action.npu_replay_graph import NPUReplayGraphCache
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import TNDChunk, TNDPlan


def _copy_output(target, source):
    if isinstance(target, torch.Tensor):
        target.copy_(source)
    elif isinstance(target, dict):
        for key in target:
            _copy_output(target[key], source[key])
    elif isinstance(target, (tuple, list)):
        for left, right in zip(target, source, strict=True):
            _copy_output(left, right)
    else:
        assert target == source


class _FakeGraph:
    def __init__(self, forward, output):
        self.forward, self.output = forward, output
        self.fail = False

    def replay(self):
        if self.fail:
            raise RuntimeError("test replay failure")
        _copy_output(self.output, self.forward())


class _FakeBackend:
    def __init__(self):
        self.captures = 0
        self.synchronizations = 0
        self.fail_capture = False
        self.fail_sync = False
        self.last_graph = None

    def supports(self, tensor):
        return tensor.device.type == "cpu"

    def synchronize(self, device):
        self.synchronizations += 1
        if self.fail_sync:
            raise RuntimeError("test synchronize failure")

    def capture(self, forward, device):
        self.captures += 1
        if self.fail_capture:
            raise RuntimeError("test capture failure")
        output = forward()
        self.last_graph = _FakeGraph(forward, output)
        return self.last_graph, output, 0.125


@dataclass
class _Entry:
    layout: object
    gen_indexes: torch.Tensor
    packed_indexes: torch.Tensor
    plan: TNDPlan
    layers: dict


def _inputs(*, nq=2, nk=4, observation=1.0, packed=None, generated=None, query=None, keys=None):
    layout = SimpleNamespace(includes_action=True)
    pack = {
        "causal_seq": torch.empty((0, 2)),
        "full_only_seq": torch.arange(nq * 2, dtype=torch.float32).reshape(nq, 2),
        "is_sharded": False,
        "_full_indices": torch.arange(nq),
        "split_lens": [0, nq],
        "attn_modes": ["causal", "full"],
    }
    chunk = TNDChunk(
        torch.tensor(list(range(nq)) if query is None else query),
        torch.tensor(list(range(nk)) if keys is None else keys),
        (nq,),
        (nk if keys is None else len(keys),),
    )
    entry = _Entry(
        layout,
        torch.tensor(list(range(nq)) if generated is None else generated),
        torch.tensor(list(range(nk - nq, nk)) if packed is None else packed),
        TNDPlan((chunk,), nq, nk),
        {
            layer: (
                torch.full((nk, 1, 2), observation + layer),
                torch.full((nk, 1, 2), observation + 2 * layer),
            )
            for layer in range(2)
        },
    )
    return pack, SimpleNamespace(dense_mode="grouped_tnd", layout=layout), torch.arange(nq), entry


def _fixture():
    net = nn.Linear(2, 2, bias=False).eval().requires_grad_(False)
    with torch.no_grad():
        net.weight.copy_(torch.eye(2))
    backend = _FakeBackend()
    cache = NPUReplayGraphCache(net, _backend=backend)
    current = {}

    def forward(pack, *, attention_mask, position_ids, natten_metadata_list, memory):
        assert attention_mask.dense_mode == "grouped_tnd" and natten_metadata_list is None and memory is None
        bias = sum(key.mean() + value.mean() for key, value in current["entry"].layers.values())
        output = net(pack["full_only_seq"]) + bias + position_ids[:, None]
        return {"full_only_seq": output, "causal_seq": pack["causal_seq"].clone()}, {}

    def run(inputs, *, branch="cond"):
        current["entry"] = inputs[3]
        with torch.no_grad():
            return cache.run(forward, *inputs, (1, branch))

    return net, backend, cache, current, forward, run


def test_replay_updates_current_inputs_and_positions_without_recapture():
    net, backend, cache, _, _, run = _fixture()
    inputs = _inputs()
    first = run(inputs)[0]["full_only_seq"].clone()
    inputs[0]["full_only_seq"].add_(3)
    inputs[2].add_(2)
    second = run(inputs)[0]["full_only_seq"]
    torch.testing.assert_close(second, first + 5)
    assert backend.captures == 1
    assert cache.stats["replays"] == 2 and cache.stats["requests"] == 1
    assert cache.stats["capture_seconds"] == 0.125


def test_new_request_refreshes_every_layer_kv_without_stale_history():
    net, backend, cache, _, _, run = _fixture()
    first = _inputs(observation=1.0)
    old = run(first)[0]["full_only_seq"].clone()
    second = _inputs(observation=4.0)
    expected_layers = {i: (key.clone(), value.clone()) for i, (key, value) in second[3].layers.items()}
    result = run(second)[0]["full_only_seq"]
    torch.testing.assert_close(result, old + 12)
    for layer, (key, value) in expected_layers.items():
        assert torch.equal(second[3].layers[layer][0], key)
        assert torch.equal(second[3].layers[layer][1], value)
    assert backend.captures == 1 and cache.stats["refreshes"] == 1


@pytest.mark.parametrize(
    "change",
    ["packed", "generated", "query", "keys", "pack_metadata", "shape", "branch", "weight_version", "weight_pointer"],
)
def test_exact_signature_changes_invalidate_single_graph(change):
    net, backend, cache, _, _, run = _fixture()
    run(_inputs())
    second = _inputs(nq=3, nk=5) if change == "shape" else _inputs()
    if change == "packed":
        second[3].packed_indexes = second[3].packed_indexes.flip(0)
    elif change == "generated":
        second[3].gen_indexes = second[3].gen_indexes.flip(0)
    elif change == "query":
        old = second[3].plan.chunks[0]
        second[3].plan = TNDPlan(
            (TNDChunk(old.query_indexes.flip(0), old.kv_indexes, old.query_ends, old.kv_ends),), 2, 4
        )
    elif change == "keys":
        old = second[3].plan.chunks[0]
        second[3].plan = TNDPlan(
            (TNDChunk(old.query_indexes, old.kv_indexes.flip(0), old.query_ends, old.kv_ends),), 2, 4
        )
    elif change == "pack_metadata":
        second[0]["_full_indices"] = second[0]["_full_indices"].flip(0)
    elif change == "weight_version":
        with torch.no_grad():
            net.weight.add_(1)
    elif change == "weight_pointer":
        net.weight = nn.Parameter(net.weight.clone(), requires_grad=False)
    run(second, branch="uncond" if change == "branch" else "cond")
    assert backend.captures == 2
    assert backend.synchronizations == 1
    assert cache.stats["invalidations"] == 1 and cache.stats["refreshes"] == 0
    assert cache.stats["has_graph"]


def test_admission_copies_indexes_to_cpu_once_per_entry(monkeypatch):
    net, backend, cache, _, _, run = _fixture()
    calls = []
    original = graph_module._index_values

    def counted(tensor):
        calls.append(tensor)
        return original(tensor)

    monkeypatch.setattr(graph_module, "_index_values", counted)
    inputs = _inputs()
    run(inputs)
    for _ in range(3):
        run(inputs)
    assert len(calls) == 4
    run(_inputs())
    assert len(calls) == 8


def test_same_entry_position_dtype_change_recaptures_without_silent_cast():
    net, backend, cache, _, _, run = _fixture()
    inputs = _inputs()
    run(inputs)
    changed = (inputs[0], inputs[1], inputs[2].to(torch.int32), inputs[3])
    run(changed)
    assert backend.captures == 2 and cache.stats["invalidations"] == 1


def test_capture_owner_lives_through_refresh_and_is_released_on_close():
    net, backend, cache, current, _, run = _fixture()
    inputs = _inputs()
    captured_owner = weakref.ref(inputs[3])
    run(inputs)
    del inputs
    run(_inputs(observation=7.0))
    gc.collect()
    assert captured_owner() is not None
    cache.close()
    gc.collect()
    assert captured_owner() is None
    assert cache.stats["closed"] and not cache.stats["has_graph"]
    assert backend.synchronizations == 1
    with pytest.raises(RuntimeError, match="closed"):
        run(_inputs())


def test_gradients_are_rejected_before_capture():
    net, backend, cache, current, forward, _ = _fixture()
    inputs = _inputs()
    current["entry"] = inputs[3]
    with torch.enable_grad(), pytest.raises(RuntimeError, match="RTC input-VJP"):
        cache.run(forward, *inputs, (1, "cond"))
    assert backend.captures == 0


def test_native_backend_rejects_cpu_tensors():
    net, _, _, current, forward, _ = _fixture()
    cache = NPUReplayGraphCache(net)
    inputs = _inputs()
    current["entry"] = inputs[3]
    with torch.no_grad(), pytest.raises(ValueError, match="NPU compact input"):
        cache.run(forward, *inputs, (1, "cond"))


@pytest.mark.parametrize("failure", ["capture", "replay"])
def test_failed_capture_or_replay_clears_buffers_and_can_retry(failure):
    net, backend, cache, _, _, run = _fixture()
    if failure == "capture":
        backend.fail_capture = True
    else:
        run(_inputs())
        backend.last_graph.fail = True
    with pytest.raises(RuntimeError, match="cache cleared"):
        run(_inputs())
    assert not cache.stats["has_graph"] and cache.stats["failures"] == 1
    assert cache._capture_entry is None and cache._layers is None
    backend.fail_capture = False
    run(_inputs())
    assert cache.stats["has_graph"]


def test_close_clears_ownership_even_if_device_synchronization_fails():
    net, backend, cache, _, _, run = _fixture()
    run(_inputs())
    backend.fail_sync = True
    with pytest.raises(RuntimeError, match="synchronize while closing"):
        cache.close()
    assert not cache.stats["has_graph"] and cache._capture_entry is None


@pytest.mark.parametrize("invalid", ["training", "sharded", "wrong_layout", "bad_indexes", "missing_kv"])
def test_unsupported_inputs_fail_before_capturing(invalid):
    net, backend, cache, _, _, run = _fixture()
    inputs = _inputs()
    if invalid == "training":
        net.train()
    elif invalid == "sharded":
        inputs[0]["is_sharded"] = True
    elif invalid == "wrong_layout":
        inputs[1].layout = SimpleNamespace(includes_action=True)
    elif invalid == "bad_indexes":
        inputs[3].packed_indexes[0] = 99
    else:
        inputs[3].layers.clear()
    with pytest.raises(ValueError):
        run(inputs)
    assert backend.captures == 0


def test_stats_are_snapshots():
    net, _, cache, _, _, run = _fixture()
    run(_inputs())
    snapshot = cache.stats
    snapshot["captures"] = 999
    assert cache.stats["captures"] == 1
