# SPDX-License-Identifier: OpenMDW-1.1
"""One bounded, model-local NPU graph for no-grad compact prefix replay.

Prefill remains request-local. Compatible requests refresh every fixed K/V
buffer before replaying the captured language model. Weights and layout must
remain unchanged within a request, as required by the ordinary prefix cache.
This cache is deliberately excluded from RTC input-VJP and training paths.
"""

import hashlib
import time
import weakref

import torch


def _tensor_description(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _index_values(tensor):
    """Only called while admitting a new entry, never on each denoising step."""
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError("NPU replay graph indexes must be one-dimensional integers")
    return tuple(tensor.detach().cpu().tolist())


def _metadata_value(value):
    if isinstance(value, torch.Tensor):
        return (
            _tensor_description(value),
            tuple(value.detach().cpu().reshape(-1).tolist()),
            getattr(value, "_cosmos_actual_seq_lengths", None),
        )
    if isinstance(value, dict):
        return tuple((key, _metadata_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported NPU replay pack metadata: {type(value).__name__}")


class _NPUBackend:
    def supports(self, tensor):
        return tensor.device.type == "npu"

    def synchronize(self, device):
        torch.npu.synchronize(device)

    def capture(self, forward, device):
        # Import only on the hardware path so CPU inference/tests stay usable.
        import torch_npu  # noqa: F401

        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            for _ in range(3):
                forward()
        torch.npu.current_stream(device).wait_stream(stream)
        self.synchronize(device)
        started = time.perf_counter()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = forward()
        self.synchronize(device)
        return graph, output, time.perf_counter() - started


class NPUReplayGraphCache:
    """Capture at most one exact layout, retaining all captured tensor owners.

    ``run`` must be called serially on one eval-mode model. A new ``entry`` is
    a request boundary: its complete prefilled K/V is copied into the captured
    buffers. Exact CPU signatures are computed once at that boundary. Ordinary
    replay never copies index tensors to CPU or checks all model parameters.

    ``_backend`` is an internal test seam; production uses the native NPU API.
    """

    def __init__(self, net, *, _backend=None):
        self._net = weakref.ref(net)
        self._backend = _backend if _backend is not None else _NPUBackend()
        self._graph = None
        self._signature = None
        self._input_signature = None
        self._entry = None
        self._capture_entry = None
        self._pack = self._positions = self._mask = self._layers = self._output = None
        self._device = None
        self._closed = False
        self._stats = {
            "captures": 0,
            "replays": 0,
            "requests": 0,
            "refreshes": 0,
            "invalidations": 0,
            "failures": 0,
            "capture_seconds": 0.0,
            "last_capture_seconds": 0.0,
            "last_invalidation": None,
            "signature": None,
        }

    @property
    def stats(self):
        return {**self._stats, "has_graph": self._graph is not None, "closed": self._closed}

    def _validate(self, pack, mask, positions, entry, cache_key):
        if self._closed:
            raise RuntimeError("NPU replay graph cache is closed")
        if torch.is_grad_enabled():
            raise RuntimeError("NPU replay graphs require no-grad inference; use eager execution for RTC input-VJP")
        net = self._net()
        if net is None or net.training:
            raise ValueError("NPU replay graphs require a live eval-mode model")
        config = getattr(getattr(getattr(net, "language_model", None), "model", None), "config", None)
        if getattr(config, "num_experts", 0):
            raise ValueError("NPU replay graphs currently require a dense language model")
        if pack.get("is_sharded", False):
            raise ValueError("NPU replay graphs require an unsharded compact pack")
        if getattr(mask, "dense_mode", None) != "grouped_tnd" or not getattr(mask.layout, "includes_action", False):
            raise ValueError("NPU replay graphs require a causal action grouped_tnd layout")
        if mask.layout is not entry.layout:
            raise ValueError("NPU replay graph entry and attention layout disagree")
        if not isinstance(cache_key, tuple) or len(cache_key) != 2:
            raise ValueError("NPU replay graphs require a (block, CFG branch) cache key")
        for field in ("causal_seq", "full_only_seq"):
            if (
                field not in pack
                or not isinstance(pack[field], torch.Tensor)
                or not self._backend.supports(pack[field])
            ):
                raise ValueError("NPU replay graphs require NPU compact input tensors")
        if pack["causal_seq"].shape[0] != 0 or pack["full_only_seq"].shape[0] == 0:
            raise ValueError("NPU replay graphs require nonempty current GEN queries and no UND queries")
        device = pack["full_only_seq"].device
        if not isinstance(positions, torch.Tensor) or positions.device != device:
            raise ValueError("NPU replay positions must be a tensor on the compact input device")
        if len(entry.plan.chunks) != 1 or len(entry.plan.chunks[0].query_ends) != 1:
            raise ValueError("NPU replay graphs currently require one current attention group")
        if entry.plan.num_queries != pack["full_only_seq"].shape[0]:
            raise ValueError("NPU replay graph query count does not match its compact pack")
        if not entry.layers:
            raise ValueError("NPU replay graphs require prefilled K/V for every decoder layer")
        for key, value in entry.layers.values():
            if key.device != device or value.device != device or key.shape != value.shape:
                raise ValueError("NPU replay K/V must share their shape and compact input device")
            if key.shape[0] != entry.plan.num_keys:
                raise ValueError("NPU replay K/V length does not match its attention plan")

    def _entry_signature(self, pack, positions, entry, cache_key):
        chunk = entry.plan.chunks[0]
        queries, keys = _index_values(chunk.query_indexes), _index_values(chunk.kv_indexes)
        packed, generated = _index_values(entry.packed_indexes), _index_values(entry.gen_indexes)
        nquery, nkey = entry.plan.num_queries, entry.plan.num_keys
        if (
            len(queries) != nquery
            or chunk.query_ends != (len(queries),)
            or chunk.kv_ends != (len(keys),)
            or len(packed) != nquery
            or len(generated) != nquery
            or any(not 0 <= index < nquery for index in queries)
            or any(not 0 <= index < nkey for index in keys + packed)
        ):
            raise ValueError("Invalid indexes or sequence ends in NPU replay graph layout")
        data_fields = {"causal_seq", "full_only_seq"}
        pack_signature = tuple(
            (name, _tensor_description(value) if name in data_fields else _metadata_value(value))
            for name, value in sorted(pack.items())
        )
        net = self._net()
        weights = tuple(
            (kind, name, _tensor_description(tensor), tensor.data_ptr(), tensor._version)
            for kind, tensors in (("parameter", net.named_parameters()), ("buffer", net.named_buffers()))
            for name, tensor in tensors
        )
        return (
            cache_key,
            (queries, keys, chunk.query_ends, chunk.kv_ends, packed, generated, nquery, nkey),
            pack_signature,
            _tensor_description(positions),
            tuple(
                (layer, _tensor_description(key), _tensor_description(value))
                for layer, (key, value) in sorted(entry.layers.items())
            ),
            weights,
            torch.is_autocast_enabled(positions.device.type),
            str(torch.get_autocast_dtype(positions.device.type)),
            torch.is_inference_mode_enabled(),
        )

    def _input_description(self, compact, positions, entry, cache_key):
        # Shape/dtype/autocast changes can be checked without reading device
        # values. Device indexes themselves remain immutable within an entry.
        return (
            cache_key,
            _tensor_description(compact["causal_seq"]),
            _tensor_description(compact["full_only_seq"]),
            _tensor_description(positions),
            id(entry.plan),
            torch.is_autocast_enabled(positions.device.type),
            str(torch.get_autocast_dtype(positions.device.type)),
            torch.is_inference_mode_enabled(),
        )

    def _clear(self, reason):
        error = None
        try:
            if self._device is not None:
                self._backend.synchronize(self._device)
        except Exception as exc:
            error = exc
        finally:
            self._graph = self._signature = self._entry = self._capture_entry = None
            self._input_signature = None
            self._pack = self._positions = self._mask = self._layers = self._output = None
            self._device = None
            self._stats["signature"] = None
            self._stats["last_invalidation"] = reason
        return error

    def close(self):
        """Synchronize pending replay work and release the single captured graph."""
        error = self._clear("closed")
        self._closed = True
        if error is not None:
            raise RuntimeError("Could not synchronize while closing the NPU replay graph") from error

    def run(self, original_forward, compact, mask, positions, entry, cache_key):
        self._validate(compact, mask, positions, entry, cache_key)
        # Signature construction reads CPU indexes and weight versions only at
        # a new request boundary. They are immutable throughout one request.
        input_signature = self._input_description(compact, positions, entry, cache_key)
        new_entry = entry is not self._entry or input_signature != self._input_signature
        signature = self._entry_signature(compact, positions, entry, cache_key) if new_entry else self._signature
        if new_entry:
            self._stats["requests"] += 1
        if self._graph is not None and signature != self._signature:
            self._stats["invalidations"] += 1
            error = self._clear("layout_or_weights_changed")
            if error is not None:
                raise RuntimeError("Could not synchronize while replacing the NPU replay graph") from error
        try:
            if self._graph is None:
                self._device = compact["full_only_seq"].device
                self._signature = signature
                self._input_signature = input_signature
                self._entry = self._capture_entry = entry
                self._mask = mask
                self._layers = {
                    layer: (key.detach().clone(), value.detach().clone())
                    for layer, (key, value) in entry.layers.items()
                }
                entry.layers = self._layers
                self._pack = dict(compact)
                for field in ("causal_seq", "full_only_seq"):
                    self._pack[field] = compact[field].detach().clone()
                self._positions = positions.detach().clone()

                def forward():
                    return original_forward(
                        self._pack,
                        attention_mask=self._mask,
                        position_ids=self._positions,
                        natten_metadata_list=None,
                        memory=None,
                    )

                self._graph, self._output, elapsed = self._backend.capture(forward, self._device)
                self._stats["captures"] += 1
                self._stats["capture_seconds"] += elapsed
                self._stats["last_capture_seconds"] = elapsed
                self._stats["signature"] = hashlib.sha256(repr(signature).encode()).hexdigest()
            elif new_entry:
                for layer, (key, value) in entry.layers.items():
                    self._layers[layer][0].copy_(key)
                    self._layers[layer][1].copy_(value)
                entry.layers = self._layers
                self._entry = entry
                self._input_signature = input_signature
                self._stats["refreshes"] += 1
            for field in ("causal_seq", "full_only_seq"):
                self._pack[field].copy_(compact[field])
            self._positions.copy_(positions)
            self._graph.replay()
            self._stats["replays"] += 1
            return self._output
        except Exception as exc:
            self._stats["failures"] += 1
            cleanup_error = self._clear("capture_or_replay_failed")
            detail = " Cleanup synchronization also failed." if cleanup_error is not None else ""
            raise RuntimeError(f"NPU replay graph capture or replay failed; cache cleared.{detail}") from exc
