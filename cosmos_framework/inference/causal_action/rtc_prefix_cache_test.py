# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-prefix RTC cache preserves current-token values and input Jacobians."""

import copy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_mode_splits, get_gen_seq, get_und_seq
from cosmos_framework.inference.causal_action.prefix_cache_test import (
    _assert_current_outputs,
    _network,
    _next_step,
    _sequence,
)
from cosmos_framework.inference.causal_action.rtc_prefix_cache import (
    _fixed_dependency_closure,
    action_rtc_prefix_cache,
)
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import TNDChunk, TNDPlan


@pytest.fixture(autouse=True)
def _cpu_attention(monkeypatch):
    from cosmos_framework.model.attention import attention

    def cpu_attention(*args, backend, **kwargs):
        del backend
        if args[0].dtype == torch.float64:
            # The frontend's backend policy excludes FP64, although PyTorch
            # SDPA supports it. Use that same primitive as an FP64 reference.
            query, key, value = args[:3]
            qends = kwargs.get("cumulative_seqlen_Q")
            kends = kwargs.get("cumulative_seqlen_KV")
            qends = qends.tolist() if qends is not None else [0, query.shape[1]]
            kends = kends.tolist() if kends is not None else [0, key.shape[1]]
            repeats = query.shape[2] // key.shape[2]
            parts = []
            for q0, q1, k0, k1 in zip(qends[:-1], qends[1:], kends[:-1], kends[1:], strict=True):
                parts.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        query[:, q0:q1].transpose(1, 2),
                        key[:, k0:k1].repeat_interleave(repeats, dim=2).transpose(1, 2),
                        value[:, k0:k1].repeat_interleave(repeats, dim=2).transpose(1, 2),
                        is_causal=kwargs.get("is_causal", False),
                        scale=kwargs.get("scale"),
                    ).transpose(1, 2)
                )
            return torch.cat(parts, dim=1)
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)


def _frozen_network(**kwargs):
    return _network(**kwargs).requires_grad_(False)


def _capture_lm_inputs(network, sequence, dtype):
    captured = {}
    original = network.language_model.forward

    def capture(pack, **kwargs):
        captured["pack"] = from_mode_splits(
            get_und_seq(pack).detach().clone().to(dtype), get_gen_seq(pack).detach().clone().to(dtype), pack
        )
        captured["kwargs"] = kwargs
        return original(pack, **kwargs)

    network.language_model.forward = capture
    try:
        with torch.no_grad():
            network(sequence)
    finally:
        network.language_model.forward = original
    network.language_model.to(dtype)
    # Production RoPE deliberately keeps inv_freq in FP32 even with FP64 hidden states.
    network.language_model.model.rotary_emb.float()
    return captured["pack"], captured["kwargs"]


def _lm_function(network, base, kwargs, z, *, block=2):
    layout = kwargs["attention_mask"].layout
    indexes = torch.where(
        (layout.stream_ids[layout.gen_query_indexes] == 1) & (layout.block_ids[layout.gen_query_indexes] == block)
    )[0]
    active = get_gen_seq(base).index_select(0, indexes)
    directions = torch.randn((3, *active.shape), generator=torch.Generator().manual_seed(91), dtype=z.dtype) * 0.15
    active = active + (z[:, None, None] * directions).sum(0)
    pack = from_mode_splits(get_und_seq(base), get_gen_seq(base).index_copy(0, indexes, active), base)
    output, _ = network.language_model(pack, **kwargs)
    current = get_gen_seq(output).index_select(0, indexes).flatten()
    return torch.stack((current[0] + current[3].square() * 0.1, current[10] * current[20], current[-1].sin()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalized_und_keys", [False, True])
def test_two_step_nonlinear_lm_jacobians_and_vjps_match_full_forward(dtype, normalized_und_keys):
    network = _frozen_network(normalized_und_keys=normalized_und_keys)
    base, kwargs = _capture_lm_inputs(network, _sequence(), dtype)
    zs = [torch.tensor([0.2, -0.4, 0.7], dtype=dtype), torch.tensor([-0.5, 0.3, -0.2], dtype=dtype)]
    expected = []
    for z in zs:
        expected.append(
            (
                _lm_function(network, base, kwargs, z).detach(),
                torch.autograd.functional.jacobian(lambda value: _lm_function(network, base, kwargs, value), z),
            )
        )
    atol, rtol = (3e-6, 2e-5) if dtype == torch.float32 else (3e-8, 2e-6)
    with action_rtc_prefix_cache(network) as controller:
        controller["key"] = (2, "cond")
        snapshots = None
        for z, (expected_output, expected_jacobian) in zip(zs, expected, strict=True):
            actual = _lm_function(network, base, kwargs, z)
            jacobian = torch.autograd.functional.jacobian(lambda value: _lm_function(network, base, kwargs, value), z)
            torch.testing.assert_close(actual, expected_output, atol=atol, rtol=rtol)
            torch.testing.assert_close(jacobian, expected_jacobian, atol=atol, rtol=rtol)
            error = z.new_tensor([0.7, -0.9, 0.4])
            x = z.clone().requires_grad_(True)
            vjp = torch.autograd.grad(_lm_function(network, base, kwargs, x), x, grad_outputs=error)[0]
            torch.testing.assert_close(vjp, expected_jacobian.T @ error, atol=atol, rtol=rtol)
            entry = controller["entries"][(2, "cond")]
            assert not entry.fixed_inputs.requires_grad and entry.fixed_inputs.grad_fn is None
            assert not entry.fixed_positions.requires_grad and entry.fixed_positions.grad_fn is None
            assert not (entry.layout.stream_ids[entry.fixed_indexes] == 1).any()
            assert entry.plan.num_keys == len(entry.fixed_indexes) + len(entry.packed_indexes)
            assert entry.plan.num_keys < kwargs["attention_mask"].tnd_plan.num_keys
            for values in entry.layers.values():
                assert all(not value.requires_grad and value.grad_fn is None for value in values)
            if snapshots is None:
                snapshots = {index: tuple(value.clone() for value in values) for index, values in entry.layers.items()}
            else:
                assert all(
                    torch.equal(value, snapshots[index][i])
                    for index, values in entry.layers.items()
                    for i, value in enumerate(values)
                )
        assert controller["stats"]["prefill_calls"] == 1
        assert controller["stats"]["compact_calls"] == 6
        assert controller["stats"]["grad_compact_calls"] == 6
    assert not controller["entries"]
    assert all(parameter.grad is None for parameter in network.parameters())


def _current_noise(sequence, block):
    vision = sequence.vision.tokens[0]
    start, end = 1 + 2 * block, 3 + 2 * block
    vi = torch.arange(vision.numel()).reshape(vision.shape)[:, :, start:end].flatten()
    action = sequence.action.tokens[0]
    ai = torch.arange(action.numel()).reshape(action.shape)[4 * block : 4 * (block + 1), :-1].flatten()
    x = torch.cat((vision.flatten()[vi], action.flatten()[ai])).detach()
    return x, vi, ai


def _network_function(network, base, noise, vi, ai, block):
    sequence = copy.deepcopy(base)
    sequence.teacher_forcing.layout = base.teacher_forcing.layout
    v = sequence.vision.tokens[0]
    a = sequence.action.tokens[0]
    sequence.vision.tokens = [v.flatten().index_copy(0, vi, noise[: len(vi)]).reshape(v.shape)]
    sequence.action.tokens = [a.flatten().index_copy(0, ai, noise[len(vi) :]).reshape(a.shape)]
    result = network(sequence)
    start, end = 1 + 2 * block, 3 + 2 * block
    return torch.cat(
        (
            result["preds_vision"][0][:, :, start:end].flatten(),
            result["preds_action"][0][4 * block : 4 * (block + 1), :-1].flatten(),
        )
    )


@pytest.mark.parametrize("block", [0, 2])
def test_real_video_action_encoders_heads_and_joint_vjp_match_two_steps(block):
    network = _frozen_network(normalized_und_keys=True)
    first = _sequence(block)
    steps = [first, _next_step(first)]
    expected = []
    for sequence in steps:
        noise, vi, ai = _current_noise(sequence, block)
        x = noise.clone().requires_grad_(True)
        velocity = _network_function(network, sequence, x, vi, ai, block)
        error = torch.linspace(-0.5, 0.7, len(velocity))
        gradient = torch.autograd.grad(velocity, x, grad_outputs=error)[0]
        expected.append((velocity.detach(), gradient, error))
    with action_rtc_prefix_cache(network) as controller:
        controller["key"] = (block, "cond")
        for sequence, (velocity_reference, gradient_reference, error) in zip(steps, expected, strict=True):
            noise, vi, ai = _current_noise(sequence, block)
            x = noise.clone().requires_grad_(True)
            velocity = _network_function(network, sequence, x, vi, ai, block)
            gradient = torch.autograd.grad(velocity, x, grad_outputs=error)[0]
            torch.testing.assert_close(velocity, velocity_reference, atol=3e-6, rtol=2e-5)
            torch.testing.assert_close(gradient, gradient_reference, atol=3e-6, rtol=2e-5)


def test_first_call_prefills_without_grad_then_always_runs_compact_including_no_grad_calls():
    network = _frozen_network(normalized_und_keys=True)
    sequence = _sequence()
    with torch.no_grad():
        reference = network(sequence)
    calls = {}
    handles = []
    for layer_index, layer in enumerate(network.language_model.model.layers):
        for name, module in (("q", layer.self_attn.q_proj_moe_gen), ("ffn", layer.mlp_moe_gen)):
            key = (layer_index, name)
            calls[key] = []
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, key=key: calls[key].append((len(args[0]), torch.is_grad_enabled()))
                )
            )
    try:
        with torch.no_grad(), action_rtc_prefix_cache(network) as controller:
            controller["key"] = (2, "cond")
            first, second = network(sequence), network(sequence)
            _assert_current_outputs(first, reference, sequence, 2)
            _assert_current_outputs(second, reference, sequence, 2)
            assert controller["stats"]["compact_calls"] == 2
            assert controller["stats"]["grad_compact_calls"] == 0
    finally:
        for handle in handles:
            handle.remove()
    full = len(sequence.teacher_forcing.layout.gen_query_indexes)
    assert all(values == [(full, False), (12, False), (12, False)] for values in calls.values())


def test_cfg_isolation_fixed_input_guards_clear_and_new_request_rebuild_state():
    network = _frozen_network(normalized_und_keys=True)
    cond, uncond = _sequence(), _sequence(num_text=1)
    uncond.text_ids.fill_(43)
    changed = copy.deepcopy(cond)
    changed.teacher_forcing.layout = cond.teacher_forcing.layout
    changed.teacher_forcing.state_tokens[0][2] += 0.5
    with torch.no_grad():
        references = {name: network(sequence) for name, sequence in (("cond", cond), ("uncond", uncond))}
        changed_reference = network(changed)
    for request in range(2):
        with action_rtc_prefix_cache(network) as controller:
            if request:
                controller["key"] = (2, "cond")
                _assert_current_outputs(network(changed), changed_reference, changed, 2)
                continue
            for name, sequence in (("cond", cond), ("uncond", uncond), ("cond", cond), ("uncond", uncond)):
                controller["key"] = (2, name)
                _assert_current_outputs(network(sequence), references[name], sequence, 2)
            assert len(controller["entries"]) == 2
            controller["key"] = (2, "cond")
            with pytest.raises(ValueError, match="fixed prefix inputs"):
                network(changed)
            controller["entries"].clear()
            _assert_current_outputs(network(changed), changed_reference, changed, 2)


def test_exception_restores_all_dispatches_and_drops_graph_free_cache():
    network = _frozen_network()
    language_model = network.language_model
    original_forward = language_model.forward
    originals = [layer.self_attn.dispatch_attention_fn for layer in language_model.model.layers]
    calls = []

    def fail_on_compact(module, args):
        calls.append(len(args[0]))
        if len(calls) == 2:
            raise RuntimeError("intentional compact failure")

    handle = language_model.model.layers[0].self_attn.q_proj_moe_gen.register_forward_pre_hook(fail_on_compact)
    try:
        with pytest.raises(RuntimeError, match="intentional compact failure"):
            with action_rtc_prefix_cache(network) as controller:
                controller["key"] = (2, "cond")
                network(_sequence())
    finally:
        handle.remove()
    assert not controller["entries"]
    assert language_model.forward == original_forward
    assert [layer.self_attn.dispatch_attention_fn for layer in language_model.model.layers] == originals
    assert not hasattr(language_model, "_rtc_prefix_cache_active")
    with torch.no_grad():
        assert torch.isfinite(network(_sequence())["preds_action"][0]).all()


@pytest.mark.parametrize("bad", ["training", "unfrozen", "rope", "moe", "inference_mode"])
def test_unsafe_model_modes_are_rejected(bad):
    network = _frozen_network()
    if bad == "training":
        network.train()
    elif bad == "unfrozen":
        next(network.parameters()).requires_grad_(True)
    elif bad == "rope":
        network.language_model.model.rotary_emb.rope_type = "dynamic"
    elif bad == "moe":
        network.language_model.model.config.num_experts = 2
    if bad == "inference_mode":
        with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
            with action_rtc_prefix_cache(network):
                pass
    else:
        with pytest.raises(ValueError):
            with action_rtc_prefix_cache(network):
                pass


def test_nested_contexts_layout_changes_positions_and_weight_mutation_are_rejected():
    network = _frozen_network()
    base, kwargs = _capture_lm_inputs(network, _sequence(), torch.float32)
    z = torch.zeros(3)
    with action_rtc_prefix_cache(network) as controller:
        controller["key"] = (2, "cond")
        _lm_function(network, base, kwargs, z)
        with pytest.raises(RuntimeError, match="nested"):
            with action_rtc_prefix_cache(network):
                pass
        shifted = dict(kwargs, position_ids=kwargs["position_ids"] + 1)
        with pytest.raises(ValueError, match="inputs/positions"):
            _lm_function(network, base, shifted, z)
        with torch.no_grad():
            next(network.parameters()).add_(0.01)
        with pytest.raises(ValueError, match="weights changed"):
            _lm_function(network, base, kwargs, z)


def test_fixed_dependency_closure_rejects_indirect_current_noisy_dependency():
    layout = SimpleNamespace(stream_ids=torch.tensor([-1, 0, 1, 3]), gen_query_indexes=torch.tensor([1, 2, 3]))
    full = TNDPlan(
        (TNDChunk(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 3]), (1, 2, 3), (3, 7, 9)),),
        3,
        4,
    )
    current = TNDPlan((TNDChunk(torch.tensor([0]), torch.tensor([0, 1, 2, 3]), (1,), (4,)),), 1, 4)
    with pytest.raises(ValueError, match="Fixed prefix.*noisy"):
        _fixed_dependency_closure(layout, full, current, torch.tensor([2]))


def test_fixed_dependency_closure_rejects_direct_historical_noisy_dependency():
    layout = SimpleNamespace(stream_ids=torch.tensor([-1, 0, 1, 3, 1]), gen_query_indexes=torch.tensor([1, 2, 3, 4]))
    full = TNDPlan((), 4, 5)
    current = TNDPlan((TNDChunk(torch.tensor([0]), torch.tensor([0, 1, 2, 3, 4]), (1,), (5,)),), 1, 5)
    with pytest.raises(ValueError, match="historical noisy"):
        _fixed_dependency_closure(layout, full, current, torch.tensor([2]))
