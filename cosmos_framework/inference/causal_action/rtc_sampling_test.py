# SPDX-License-Identifier: OpenMDW-1.1
"""RTC through packing, CFG, native solvers, cache selection and physical export."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.inference.causal_action import sampling
from cosmos_framework.inference.causal_action.outputs import real_actions
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.sampling_cache_test import _Model, _request, _Sampler, _spy_contexts
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig


class _Template:
    width = 3

    def validate_valid_mask(self, mask):
        assert mask.dtype == torch.bool and mask.shape == (3,)
        return mask

    def required_anchor_mask(self, mask):
        return mask

    def sanitize(self, values, mask):
        value = values.masked_fill(~mask, 0)
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite valid values")
        return value

    def encode_action_delta(self, target, anchor, mask):
        return (target - anchor).masked_fill(~mask, 0)

    def decode_action_delta(self, delta, anchor, mask):
        return (delta + anchor).masked_fill(~mask, 0)


class _CoupledModel(_Model):
    def __init__(self):
        super().__init__()
        self.net.language_model = SimpleNamespace(
            model=SimpleNamespace(
                rotary_emb=SimpleNamespace(rope_type="default"), config=SimpleNamespace(num_experts=0)
            )
        )
        self.metadata.template = _Template()
        self.metadata.anchors = self.metadata.states.clone()
        quantiles = Quantiles(
            torch.tensor([-1.0, -1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]), torch.tensor([True, True, False])
        )
        self.metadata.statistics = BlockStatistics(quantiles, quantiles)

    def denoise(self, *, net, data_batch_packed):
        packed = data_batch_packed
        v, a = packed.vision.tokens[0], packed.action.tokens[0]
        coefficient = 1.0 if packed.text_ids[0] == 11 else 0.5
        return {
            "preds_vision": [coefficient * (0.1 * v + 0.2 * a[-2:].mean())],
            "preds_action": [coefficient * (0.2 * a + 0.1 * v[:, :, -2:].mean())],
        }


class _ExactSampler:
    def __call__(self, velocity_fn, noise, **options):
        x = noise[0].clone()
        for sigma in (0.9007, 0.2003):
            if hasattr(velocity_fn, "set_integration_sigma"):
                velocity_fn.set_integration_sigma(sigma)
            x -= 0.2 * velocity_fn([x], torch.tensor([[int(sigma * 1000)]]))[0]
        return [x]


def _rtc_request(model, *, cfg=1.0):
    request = _request(model, custom_sampler=_ExactSampler())
    request["guidance"] = cfg
    request["initial_noise"] = [torch.linspace(-0.25, 0.25, 17)]
    request["data_batch"]["causal_action_rtc"] = RTCPrefix(
        torch.tensor([[3.5555556, 4.9999995, float("nan")], [3.3, 4.7, 0.0]]), 1, beta=1.0
    )
    return request


@pytest.mark.parametrize("cfg", [1.0, 3.0])
def test_rtc_guides_joint_targets_disables_value_cache_and_restores_physical_prefix(monkeypatch, cfg):
    model = _CoupledModel()
    events = _spy_contexts(monkeypatch, model)
    request = _rtc_request(model, cfg=cfg)
    result = sampling.sample_prepared(model, **request)
    assert ("legacy", False) in events
    assert not any(e[0] == "prefix" for e in events)
    assert any(e[0] == "plan" for e in events)
    assert not result["inference_cache"]["full_cache"]
    assert result["rtc"]["vjp_calls"] == 2
    assert result["rtc"]["delay_steps"] == 1
    assert all(p.grad is None for p in model.net.parameters())
    torch.testing.assert_close(result["vision"][0].flatten()[:3], torch.tensor([100.0, 101.0, 102.0]), rtol=0, atol=0)
    assert not result["action"][0][:2].any()  # No historical action targets enter the output.
    absolute = real_actions(result, {"sequence_plan": request["sequence_plans"]})[0]
    torch.testing.assert_close(absolute[0], torch.tensor([3.5555556, 4.9999995, 0.0]), rtol=0, atol=0)
    assert torch.isfinite(absolute).all() and not absolute[:, 2].any()


def test_rtc_changes_current_video_through_action_only_joint_vjp(monkeypatch):
    model = _CoupledModel()
    _spy_contexts(monkeypatch, model)
    request = _rtc_request(model)
    guided = sampling.sample_prepared(model, **request)
    prefix = request["data_batch"]["causal_action_rtc"]
    request["data_batch"]["causal_action_rtc"] = RTCPrefix(prefix.previous_actions, prefix.delay_steps, beta=0)
    plain = sampling.sample_prepared(model, **request)
    assert not torch.equal(guided["vision"][0][..., -2:, :, :], plain["vision"][0][..., -2:, :, :])
    assert not torch.equal(guided["action"][0][-1], plain["action"][0][-1])
    assert plain["rtc"]["vjp_calls"] == 0


def test_custom_rtc_solver_must_notify_actual_integration_sigma(monkeypatch):
    model = _CoupledModel()
    _spy_contexts(monkeypatch, model)
    request = _rtc_request(model)
    request["sampler"] = _Sampler()
    with pytest.raises(ValueError, match="set_integration_sigma"):
        sampling.sample_prepared(model, **request)


def test_rtc_rejects_inference_mode_and_preview(monkeypatch):
    model = _CoupledModel()
    _spy_contexts(monkeypatch, model)
    request = _rtc_request(model)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        sampling.sample_prepared(model, **request)
    request["data_batch"]["causal_action_preview"] = True
    with pytest.raises(ValueError, match="preview"):
        sampling.sample_prepared(model, **request)


def test_unipc_rtc_notifies_true_sigma_before_quantized_network_timestep():
    cfg = UniPCSamplerConfig(shift=5.0)
    sampler = UniPCSampler(cfg, tensor_kwargs={"device": "cpu"})
    observed_sigma, observed_t = [], []

    def velocity(noise, timestep):
        observed_t.append(float(timestep.item()))
        assert len(observed_sigma) == len(observed_t)
        return [0.1 * noise[0]]

    velocity.set_integration_sigma = lambda sigma: observed_sigma.append(float(sigma))
    sampler(velocity, [torch.ones(4)], num_steps=3, shift=5.0, seed=[42])
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=5.0, use_dynamic_shifting=False)
    scheduler.set_timesteps(3, device="cpu", shift=5.0)
    assert observed_sigma == [float(x) for x in scheduler.sigmas[:3]]
    assert any(abs(sigma - timestep / 1000) > 1e-5 for sigma, timestep in zip(observed_sigma, observed_t))


def test_fixed_step_notifies_actual_sigma():
    sampler = FixedStepSampler([0.73, 0.21])
    observed = []

    def velocity(noise, timestep):
        return [torch.zeros_like(noise[0])]

    velocity.set_integration_sigma = observed.append
    sampler(velocity, [torch.zeros(4)], seed=[42])
    assert observed == [0.73, 0.21]


def test_frozen_rtc_full_mode_routes_to_gradient_safe_prefix(monkeypatch):
    model = _CoupledModel()
    model.net.requires_grad_(False)
    _spy_contexts(monkeypatch, model)
    entered = []

    @contextmanager
    def gradient_cache(net):
        entered.append(net)
        yield {"entries": {}, "key": None, "stats": {"compact_calls": 2}}

    monkeypatch.setattr(sampling, "action_rtc_prefix_cache", gradient_cache)
    result = sampling.sample_prepared(model, **_rtc_request(model))
    assert entered == [model.net]
    assert result["inference_cache"]["full_cache"]
    assert result["inference_cache"]["gradient_safe_prefix"]
    assert result["rtc"]["gradient_safe_prefix"]
    assert not result["rtc"]["value_cache_disabled_for_input_vjp"]
    assert result["rtc"]["legacy_value_cache_disabled_for_input_vjp"]
    assert result["rtc"]["cache_fallback_reason"] is None
    assert result["rtc"]["vjp_calls"] == 2


@pytest.mark.parametrize("mode,kv", [("plans", True), ("off", True), ("full", False)])
def test_rtc_reference_flags_bypass_gradient_prefix_cache(monkeypatch, mode, kv):
    model = _CoupledModel()
    model.net.requires_grad_(False)
    events = _spy_contexts(monkeypatch, model)

    def unexpected(net):
        pytest.fail("Explicit RTC reference mode must not invoke the gradient prefix cache")

    monkeypatch.setattr(sampling, "action_rtc_prefix_cache", unexpected)
    request = _rtc_request(model)
    request["data_batch"]["causal_action_cache_mode"] = mode
    request["causal_use_kv_cache"] = kv
    result = sampling.sample_prepared(model, **request)
    assert ("legacy", False) in events
    assert not result["inference_cache"]["full_cache"]
    assert not result["inference_cache"]["gradient_safe_prefix"]
    assert result["rtc"]["vjp_calls"] == 2


def test_existing_eval_rtc_callers_with_trainable_flags_keep_reference_path(monkeypatch):
    model = _CoupledModel()
    _spy_contexts(monkeypatch, model)
    assert any(p.requires_grad for p in model.net.parameters())
    result = sampling.sample_prepared(model, **_rtc_request(model))
    assert not result["inference_cache"]["full_cache"]
    assert result["rtc"]["cache_fallback_reason"] == "gradient_prefix_cache_requires_frozen_weights"
    assert result["rtc"]["vjp_calls"] == 2
    assert all(p.grad is None for p in model.net.parameters())


@pytest.mark.parametrize("unsupported", ["rope", "moe", "language"])
def test_unsupported_gradient_prefix_configs_keep_rtc_reference(monkeypatch, unsupported):
    model = _CoupledModel()
    model.net.requires_grad_(False)
    _spy_contexts(monkeypatch, model)
    if unsupported == "rope":
        model.net.language_model.model.rotary_emb.rope_type = "dynamic"
        reason = "gradient_prefix_cache_requires_fixed_rope"
    elif unsupported == "moe":
        model.net.language_model.model.config.num_experts = 8
        reason = "gradient_prefix_cache_requires_dense_model"
    else:
        del model.net.language_model
        reason = "gradient_prefix_cache_requires_causal_language_model"

    def unexpected(net):
        pytest.fail("Unsupported cache configurations must retain the reference RTC path")

    monkeypatch.setattr(sampling, "action_rtc_prefix_cache", unexpected)
    result = sampling.sample_prepared(model, **_rtc_request(model))
    assert not result["inference_cache"]["gradient_safe_prefix"]
    assert result["rtc"]["cache_fallback_reason"] == reason
    assert result["rtc"]["vjp_calls"] == 2
