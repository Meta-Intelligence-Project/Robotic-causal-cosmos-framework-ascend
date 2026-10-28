# SPDX-License-Identifier: OpenMDW-1.1
"""RTC sampling with real causal encoders, transformer input VJPs and solvers."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action import sampling
from cosmos_framework.inference.causal_action.outputs import real_actions
from cosmos_framework.inference.causal_action.prefix_cache_test import _SmallLanguageModel
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.rtc_sampling_test import _CoupledModel
from cosmos_framework.inference.causal_action.sampling_cache_test import _request
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


@pytest.fixture(autouse=True)
def _cpu_packing_and_attention(monkeypatch):
    from cosmos_framework.model.attention import attention

    def cpu_attention(*args, backend, **kwargs):
        del backend
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)


class _RealModel(_CoupledModel):
    """Reuse small packing geometry, replacing every denoise with the real net."""

    def __init__(self, solver):
        super().__init__()
        torch.manual_seed(71)
        config = Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        )
        language_model = _SmallLanguageModel(config, normalized_und_keys=True)
        self.net = (
            CausalActionNetwork(
                language_model,
                Cosmos3VFMNetworkConfig(
                    vlm_config=SimpleNamespace(text_config=config),
                    vision_gen=True,
                    action_gen=True,
                    sound_gen=False,
                    action_dim=3,
                    latent_patch_size=1,
                    latent_channel_size=1,
                    max_latent_h=1,
                    max_latent_w=1,
                    max_latent_t=5,
                    joint_attn_implementation="teacher_forcing",
                    teacher_forcing_dense_mode="grouped_tnd",
                    timestep_scale=1.0,
                ),
            )
            .eval()
            .requires_grad_(False)
        )
        with torch.no_grad():
            for layer in language_model.model.layers:
                layer.self_attn.k_norm_und_for_gen.weight.copy_(torch.linspace(0.5, 2.0, config.head_dim))
        self.sampler = (
            UniPCSampler(UniPCSamplerConfig(shift=5.0), tensor_kwargs={"device": "cpu"})
            if solver == "unipc"
            else FixedStepSampler([0.9007, 0.6003, 0.2001])
        )

    def denoise(self, *, net, data_batch_packed):
        return (net or self.net)(data_batch_packed)


def _real_request(model, *, cache_mode="full", cfg=1.0, rtc=True):
    request = _request(model, cache_mode=cache_mode)
    request["guidance"] = cfg
    request["num_steps"] = 3
    request["initial_noise"] = [torch.randn(17, generator=torch.Generator().manual_seed(23)) * 0.2]
    video = torch.linspace(-0.2, 0.2, 5).reshape(1, 1, 5, 1, 1)
    request["gen_data_clean"].x0_tokens_vision = [video]
    request["condition_reference"] = [torch.cat((video.flatten(), torch.zeros(12)))]
    if rtc:
        request["data_batch"]["causal_action_rtc"] = RTCPrefix(
            torch.tensor([[3.3, 4.2, float("nan")], [2.9, 4.4, 0.0]]), 1, beta=0.7
        )
    return request


def _observe_input_vjp(monkeypatch):
    """Record the real leaf gradient without replacing any gradient arithmetic."""
    original = sampling.guided_velocity
    observations = []

    def observe(sample, velocity_fn, *, sigma, target_flat, weights_flat, beta):
        record = {
            "sigma": sigma,
            "x": sample.detach().clone(),
            "target": target_flat.detach().clone(),
            "weights": weights_flat.detach().clone(),
        }

        def velocity(x):
            if x.requires_grad:
                x.register_hook(lambda gradient: record.update(vjp=gradient.detach().clone()))
            result = velocity_fn(x)
            record["base"] = result.detach().clone()
            return result

        result = original(sample, velocity, sigma, target_flat, weights_flat, beta)
        record["guided"] = result.detach().clone()
        observations.append(record)
        return result

    monkeypatch.setattr(sampling, "guided_velocity", observe)
    return observations


@pytest.mark.parametrize("solver", ["unipc", "fixed"])
@pytest.mark.parametrize("cfg", [1.0, 3.0])
def test_gradient_cache_matches_reference_native_solver_trajectories_and_joint_vjps(monkeypatch, solver, cfg):
    model = _RealModel(solver)
    observations = _observe_input_vjp(monkeypatch)
    reference_request = _real_request(model, cache_mode="plans", cfg=cfg)
    # FixedStep re-noises between evaluations; sample_prepared intentionally
    # delegates RNG state to its caller, independently of the initial noise.
    torch.manual_seed(101)
    reference = sampling.sample_prepared(model, **reference_request)
    expected, observations[:] = observations[:], []
    actual_request = _real_request(model, cache_mode="full", cfg=cfg)
    torch.manual_seed(101)
    actual = sampling.sample_prepared(model, **actual_request)

    assert len(expected) == len(observations) == 3
    assert actual["inference_cache"]["gradient_safe_prefix"]
    assert actual["inference_cache"]["full_cache"]
    assert not reference["inference_cache"]["full_cache"]
    branches = 1 if cfg == 1 else 2
    assert actual["inference_cache"]["prefill_calls"] == branches
    assert actual["inference_cache"]["compact_calls"] == 3 * branches
    assert actual["inference_cache"]["grad_compact_calls"] == 3 * branches
    assert actual["rtc"]["vjp_calls"] == reference["rtc"]["vjp_calls"] == 3
    assert all(parameter.grad is None for parameter in model.net.parameters())
    for observed, wanted in zip(observations, expected, strict=True):
        assert observed["sigma"] == wanted["sigma"]
        for key in ("target", "weights"):
            torch.testing.assert_close(observed[key], wanted[key], rtol=0, atol=0)
        for key in ("x", "base", "guided", "vjp"):
            torch.testing.assert_close(observed[key], wanted[key], atol=4e-6, rtol=3e-5)

        # The clean estimate's derivative is I-sigma*J_v. This explicitly rejects
        # a detached/identity-only surrogate and proves action -> video coupling.
        error = (observed["target"] - (observed["x"] - observed["sigma"] * observed["base"])) * observed["weights"]
        assert (observed["vjp"] - error).abs().max() > 1e-6
        assert not observed["weights"][:5].any()
        assert observed["vjp"][3:5].abs().max() > 1e-7

    for field in ("action", "vision"):
        torch.testing.assert_close(actual[field][0], reference[field][0], atol=4e-6, rtol=3e-5)
    absolute = real_actions(actual, {"sequence_plan": actual_request["sequence_plans"]})[0]
    torch.testing.assert_close(absolute[0], torch.tensor([3.3, 4.2, 0.0]), rtol=0, atol=0)
    assert not actual["action"][0][:2].any()
    assert not actual["action"][0][:, 2].any()
    torch.testing.assert_close(actual["vision"][0].flatten()[:3], torch.tensor([-0.2, -0.1, 0.0]), rtol=0, atol=0)


@pytest.mark.parametrize("abort_rtc", [False, True])
def test_ordinary_b_bytes_and_model_patches_survive_successful_or_failed_rtc(monkeypatch, abort_rtc):
    model = _RealModel("unipc")
    language_model = model.net.language_model
    forward_before = language_model.forward
    dispatch_before = [layer.self_attn.dispatch_attention_fn for layer in language_model.model.layers]
    weights_before = [(parameter._version, parameter.detach().clone()) for parameter in model.net.parameters()]
    original_factory = sampling.action_rtc_prefix_cache
    factory_calls = []

    def cache(net):
        factory_calls.append(net)
        return original_factory(net)

    monkeypatch.setattr(sampling, "action_rtc_prefix_cache", cache)
    before = sampling.sample_prepared(model, **_real_request(model, rtc=False))
    assert before["inference_cache"]["full_cache"]
    assert factory_calls == []
    if abort_rtc:
        original_guidance = sampling.guided_velocity

        def fail_after_real_forward(*args, **kwargs):
            original_guidance(*args, **kwargs)
            raise RuntimeError("test interruption after real cached input VJP")

        with monkeypatch.context() as scoped:
            scoped.setattr(sampling, "guided_velocity", fail_after_real_forward)
            with pytest.raises(RuntimeError, match="test interruption"):
                sampling.sample_prepared(model, **_real_request(model))
    else:
        sampling.sample_prepared(model, **_real_request(model))
    assert factory_calls == [model.net]
    assert language_model.forward == forward_before
    assert [layer.self_attn.dispatch_attention_fn for layer in language_model.model.layers] == dispatch_before
    assert not hasattr(language_model, "_rtc_prefix_cache_active")
    after = sampling.sample_prepared(model, **_real_request(model, rtc=False))
    assert factory_calls == [model.net]
    for name in ("action", "vision", "generated_action_mask"):
        assert torch.equal(
            before[name][0].contiguous().view(torch.uint8), after[name][0].contiguous().view(torch.uint8)
        )
    for parameter, (version, weight) in zip(model.net.parameters(), weights_before, strict=True):
        assert parameter._version == version
        assert torch.equal(parameter.detach().view(torch.uint8), weight.view(torch.uint8))
        assert parameter.grad is None
