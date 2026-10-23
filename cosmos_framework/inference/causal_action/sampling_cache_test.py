# SPDX-License-Identifier: OpenMDW-1.1
"""Sampling integration: cache selection, CFG routing, and preserved observations.

Contexts are spied only to verify policy selection. Real sequence expansion,
attention planning, template refresh, active masks, and sampling loops execute.
Prefix arithmetic equivalence is covered separately by prefix_cache_test.py.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action import sampling
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class _Sampler:
    def __init__(self, *, gradient_probe=False):
        self.calls = []
        self.gradient_probe = gradient_probe

    def __call__(self, velocity_fn, noise, **options):
        self.calls.append(options)
        value = noise[0].clone()
        for timestep in (900.0, 200.0):
            if self.gradient_probe:
                with torch.enable_grad():
                    value = value.detach().requires_grad_()
                    velocity = velocity_fn([value], torch.tensor([[timestep]]))[0]
                    gradient = torch.autograd.grad(velocity.sum(), value)[0]
                    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            else:
                velocity = velocity_fn([value], torch.tensor([[timestep]]))[0]
            value = value - 0.25 * velocity
        return [value]


class _Model:
    _update_inference_pack_template = OmniMoTModel._update_inference_pack_template
    _copy_timestep_to_template = OmniMoTModel._copy_timestep_to_template

    def __init__(self, *, preview=False, training=False, gradient_probe=False):
        self.precision = torch.float32
        self.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        self.config = SimpleNamespace(
            teacher_forcing_dense_mode="grouped_tnd",
            teacher_forcing_tnd_max_kv_tokens=131072,
            rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc"),
        )
        self.net = nn.Linear(1, 1)
        self.net.train(training)
        self.sampler = _Sampler()
        self.preview = preview
        self.gradient_probe = gradient_probe
        self.calls = []
        self.controller = None
        self.cache_targets = []
        self.metadata = SimpleNamespace(
            block_size=2,
            states=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
            state_mask=torch.tensor([[True, True, False], [True, True, False]]),
            action_mask=torch.tensor([[True, True, False]]).expand(4, -1),
            action_frame_ids=torch.arange(1, 5),
            state_action_indexes=torch.tensor([0, 2]),
            state_latent_indexes=torch.tensor([0, 2]),
        )

    def _pack_input_sequence(self, plans, text, data, timestep, *, skip_text_tokens):
        assert plans[0].causal_action_metadata is self.metadata
        assert timestep.shape == (1, 1)
        text_ids = torch.tensor(text[0][:1] if skip_text_tokens else text[0])
        nt = len(text_ids)
        vi, ai = torch.arange(nt, nt + 5), torch.arange(nt + 5, nt + 9)
        condition = torch.zeros(5, 1, 1)
        condition[: 1 if self.preview else 3] = 1
        noisy = torch.where(~condition.flatten().bool())[0]
        return PackedSequence(
            sample_lens=[nt + 9],
            split_lens=[nt, 9],
            attn_modes=["causal", "full"],
            sequence_length=nt + 9,
            text_ids=text_ids,
            text_indexes=torch.arange(nt),
            position_ids=torch.arange(nt + 9).expand(3, -1).clone().float(),
            vision=ModalityData(
                sequence_indexes=vi,
                mse_loss_indexes=vi[noisy],
                timesteps=torch.zeros(len(noisy)),
                tokens=data.x0_tokens_vision,
                token_shapes=[(5, 1, 1)],
                condition_mask=[condition],
                noisy_frame_indexes=[noisy],
            ),
            action=ModalityData(
                sequence_indexes=ai,
                mse_loss_indexes=ai,
                timesteps=torch.zeros(4),
                tokens=data.x0_tokens_action,
                token_shapes=[(4,)],
                condition_mask=[torch.zeros(4, 1)],
                noisy_frame_indexes=[torch.arange(4)],
                domain_id=[torch.zeros(1, dtype=torch.long)],
            ),
            causal_action_metadata=[self.metadata],
        )

    def denoise(self, *, net, data_batch_packed):
        packed = data_batch_packed
        name = "cond" if packed.text_ids[0] == 11 else "uncond"
        self.calls.append(
            dict(
                name=name,
                net=net,
                key=None if self.controller is None else self.controller["key"],
                vision=packed.vision.tokens[0].detach().clone(),
                timestep=packed.action.timesteps.clone(),
            )
        )
        if self.gradient_probe:
            return {
                "preds_vision": [packed.vision.tokens[0].square()],
                "preds_action": [packed.action.tokens[0].square()],
            }
        coefficient = 2.0 if name == "cond" else 1.0
        return {
            "preds_vision": [torch.full_like(packed.vision.tokens[0], coefficient)],
            "preds_action": [torch.full_like(packed.action.tokens[0], coefficient)],
        }


def _request(model, *, cache_mode="full", preview=False, custom_sampler=None, kv=True):
    video = torch.arange(100.0, 105.0).reshape(1, 1, 5, 1, 1)
    data = SimpleNamespace(batch_size=1, x0_tokens_vision=[video], x0_tokens_action=[torch.zeros(4, 3)])
    reference = torch.cat([video.flatten(), torch.zeros(12)])
    condition_mask = torch.zeros_like(reference)
    condition_mask[: 1 if preview else 3] = 1
    options = {"causal_action_preview": preview, "causal_action_current_block": 1}
    if cache_mode is not None:
        options["causal_action_cache_mode"] = cache_mode
    return dict(
        net=None,
        sampler=custom_sampler,
        guidance=3.0,
        guidance_interval=None,
        velocity_postprocess_builder=None,
        num_steps=2,
        shift=5.0,
        sigma_max=1.0,
        skip_text_tokens_for_cfg=True,
        normalize_cfg=False,
        sequence_plans=[SimpleNamespace(causal_action_metadata=model.metadata)],
        gen_data_clean=data,
        cond_tokens=[[11, 12]],
        uncond_tokens=[[43]],
        initial_noise=[torch.arange(10.0, 27.0)],
        condition_reference=[reference],
        condition_mask=[condition_mask],
        has_noisy_actions=True,
        causal_num_blocks=2,
        causal_block_size=2,
        causal_history_blocks=2,
        causal_use_kv_cache=kv,
        data_batch=options,
    )


def _spy_contexts(monkeypatch, model):
    events = []

    class Entries(dict):
        def clear(self):
            events.append(("invalidate",))
            super().clear()

    @contextmanager
    def context(kind, enabled=True):
        events.append((kind, enabled))
        controller = {"entries": Entries(), "key": None}
        model.controller = controller
        try:
            yield controller
        finally:
            model.controller = None
            events.append(("exit", kind))

    def to_cpu(template):
        events.append(("device", len(template.text_ids)))

    original_plan = sampling.prepare_attention_plan

    def prepare(template, **kwargs):
        events.append(("plan", len(template.text_ids)))
        return original_plan(template, **kwargs)

    def prefix(net):
        model.cache_targets.append(net)
        return context("prefix")

    def legacy(net, enabled):
        model.cache_targets.append(net)
        return context("legacy", enabled)

    monkeypatch.setattr(PackedSequence, "to_cuda", to_cpu)
    monkeypatch.setattr(sampling, "action_prefix_cache", prefix)
    monkeypatch.setattr(sampling, "action_attention_cache", legacy)
    monkeypatch.setattr(sampling, "prepare_attention_plan", prepare)
    return events


@pytest.mark.parametrize(
    ("mode", "preview", "custom", "kv", "training", "expected"),
    [
        (None, False, False, True, False, "prefix"),
        ("off", False, False, True, False, "legacy"),
        ("plans", False, False, True, False, "legacy"),
        ("full", False, False, True, False, "prefix"),
        ("full", True, False, True, False, "legacy"),
        ("full", False, True, True, False, "legacy"),
        ("full", False, False, False, False, "legacy"),
        ("full", False, False, True, True, "legacy"),
        ("off", False, False, False, False, "legacy"),
    ],
)
def test_cache_policy_preserves_current_block_sampling(monkeypatch, mode, preview, custom, kv, training, expected):
    model = _Model(preview=preview, training=training)
    events = _spy_contexts(monkeypatch, model)
    sampler = _Sampler() if custom else None
    result = sampling.sample_prepared(
        model, **_request(model, cache_mode=mode, preview=preview, custom_sampler=sampler, kv=kv)
    )
    assert (expected, True if expected == "prefix" else kv) in events
    assert sum(event[0] == "plan" for event in events) == (0 if mode == "off" else 2)
    assert sum(event[0] == "device" for event in events) == 2
    assert sum(event[0] == "invalidate" for event in events) == (2 if preview else 1)
    assert events[-1] == ("exit", expected)
    assert result["inference_cache"]["requested_mode"] == ("full" if mode is None else mode)
    assert result["inference_cache"]["full_cache"] == (expected == "prefix")
    assert model.cache_targets == [model.net]
    assert [call["name"] for call in model.calls] == ["cond", "uncond"] * (4 if preview else 2)
    assert [float(call["timestep"][0]) for call in model.calls] == [900, 900, 200, 200] * (2 if preview else 1)
    assert all(call["key"][1] == call["name"] for call in model.calls)
    assert not result["action"][0][:, -1].any()
    if preview:
        assert [call["key"][0] for call in model.calls] == [0] * 4 + [1] * 4
        assert result["generated_action_mask"][0].tolist() == [True] * 4
    else:
        assert all(call["key"][0] == 1 for call in model.calls)
        assert result["generated_action_mask"][0].tolist() == [False, False, True, True]
        torch.testing.assert_close(result["vision"][0].flatten(), torch.tensor([100.0, 101.0, 102.0, 11.0, 12.0]))
        torch.testing.assert_close(result["action"][0][:2], torch.zeros(2, 3))
        torch.testing.assert_close(result["action"][0][2:], torch.tensor([[19.0, 20.0, 0.0], [22.0, 23.0, 0.0]]))
        assert all(
            torch.equal(call["vision"].flatten()[:3], torch.tensor([100.0, 101.0, 102.0])) for call in model.calls
        )
    assert len((sampler or model.sampler).calls) == (2 if preview else 1)


@pytest.mark.parametrize("explicit_training", [False, True])
def test_explicit_network_controls_cache_selection_and_denoise_target(monkeypatch, explicit_training):
    model = _Model(training=not explicit_training)
    events = _spy_contexts(monkeypatch, model)
    explicit_net = nn.Linear(1, 1)
    explicit_net.train(explicit_training)
    request = _request(model)
    request["net"] = explicit_net
    result = sampling.sample_prepared(model, **request)
    assert result["inference_cache"]["full_cache"] == (not explicit_training)
    assert ("legacy" if explicit_training else "prefix", True) in events
    assert model.cache_targets == [explicit_net]
    assert all(call["net"] is explicit_net for call in model.calls)


def test_custom_gradient_sampler_uses_disabled_value_cache(monkeypatch):
    model = _Model(gradient_probe=True)
    sampler = _Sampler(gradient_probe=True)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda template: None)

    def unexpected_prefix(net):
        pytest.fail("A custom gradient sampler must not select prefix caching")

    monkeypatch.setattr(sampling, "action_prefix_cache", unexpected_prefix)
    result = sampling.sample_prepared(model, **_request(model, custom_sampler=sampler, kv=False))
    assert not result["inference_cache"]["full_cache"]
    assert len(model.calls) == 4


@pytest.mark.parametrize("dense_mode", ["global", "per_sample"])
@pytest.mark.parametrize("cache_mode", [None, "off", "plans", "full"])
def test_non_grouped_backend_without_kv_cache_skips_tnd_planning(monkeypatch, dense_mode, cache_mode):
    model = _Model()
    model.config.teacher_forcing_dense_mode = dense_mode
    events = _spy_contexts(monkeypatch, model)

    def unexpected_plan(*args, **kwargs):
        pytest.fail("Dense sampling without KV caching must not construct a TND plan")

    monkeypatch.setattr(sampling, "prepare_attention_plan", unexpected_plan)
    result = sampling.sample_prepared(model, **_request(model, cache_mode=cache_mode, kv=False))
    assert ("legacy", False) in events
    assert not result["inference_cache"]["full_cache"]
    assert len(model.calls) == 4
    torch.testing.assert_close(result["vision"][0].flatten(), torch.tensor([100.0, 101.0, 102.0, 11.0, 12.0]))
    torch.testing.assert_close(result["action"][0][2:], torch.tensor([[19.0, 20.0, 0.0], [22.0, 23.0, 0.0]]))


@pytest.mark.parametrize("dense_mode", ["global", "per_sample"])
def test_non_grouped_backend_still_rejects_kv_caching(monkeypatch, dense_mode):
    model = _Model()
    model.config.teacher_forcing_dense_mode = dense_mode
    events = _spy_contexts(monkeypatch, model)
    with pytest.raises(ValueError, match="KV cache requires grouped_tnd"):
        sampling.sample_prepared(model, **_request(model, kv=True))
    assert not events


def test_invalid_cache_policy_fails_before_planning(monkeypatch):
    model = _Model()
    events = _spy_contexts(monkeypatch, model)
    with pytest.raises(ValueError, match="cache_mode"):
        sampling.sample_prepared(model, **_request(model, cache_mode="unknown"))
    assert not events
