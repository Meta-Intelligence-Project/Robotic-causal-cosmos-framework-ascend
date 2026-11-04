# SPDX-License-Identifier: OpenMDW-1.1
"""Compact windows keep action/state payload independent from observed history."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action import sampling
from cosmos_framework.inference.causal_action.outputs import real_actions
from cosmos_framework.inference.causal_action.rtc import RTCPrefix, prepare_prefix
from cosmos_framework.inference.causal_action.rtc_batch import RTCObservedBatchBuilder
from cosmos_framework.inference.causal_action.rtc_batch_test import _image, _reference, _request
from cosmos_framework.inference.causal_action.rtc_sampling_test import _CoupledModel, _ExactSampler
from cosmos_framework.inference.causal_action.sampling_cache_test import _spy_contexts


@pytest.mark.parametrize("past", [0, 48, 1024])
@pytest.mark.parametrize("horizon", [16, 32, 48])
def test_builder_uses_independent_real_history_and_prediction_lengths(past, horizon):
    builder = RTCObservedBatchBuilder(
        _reference(),
        controller_hz=30,
        history_steps=past,
        action_horizon=horizon,
        history_padding="repeat_first",
    )
    request = _request(49)
    batch = builder(**request, first_image=_image(0))
    m = batch["sequence_plan"][0].causal_action_metadata
    observed, future = 1 + past // 16, horizon // 16
    assert batch["action"][0].shape == (horizon, 55)
    assert batch["action_raw"][0].shape == (horizon, 55)
    assert m.states.shape == m.anchors.shape == m.state_mask.shape == (1, 55)
    assert m.action_mask.shape == (horizon, 55)
    assert m.action_frame_ids.tolist() == (observed + torch.arange(horizon) // 16).tolist()
    assert m.target_vision_span == (observed, observed + future)
    assert m.current_group_id == observed and m.current_state_index == 0
    assert not m.action_anchor_indexes.any()
    assert batch["causal_action_current_block"] == observed
    assert batch["sequence_plan"][0].action_start_frame_offset == past + 1
    assert batch["video"][0].shape[1] == 1 + (past + horizon) // 4
    assert batch["rtc_observation_ticks"].tolist() == list(range(49 - past, 50, 4))


class CompactModel(_CoupledModel):
    def __init__(self, past):
        super().__init__()
        old = self.metadata
        observed = 1 + past // 16
        self.metadata = BlockStateMetadata(
            block_size=2,
            history_blocks=observed,
            states=old.states[1:2].clone(),
            anchors=old.anchors[1:2].clone(),
            state_mask=old.state_mask[1:2].clone(),
            action_mask=old.action_mask[-1:].expand(32, -1).clone(),
            action_frame_ids=observed + torch.arange(32) // 16,
            state_action_indexes=torch.tensor([0]),
            state_latent_indexes=torch.tensor([observed - 1]),
            statistics=old.statistics,
            template=old.template,
            include_visual_prefix=True,
            layout_mode="decision_window",
            history_steps=past,
            action_horizon=32,
            observed_latents=observed,
            future_latents=2,
            actions_per_latent_frame=16,
        )
        self.sampler = _ExactSampler()
        self.calls = []

    def _pack_input_sequence(self, plans, text, data, timestep, *, skip_text_tokens):
        del timestep
        text_ids = torch.tensor(text[0][:1] if skip_text_tokens else text[0])
        nt, frames, horizon = len(text_ids), self.metadata.observed_latents + 2, 32
        vi = torch.arange(nt, nt + frames)
        ai = torch.arange(nt + frames, nt + frames + horizon)
        condition = torch.zeros(frames, 1, 1)
        condition[: self.metadata.observed_latents] = 1
        noisy = torch.where(~condition.flatten().bool())[0]
        total = nt + frames + horizon
        return PackedSequence(
            sample_lens=[total],
            split_lens=[nt, frames + horizon],
            attn_modes=["causal", "full"],
            sequence_length=total,
            text_ids=text_ids,
            text_indexes=torch.arange(nt),
            position_ids=torch.arange(total).expand(3, -1).clone().float(),
            vision=ModalityData(
                sequence_indexes=vi,
                mse_loss_indexes=vi[noisy],
                timesteps=torch.zeros(len(noisy)),
                tokens=data.x0_tokens_vision,
                token_shapes=[(frames, 1, 1)],
                condition_mask=[condition],
                noisy_frame_indexes=[noisy],
            ),
            action=ModalityData(
                sequence_indexes=ai,
                mse_loss_indexes=ai,
                timesteps=torch.zeros(horizon),
                tokens=data.x0_tokens_action,
                token_shapes=[(horizon,)],
                condition_mask=[torch.zeros(horizon, 1)],
                noisy_frame_indexes=[torch.arange(horizon)],
                domain_id=[torch.zeros(1, dtype=torch.long)],
            ),
            causal_action_metadata=[self.metadata],
        )

    def denoise(self, *, net, data_batch_packed):
        self.calls.append((self.controller["key"], data_batch_packed))
        return super().denoise(net=net, data_batch_packed=data_batch_packed)


def compact_request(model, *, cfg=1, cache_mode="full", rtc=False):
    observed = model.metadata.observed_latents
    frames = observed + 2
    video = torch.arange(frames).float().reshape(1, 1, frames, 1, 1) / 100
    action = torch.zeros(32, 3)
    data = SimpleNamespace(batch_size=1, x0_tokens_vision=[video], x0_tokens_action=[action])
    reference = torch.cat((video.flatten(), action.flatten()))
    condition = torch.zeros_like(reference)
    condition[:observed] = 1
    batch = dict(causal_action_current_block=observed, causal_action_cache_mode=cache_mode)
    if rtc:
        batch["causal_action_rtc"] = RTCPrefix(torch.tensor([[3.5, 4.2, 0.0]]).expand(16, -1).clone(), 2, beta=1)
    return dict(
        net=None,
        sampler=None,
        guidance=cfg,
        guidance_interval=None,
        velocity_postprocess_builder=None,
        num_steps=2,
        shift=5.0,
        sigma_max=1,
        skip_text_tokens_for_cfg=True,
        normalize_cfg=False,
        sequence_plans=[SimpleNamespace(causal_action_metadata=model.metadata)],
        gen_data_clean=data,
        cond_tokens=[[11, 12]],
        uncond_tokens=[[43]],
        initial_noise=[torch.linspace(-0.2, 0.2, len(reference))],
        condition_reference=[reference],
        condition_mask=[condition],
        has_noisy_actions=True,
        causal_num_blocks=None,
        causal_block_size=2,
        causal_history_blocks=observed,
        causal_use_kv_cache=True,
        data_batch=batch,
    )


@pytest.mark.parametrize("past", [48, 1024])
@pytest.mark.parametrize("cfg", [1, 3])
@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("cache_mode", ["off", "plans", "full"])
def test_one_joint_window_solve_uses_group_m_but_anchor_zero(monkeypatch, past, cfg, rtc, cache_mode):
    model = CompactModel(past)
    events = _spy_contexts(monkeypatch, model)
    request = compact_request(model, cfg=cfg, cache_mode=cache_mode, rtc=rtc)
    result = sampling.sample_prepared(model, **request)
    observed = model.metadata.observed_latents
    assert len(model.calls) == 2 * (1 if cfg == 1 else 2)
    assert {key[0] for key, _ in model.calls} == {observed}
    roles = model.calls[0][1].teacher_forcing.layout.stream_ids
    assert int((roles == 0).sum()) == observed
    assert int((roles == 1).sum()) == 34
    assert int((roles == 3).sum()) == 1
    torch.testing.assert_close(
        result["vision"][0][:, :, :observed], request["gen_data_clean"].x0_tokens_vision[0][:, :, :observed]
    )
    assert result["action"][0].shape == (32, 3)
    assert result["generated_action_mask"][0].all()
    assert not result["action"][0][:, 2].any()
    physical = real_actions(result, {"sequence_plan": request["sequence_plans"]})[0]
    assert physical.shape == (32, 3)
    if rtc:
        assert result["rtc"]["vjp_calls"] == 2
        torch.testing.assert_close(
            physical[:2], request["data_batch"]["causal_action_rtc"].previous_actions[:2], rtol=0, atol=0
        )
        prepared = prepare_prefix(request["data_batch"]["causal_action_rtc"], model.metadata, observed, device="cpu")
        assert prepared.action_indexes.tolist() == list(range(32))
    assert any(event[0] == "plan" for event in events) == (cache_mode != "off")


@pytest.mark.parametrize("past", [48, 1024])
@pytest.mark.parametrize("solver", ["fixed", "unipc"])
@pytest.mark.parametrize("cfg", [1, 3])
@pytest.mark.parametrize("rtc", [False, True])
def test_compact_real_transformer_prefix_cache_matches_reference(monkeypatch, past, solver, cfg, rtc):
    from cosmos_framework.inference.causal_action.rtc_gradient_sampling_test import _RealModel
    from cosmos_framework.model.attention import attention

    def cpu_attention(*args, backend, **kwargs):
        del backend
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _RealModel(solver)
        model.metadata = CompactModel(past).metadata
        model._pack_input_sequence = CompactModel._pack_input_sequence.__get__(model)
        model.net.max_latent_t = model.metadata.observed_latents + 2
        plain_request = compact_request(model, cfg=cfg, cache_mode="plans", rtc=rtc)
        torch.manual_seed(101)
        reference = sampling.sample_prepared(model, **plain_request)
        torch.manual_seed(101)
        actual = sampling.sample_prepared(model, **compact_request(model, cfg=cfg, cache_mode="full", rtc=rtc))
        torch.testing.assert_close(actual["vision"][0], reference["vision"][0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual["action"][0], reference["action"][0], rtol=2e-5, atol=2e-5)
        assert actual["inference_cache"]["full_cache"]
        if rtc:
            assert actual["rtc"]["gradient_safe_prefix"] and actual["rtc"]["vjp_calls"] > 0
            physical = real_actions(actual, {"sequence_plan": plain_request["sequence_plans"]})[0]
            torch.testing.assert_close(
                physical[:2], plain_request["data_batch"]["causal_action_rtc"].previous_actions[:2], rtol=0, atol=0
            )
        assert all(parameter.grad is None for parameter in model.net.parameters())
    finally:
        torch.set_num_threads(previous)
