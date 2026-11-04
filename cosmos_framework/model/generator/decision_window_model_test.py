# SPDX-License-Identifier: OpenMDW-1.1
"""Decision windows preserve current-time positions and sample one joint sigma."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
from cosmos_framework.model.generator.causal_multitask_test import make_model
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _decision(history_steps, horizon):
    observed, future = 1 + history_steps // 16, horizon // 16
    state = torch.zeros(1, 55)
    metadata = BlockStateMetadata(
        block_size=future,
        history_blocks=observed,
        action_frame_ids=observed + torch.arange(horizon) // 16,
        action_mask=torch.ones(horizon, 55, dtype=torch.bool),
        states=state,
        state_mask=torch.ones_like(state, dtype=torch.bool),
        anchors=state.clone(),
        state_action_indexes=torch.tensor([0]),
        state_latent_indexes=torch.tensor([observed - 1]),
        layout_mode="decision_window",
        history_steps=history_steps,
        action_horizon=horizon,
        observed_latents=observed,
        future_latents=future,
        actions_per_latent_frame=16,
    )
    # The model must enforce the absolute current-time offset even for callers
    # still supplying SequencePlan's legacy default of one.
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=list(range(observed)),
        causal_action_metadata=metadata,
    )
    return plan, GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 2, observed + future, 1, 1)],
        x0_tokens_action=[torch.randn(horizon, 55)],
        action_domain_id=[torch.tensor(0)],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([30.0]),
    )


@pytest.mark.parametrize("fps_modulation", [False, True])
@pytest.mark.parametrize("history,horizon", [(0, 32), (48, 32), (64, 48), (1024, 32)])
def test_compact_action_positions_continue_from_the_current_observation(history, horizon, fps_modulation):
    model = make_model()
    model.config.max_action_dim = 55
    model.config.diffusion_expert_config.patch_spatial = 1
    model.config.diffusion_expert_config.enable_fps_modulation = fps_modulation
    model.tokenizer_vision_gen = SimpleNamespace(temporal_compression_factor=4)
    model.llm_special_tokens = dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4)
    plan, clean = _decision(history, horizon)
    packed = model._pack_input_sequence([plan], [[11, 12]], clean, torch.tensor([[700.0]]))
    metadata = plan.causal_action_metadata
    geometry = model.prepare_teacher_forcing_geometry_from_plans(
        [metadata.observed_latents + metadata.future_latents], [plan]
    )
    assert geometry.decision_windows == (True,)
    assert geometry.history_blocks == (metadata.observed_latents,)
    video_times = packed.position_ids[0, packed.vision.sequence_indexes]
    anchor = video_times[metadata.observed_latents - 1]
    interval = video_times[metadata.observed_latents] - anchor
    expected = anchor + torch.arange(1, horizon + 1) * interval / 16
    torch.testing.assert_close(packed.position_ids[0, packed.action.sequence_indexes], expected)
    torch.testing.assert_close(expected[-1], video_times[-1])
    expanded = expand_action_sequence(packed, clean.x0_tokens_vision, geometry)
    assert expanded.teacher_forcing.layout.state_indexes.numel() == 1
    torch.testing.assert_close(expanded.position_ids[0, expanded.teacher_forcing.layout.state_indexes], anchor[None])


def test_noise_sampling_draws_one_sigma_per_window_instead_of_per_history_frame(monkeypatch):
    model = make_model()
    model.config.rectified_flow_training_config.shift = 1
    flow = RectifiedFlow(velocity_field=None, device="cpu")
    model.rectified_flow_video = flow
    counts = []

    def draw(count, **kwargs):
        counts.append(count)
        return torch.arange(1, count + 1).float() / (count + 1)

    monkeypatch.setattr(flow, "sample_train_time", draw)
    geometry = CausalActionGeometry((2, 2), (4, 65), (False, False), (True, True))
    times, sigmas = model._get_train_noise_level_vision(
        batch_size=2,
        is_image_batch=False,
        num_vision_latent_frames=[6, 67],
        teacher_forcing_geometry=geometry,
    )
    assert counts == [2]
    assert sigmas.shape == times.shape == (2, 67)
    torch.testing.assert_close(sigmas[0, :6], torch.full((6,), 1 / 3))
    torch.testing.assert_close(sigmas[1], torch.full((67,), 2 / 3))
    assert not sigmas[0, 6:].any()


def test_mixed_video_blocks_and_decision_windows_keep_separate_noise_policies(monkeypatch):
    model = make_model()
    model.config.rectified_flow_training_config.shift = 1
    flow = RectifiedFlow(velocity_field=None, device="cpu")
    model.rectified_flow_video = flow
    counts = []

    def draw(count, **kwargs):
        counts.append(count)
        return torch.arange(1, count + 1).float() / 10

    monkeypatch.setattr(flow, "sample_train_time", draw)
    geometry = CausalActionGeometry((2, 2), (1, 65), (False, False), (False, True))
    _, sigmas = model._get_train_noise_level_vision(
        batch_size=2,
        is_image_batch=False,
        num_vision_latent_frames=[5, 67],
        teacher_forcing_geometry=geometry,
    )
    assert counts == [4]  # legacy: singleton prime + two blocks; decision window: one.
    torch.testing.assert_close(sigmas[0, :5], torch.tensor([0.1, 0.2, 0.2, 0.3, 0.3]))
    torch.testing.assert_close(sigmas[1], torch.full((67,), 0.4))
