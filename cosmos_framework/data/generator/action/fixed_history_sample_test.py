# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-window supervision, fresh anchors and historical-target isolation."""

import copy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import CausalActionSFTDataset
from cosmos_framework.data.generator.action.fixed_history_sample import build_fixed_history_sample
from cosmos_framework.data.generator.action.history_window import build_history_window
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.sequence_packing import SequencePlan


def sample():
    template = ActionStateTemplate55()
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    planner = SegmentPlanner(max_action_steps=64, overlap_action_steps=0, geometry=geometry)
    state = torch.zeros(33, 55)
    state[:, 0] = torch.arange(42, 75)
    mask = torch.zeros(55, dtype=torch.bool)
    mask[0] = True
    statistics = BlockStatistics(
        Quantiles(torch.full((55,), -100.0), torch.full((55,), 100.0), mask),
        Quantiles(torch.full((55,), -100.0), torch.full((55,), 100.0), mask),
    )
    raw = dict(
        state_trajectory=state,
        action_target=state[1:].clone(),
        state_mask=mask,
        action_mask=mask.clone(),
        conditioning_fps=30.0,
        video=torch.arange(25, dtype=torch.uint8).view(1, 25, 1, 1).expand(3, -1, 2, 2).clone(),
        history_window=build_history_window(16, history_blocks=2, history_padding="repeat_first"),
    )
    return raw, template, planner, statistics


def test_fixed_history_uses_only_current_real_anchor_and_target_block():
    raw, template, planner, stats = sample()
    before = copy.deepcopy(raw)
    result, metadata = build_fixed_history_sample(raw, template=template, planner=planner, statistics=stats)
    current, expected = build_block_sample(
        dict(raw, video=None), template=template, planner=planner, history_blocks=2, statistics=stats
    )
    assert result["action"].shape == (32, 55)
    torch.testing.assert_close(result["action"], current["action"])
    torch.testing.assert_close(metadata.states[0], expected.states[0])
    torch.testing.assert_close(metadata.anchors[0], raw["state_trajectory"][0])
    assert metadata.states.shape == metadata.state_mask.shape == (1, 55)
    assert torch.equal(metadata.action_mask, raw["action_mask"].expand(32, -1))
    assert metadata.include_visual_prefix
    assert metadata.action_frame_ids.tolist() == torch.arange(5, 7).repeat_interleave(16).tolist()
    assert metadata.state_latent_indexes.tolist() == [4]
    assert metadata.current_group_id == 5 and metadata.current_state_index == 0
    assert metadata.target_vision_span == (5, 7)
    assert not metadata.action_anchor_indexes.any()
    assert metadata.action_indexes_for_group(5).tolist() == list(range(32))
    assert not len(metadata.action_indexes_for_group(0))
    assert torch.equal(result["video"], raw["video"])
    assert torch.equal(raw["state_trajectory"], before["state_trajectory"])


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
def test_fixed_history_dataset_conditions_history_and_supervises_only_future(mode):
    raw, template, planner, stats = sample()
    reader = SimpleNamespace(
        template=template,
        planner=planner,
        root="synthetic",
        read_prediction_window=lambda idx, history_steps, action_horizon: copy.deepcopy(raw),
    )

    def transform(data, resolution, action_normalizer):
        return dict(
            data,
            sequence_plan=SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=list(range(7)) if mode == "inverse_dynamics" else [0],
            ),
        )

    dataset = CausalActionSFTDataset(
        ActionSFTDataset(reader, transform, "480"),
        mode=mode,
        statistics=stats,
        history_blocks_max=2,
        history_padding="repeat_first",
    )
    result = dataset[0]
    plan = result["sequence_plan"]
    assert plan.condition_frame_indexes_action == list(range(32 if mode == "forward_dynamics" else 0))
    assert plan.condition_frame_indexes_vision == list(range(7 if mode == "inverse_dynamics" else 5))
    assert plan.causal_action_metadata.state_mask.shape == (1, 55)
    assert plan.action_start_frame_offset == 65
    assert "state_trajectory" not in result and "history_window" not in result


def test_fixed_history_training_requires_reader_with_real_episode_boundaries():
    _, template, planner, stats = sample()
    reader = SimpleNamespace(template=template, planner=planner)
    with pytest.raises(ValueError, match="true episode"):
        CausalActionSFTDataset(
            ActionSFTDataset(reader, None, "480"),
            mode="policy",
            statistics=stats,
            history_padding="repeat_first",
        )


def test_full_1024_step_history_still_supervises_only_one_real_future_block():
    raw, template, planner, stats = sample()
    raw["history_window"] = build_history_window(16, history_blocks=32, history_padding="repeat_first")
    raw["video"] = torch.zeros(3, 265, 2, 2, dtype=torch.uint8)
    result, metadata = build_fixed_history_sample(raw, template=template, planner=planner, statistics=stats)
    assert result["action"].shape == (32, 55)
    assert torch.equal(metadata.action_mask, raw["action_mask"].expand(32, -1))
    assert metadata.states.shape == (1, 55)
    assert metadata.anchors[0, 0] == 42
    assert metadata.state_latent_indexes[-1] == 64


@pytest.mark.parametrize("history,horizon", [(0, 16), (48, 32), (64, 48), (1024, 32)])
def test_history_length_and_action_horizon_are_independent_action_step_units(history, horizon):
    raw, template, planner, stats = sample()
    state = torch.zeros(horizon + 1, 55)
    state[:, 0] = torch.arange(42, 43 + horizon)
    raw.update(
        state_trajectory=state,
        action_target=state[1:].clone(),
        action_horizon=horizon,
        history_window=build_history_window(17, history_steps=history, history_padding="repeat_first"),
        video=torch.zeros(3, 1 + (history + horizon) // 4, 2, 2, dtype=torch.uint8),
    )
    result, metadata = build_fixed_history_sample(raw, template=template, planner=planner, statistics=stats)
    observed = 1 + history // 16
    assert result["action"].shape == (horizon, 55)
    assert metadata.layout_mode == "decision_window"
    assert metadata.history_steps == history and metadata.action_horizon == horizon
    assert metadata.observed_latents == observed and metadata.future_latents == horizon // 16
    assert metadata.states.shape == (1, 55)
    assert metadata.state_latent_indexes.tolist() == [observed - 1]
    assert metadata.action_frame_ids.tolist() == (observed + torch.arange(horizon) // 16).tolist()


def test_compact_metadata_rejects_historical_dummy_state_or_action_slots():
    from dataclasses import replace

    raw, template, planner, stats = sample()
    _, metadata = build_fixed_history_sample(raw, template=template, planner=planner, statistics=stats)
    with pytest.raises(ValueError, match="exactly one"):
        replace(metadata, states=metadata.states.expand(3, -1))
    with pytest.raises(ValueError, match="predicted horizon"):
        replace(metadata, action_mask=metadata.action_mask.repeat(3, 1))


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
def test_real_action_transform_preserves_compact_window_and_target_modes(mode):
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    raw, template, planner, stats = sample()
    raw.update(
        history_window=build_history_window(17, history_steps=48, history_padding="repeat_first"),
        video=torch.zeros(3, 21, 2, 2, dtype=torch.uint8),
        ai_caption="Move the object",
        viewpoint="ego_view",
    )
    reader = SimpleNamespace(
        template=template,
        planner=planner,
        root="synthetic",
        read_prediction_window=lambda idx, history_steps, action_horizon: copy.deepcopy(raw),
    )
    transform = ActionTransformPipeline(
        max_action_dim=55,
        video_temporal_downsample=4,
        tokenizer_config=None,
        cfg_dropout_rate=0,
        format_prompt_as_json=True,
    )
    dataset = CausalActionSFTDataset(
        ActionSFTDataset(reader, transform, "256"),
        mode=mode,
        statistics=stats,
        history_padding="repeat_first",
        history_steps=48,
        action_horizon=32,
    )
    result = dataset[0]
    metadata = result["sequence_plan"].causal_action_metadata
    assert result["action"].shape == result["action_raw"].shape == (32, 55)
    assert metadata.states.shape == (1, 55)
    assert result["video"].shape[1] == 21
    assert result["sequence_plan"].action_start_frame_offset == 49
    assert result["sequence_plan"].condition_frame_indexes_action == list(
        range(32 if mode == "forward_dynamics" else 0)
    )
    assert result["sequence_plan"].condition_frame_indexes_vision == list(range(6 if mode == "inverse_dynamics" else 4))
    assert result["ai_caption"]["duration"] == "2s"  # caption描述完整P+H窗口，非仅H。
