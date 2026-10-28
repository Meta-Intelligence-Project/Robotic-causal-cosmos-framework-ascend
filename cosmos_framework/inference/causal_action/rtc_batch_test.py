# SPDX-License-Identifier: OpenMDW-1.1
"""Real-observation RTC windows, partial old-block starts and target isolation."""

import copy
from dataclasses import asdict

import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.causal_action.rtc_batch import RTCObservedBatchBuilder
from cosmos_framework.inference.causal_action.rtc_session import RTCObservation


def _state():
    state = torch.full((55,), float("nan"))
    sm = torch.zeros(55, dtype=torch.bool)
    sm[[0, 14, 15, 16, 17, 18, 19, 20]] = True
    am = sm.clone()
    am[28] = True
    state[sm] = 0
    state[0], state[14] = 0.4, 0.123
    state[17:21] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]) * (1 + 1e-6)
    return state, sm, am


def _reference():
    template = ActionStateTemplate55()
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    planner = SegmentPlanner(max_action_steps=64, overlap_action_steps=0, geometry=geometry)
    state, sm, am = _state()
    valid_state = template.sanitize(state, sm)
    target = template.sanitize(valid_state.masked_fill(~am, 0), am)
    target[28] = 0.5
    stats = BlockStatistics(
        Quantiles(-torch.ones(55), torch.ones(55), sm),
        Quantiles(-torch.ones(55), torch.ones(55), am),
        provenance={"kind": "fitted", "geometry": asdict(geometry), "template_id": template.template_id},
    )
    _, metadata = build_block_sample(
        dict(
            state_trajectory=valid_state.expand(65, -1).clone(),
            action_target=target.expand(64, -1).clone(),
            state_mask=sm,
            action_mask=am,
            conditioning_fps=30.0,
        ),
        template=template,
        planner=planner,
        history_blocks=8,
        statistics=stats,
    )
    return {
        "video": [torch.full((3, 17, 16, 16), 239, dtype=torch.uint8)],
        "action": [torch.full((64, 55), 173.0)],
        "action_raw": [torch.full((64, 55), 177.0)],
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=list(range(5)),
                condition_frame_indexes_action=list(range(64)),
                causal_action_metadata=metadata,
            )
        ],
        "ai_caption": ["Pick up the cup"],
        "text_token_ids": [torch.tensor([[1, 2, 3]])],
        "conditioning_fps": torch.tensor([7.5]),
        "conditioning_fps_action": torch.tensor([30.0]),
        "image_size": [torch.tensor([16, 16, 16, 16])],
        "domain_id": [torch.tensor(0)],
        "action_processing_record": [ActionProcessingRecord(55, None)],
        "causal_action_preview": True,
        "ground_truth_future": torch.full((10,), 999.0),
        "state_trajectory": torch.full((65, 55), 999.0),
    }


def _image(tick):
    return torch.full((3, 16, 16), tick % 200, dtype=torch.uint8)


def _request(tick, *, revision=0):
    state, sm, am = _state()
    return dict(
        state=state,
        state_mask=sm,
        action_mask=am,
        image=_image(tick),
        history=tuple(RTCObservation(t, _image(t)) for t in range(tick)),
        tick=tick,
        revision=revision,
        rtc=object(),
    )


@pytest.mark.parametrize("tick,current,first", [(0, 0, 0), (16, 0, 16), (32, 1, 0), (48, 1, 16), (64, 2, 0)])
def test_partial_start_uses_true_recent_complete_visual_blocks(tick, current, first):
    builder = RTCObservedBatchBuilder(_reference(), controller_hz=30.0)
    request = _request(tick)
    batch = builder(**request)
    ticks = list(range(first, tick + 1, 4))
    video = batch["video"][0]
    assert batch["causal_action_current_block"] == current
    assert batch["rtc_window_start_tick"] == first
    assert batch["rtc_observation_ticks"].tolist() == ticks
    assert video.shape == (3, len(ticks) + 8, 16, 16)
    assert torch.equal(video[:, : len(ticks)], torch.stack([_image(t) for t in ticks], dim=1))
    assert not video[:, len(ticks) :].any()
    assert not batch["action"][0].any()
    assert not batch["action_raw"][0].any()
    assert batch["action"][0].shape == ((current + 1) * 32, 55)
    plan = batch["sequence_plan"][0]
    metadata = plan.causal_action_metadata
    assert plan.condition_frame_indexes_vision == list(range(1 + 2 * current))
    assert plan.condition_frame_indexes_action == []
    assert not batch["causal_action_preview"]
    assert "causal_action_rtc" not in batch
    assert "ground_truth_future" not in batch and "state_trajectory" not in batch
    assert not metadata.states[:current].any()
    assert not metadata.state_mask[:current].any()
    assert not metadata.anchors[:current].any()
    expected = builder.template.sanitize(request["state"], request["state_mask"])
    assert torch.equal(metadata.anchors[current], expected)
    assert torch.equal(metadata.state_mask[current], request["state_mask"])
    assert torch.equal(metadata.action_mask, request["action_mask"].expand((current + 1) * 32, -1))
    assert metadata.state_action_indexes.tolist() == list(range(0, (current + 1) * 32, 32))
    assert metadata.state_latent_indexes.tolist() == list(range(0, (current + 1) * 2, 2))


def test_history_cap_slides_at_current_observation_and_uses_no_old_prototype_data():
    reference = _reference()
    builder = RTCObservedBatchBuilder(reference, controller_hz=30.0, history_blocks=1)
    # Destroy every mutable reference input after construction: none is a live source.
    reference["video"][0].fill_(255)
    reference["action"][0].fill_(float("nan"))
    reference["sequence_plan"][0].causal_action_metadata.anchors.fill_(999)
    reference["sequence_plan"][0].causal_action_metadata.statistics.state.low.fill_(999)
    reference["text_token_ids"][0].fill_(999)
    reference["ai_caption"][0] = "stale changed task"
    for tick in (32, 48, 64):
        request = _request(tick)
        batch = builder(**request)
        assert batch["causal_action_current_block"] == 1
        assert batch["rtc_window_start_tick"] == tick - 32
        assert batch["rtc_observation_ticks"].tolist() == list(range(tick - 32, tick + 1, 4))
        assert batch["ai_caption"] == ["Pick up the cup"]
        assert batch["text_token_ids"][0].tolist() == [[1, 2, 3]]
        assert torch.isfinite(batch["sequence_plan"][0].causal_action_metadata.states).all()


def test_current_state_normalization_matches_training_encoder_and_masks():
    builder = RTCObservedBatchBuilder(_reference(), controller_hz=30.0)
    request = _request(48)
    batch = builder(**request)
    m = batch["sequence_plan"][0].causal_action_metadata
    geometry = builder.geometry
    planner = SegmentPlanner(max_action_steps=32, overlap_action_steps=0, geometry=geometry)
    state = builder.template.sanitize(request["state"], request["state_mask"])
    target = state.clone()
    target[28] = 0.5
    _, expected = build_block_sample(
        dict(
            state_trajectory=state.expand(33, -1).clone(),
            action_target=target.expand(32, -1).clone(),
            state_mask=request["state_mask"],
            action_mask=request["action_mask"],
            conditioning_fps=30.0,
        ),
        template=builder.template,
        planner=planner,
        history_blocks=8,
        statistics=m.statistics,
    )
    torch.testing.assert_close(m.states[1], expected.states[0])
    torch.testing.assert_close(m.anchors[1], expected.anchors[0])
    torch.testing.assert_close(m.state_clip_fraction, expected.state_clip_fraction)
    assert m.anchors[1, 28] == 0  # The unknown absolute-gripper state is invalid.
    assert torch.isfinite(m.states).all()


def test_builder_never_mutates_reference_or_live_measurements():
    reference = _reference()
    before = copy.deepcopy(reference)
    builder = RTCObservedBatchBuilder(reference, controller_hz=30.0)
    request = _request(48)
    image_before = request["image"].clone()
    state_before = request["state"].clone()
    history_before = request["history"][16].image.clone()
    batch = builder(**request)
    batch["video"][0].zero_()
    batch["sequence_plan"][0].causal_action_metadata.anchors.fill_(777)
    batch["sequence_plan"][0].causal_action_metadata.statistics.state.low.fill_(777)
    batch["text_token_ids"][0].fill_(777)
    assert torch.equal(request["image"], image_before)
    torch.testing.assert_close(request["state"], state_before, equal_nan=True)
    assert torch.equal(request["history"][16].image, history_before)
    assert torch.equal(reference["video"][0], before["video"][0])
    assert torch.equal(reference["action"][0], before["action"][0])
    assert torch.equal(reference["text_token_ids"][0], before["text_token_ids"][0])
    assert reference["ai_caption"] == before["ai_caption"]
    fresh = builder(**request)
    assert fresh["text_token_ids"][0].tolist() == [[1, 2, 3]]
    assert fresh["sequence_plan"][0].causal_action_metadata.anchors[1, 0] == 0.4


def test_uncollated_reference_sample_is_accepted_and_collated_safely():
    batch = _reference()
    sample = {key: value[0] if isinstance(value, list) else value for key, value in batch.items()}
    sample["conditioning_fps"], sample["conditioning_fps_action"] = torch.tensor(7.5), torch.tensor(30.0)
    result = RTCObservedBatchBuilder(sample, controller_hz=30.0)(**_request(16))
    assert isinstance(result["video"], list) and len(result["video"]) == 1
    assert result["ai_caption"] == ["Pick up the cup"]
    assert result["text_token_ids"][0].tolist() == [[1, 2, 3]]


def test_missing_selected_frame_is_rejected_instead_of_repeating_a_picture():
    request = _request(48)
    request["history"] = tuple(r for r in request["history"] if r.tick != 20)
    with pytest.raises(ValueError, match="missing.*20"):
        RTCObservedBatchBuilder(_reference(), controller_hz=30.0)(**request)


@pytest.mark.parametrize("bad", ["future", "duplicate", "unsorted"])
def test_invalid_history_timestamps_are_rejected(bad):
    request = _request(48)
    records = list(request["history"])
    if bad == "future":
        records.append(RTCObservation(49, _image(49)))
    elif bad == "duplicate":
        records.insert(4, records[3])
    else:
        records[0], records[1] = records[1], records[0]
    request["history"] = tuple(records)
    with pytest.raises(ValueError, match="history observations"):
        RTCObservedBatchBuilder(_reference(), controller_hz=30.0)(**request)


@pytest.mark.parametrize("bad", ["float_image", "image_shape", "state_shape", "missing_anchor", "nonfinite_state"])
def test_invalid_fresh_measurement_is_rejected(bad):
    request = _request(16)
    if bad == "float_image":
        request["image"] = request["image"].float()
    elif bad == "image_shape":
        request["image"] = torch.zeros(3, 8, 8, dtype=torch.uint8)
    elif bad == "state_shape":
        request["state"] = torch.zeros(54)
    elif bad == "missing_anchor":
        request["state_mask"][0] = False
    else:
        request["state"][0] = float("nan")
    with pytest.raises(ValueError):
        RTCObservedBatchBuilder(_reference(), controller_hz=30.0)(**request)


@pytest.mark.parametrize("bad", ["fps", "geometry", "frame_ids", "preprocessed", "dtype", "statistics", "normalizer"])
def test_incompatible_reference_is_rejected(bad):
    reference = _reference()
    if bad == "fps":
        reference["conditioning_fps_action"] = torch.tensor([25.0])
    elif bad == "geometry":
        reference["sequence_plan"][0].causal_action_metadata.statistics.provenance["geometry"]["video_stride"] = 2
    elif bad == "frame_ids":
        reference["sequence_plan"][0].causal_action_metadata.action_frame_ids[0] = 0
    elif bad == "preprocessed":
        reference["is_preprocessed"] = True
    elif bad == "dtype":
        reference["video"][0] = reference["video"][0].float()
    elif bad == "statistics":
        reference["sequence_plan"][0].causal_action_metadata.statistics.state.low[0] = float("nan")
    else:
        reference["action_processing_record"] = [ActionProcessingRecord(32, None)]
    with pytest.raises(ValueError):
        RTCObservedBatchBuilder(reference, controller_hz=30.0)
