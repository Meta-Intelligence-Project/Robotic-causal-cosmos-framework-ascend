# SPDX-License-Identifier: OpenMDW-1.1
"""Live serving uses real observations and the same local sampling adapter."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, prediction_block_ids
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.transforms import VideoResize
from cosmos_framework.data.generator.action.video_view import VideoViewConfig
from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.rtc_session import RTCObservation, make_model_planner
from cosmos_framework.inference.causal_action.serving_model import (
    ServingModelPolicy,
    adapter_from_metadata,
    prepare_camera_observation,
)


class Model:
    tensor_kwargs = {"device": "cpu"}

    def __init__(self):
        self.net = torch.nn.Linear(1, 1)
        self.calls = []

    def generate_samples_from_batch(self, batch, **options):
        assert isinstance(options.get("seed"), list)
        assert all(isinstance(caption, str) for caption in batch["ai_caption"])
        self.calls.append((batch, options))
        metadata = batch["sequence_plan"][0].causal_action_metadata
        current = batch["causal_action_current_block"]
        generated = prediction_block_ids(metadata.action_frame_ids, metadata.block_size) == current
        result = dict(
            action=[torch.zeros_like(metadata.action_mask, dtype=torch.float32)], generated_action_mask=[generated]
        )
        if batch.get("causal_action_rtc") is not None:
            rtc = batch["causal_action_rtc"]
            result["rtc_committed_actions"] = [rtc.previous_actions[: rtc.delay_steps].clone()]
        return result


def policy(*, enable_rtc=True, **kwargs):
    template = ActionStateTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[:6] = mask[7:13] = mask[28:30] = True
    adapter = RobotwinAdapter(template, state_mask=mask, action_mask=mask)
    stats = BlockStatistics(
        Quantiles(-torch.ones(55), torch.ones(55), mask),
        Quantiles(-torch.ones(55), torch.ones(55), mask),
        dict(
            kind="fitted",
            geometry=asdict(CausalBlockGeometry(temporal_compression_factor=4)),
            template_id=template.template_id,
        ),
    )
    return ServingModelPolicy(Model(), statistics=stats, adapter=adapter, enable_rtc=enable_rtc, **kwargs)


def image(tick):
    return np.full((3, 16, 16), tick % 255, dtype=np.uint8)


def reset_request(**kwargs):
    request = dict(
        operation="reset",
        state=np.arange(55, dtype=np.float32) / 100,
        image=image(0),
        image_size=np.array([16, 16, 16, 16], dtype=np.float32),
        prompt="Pick up the cup",
        tick=0,
        revision=0,
    )
    return request | kwargs


def predict_request(tick=48, **kwargs):
    first = tick - min(tick // 32, 32) * 32
    return (
        dict(
            operation="predict",
            state=np.arange(55, dtype=np.float32) / 90,
            image=image(tick),
            tick=tick,
            revision=0,
            history=[dict(tick=t, image=image(t)) for t in range(first, tick, 4)],
        )
        | kwargs
    )


def test_reset_bootstraps_ordinary_inference_from_a_runtime_only_schema():
    service = policy()
    request = reset_request()
    result = service.infer(request)
    assert result["actions"].shape == (32, 55)
    assert result["actions"].dtype == np.float32
    batch, options = service.model.calls[-1]
    metadata = batch["sequence_plan"][0].causal_action_metadata
    assert options == dict(causal_block_size=2, causal_history_blocks=32, num_steps=10, guidance=1.0, seed=[42])
    assert not batch["video"][0][:, 1:].any()
    assert not batch["action"][0].any()
    assert not batch["action_raw"][0].any()
    assert "causal_action_rtc" not in batch
    assert json.loads(batch["ai_caption"][0])["actions"][0]["description"] == "Pick up the cup."
    torch.testing.assert_close(
        metadata.anchors[0], torch.from_numpy(request["state"]).masked_fill(~service.adapter.state_mask, 0)
    )
    assert not service.model.net.training
    assert not any(p.requires_grad for p in service.model.net.parameters())


@pytest.mark.parametrize("guided", [False, True])
def test_service_and_local_model_planner_are_identical_for_the_same_measured_inputs(guided):
    service = policy()
    service.infer(reset_request())
    prefix = np.tile(np.arange(55, dtype=np.float32) / 80, (16, 1))
    request = predict_request(rtc=dict(previous_actions=prefix, delay_steps=4, beta=5) if guided else None)
    before = copy.deepcopy(request)
    remote = service.infer(request)["actions"]
    planner = make_model_planner(
        service.model,
        service.builder,
        template=service.template,
        horizon=32,
        controller_hz=30,
        history_blocks=32,
        seed=[42],
    )
    state = service.template.sanitize(torch.from_numpy(request["state"]), service.adapter.state_mask)
    direct = planner(
        state=state,
        state_mask=service.adapter.state_mask,
        action_mask=service.adapter.action_mask,
        image=torch.from_numpy(request["image"]),
        history=tuple(RTCObservation(row["tick"], torch.from_numpy(row["image"])) for row in request["history"]),
        tick=request["tick"],
        revision=request["revision"],
        rtc=RTCPrefix(torch.from_numpy(prefix), 4, 5) if guided else None,
    )
    np.testing.assert_array_equal(remote, direct.numpy())
    np.testing.assert_array_equal(request["state"], before["state"])
    np.testing.assert_array_equal(request["image"], before["image"])
    batch, _ = service.model.calls[-1]
    assert batch["rtc_observation_ticks"].tolist() == list(range(16, 49, 4))
    assert not batch["video"][0][:, 9:].any()
    assert batch["sequence_plan"][0].causal_action_metadata.anchors[1, 0] == state[0]


def test_repeated_startup_requests_do_not_append_hidden_history_or_mutate_weights():
    service = policy()
    boot = service.infer(reset_request())["actions"]
    request = predict_request(0, rtc=dict(previous_actions=boot, delay_steps=1, beta=5))
    versions = [p._version for p in service.model.net.parameters()]
    one = service.infer(request)["actions"]
    two = service.infer(request)["actions"]
    np.testing.assert_array_equal(one, two)
    assert [p._version for p in service.model.net.parameters()] == versions
    assert service.builder.history_blocks == 32
    assert all(call[0]["causal_action_current_block"] == 0 for call in service.model.calls)


def test_metadata_recreates_exact_native_adapter_without_loading_any_model():
    service = policy()
    rebuilt = adapter_from_metadata(service.metadata)
    assert rebuilt.dofs == service.adapter.dofs
    torch.testing.assert_close(rebuilt.state_mask, service.adapter.state_mask)
    action = np.zeros((32, 55), dtype=np.float32)
    np.testing.assert_array_equal(
        rebuilt.split_actions(action)["joint"], service.adapter.split_actions(action)["joint"]
    )


def test_camera_preprocessing_is_exactly_the_shared_training_composition_and_resize():
    cameras = dict(
        cam_high=np.arange(16 * 18 * 3, dtype=np.uint8).reshape(16, 18, 3),
        cam_left_wrist=np.full((10, 14, 3), 31, dtype=np.uint8),
        cam_right_wrist=np.full((12, 16, 3), 193, dtype=np.uint8),
    )
    prepared, size = prepare_camera_observation(cameras, "256")
    frames = {k: torch.from_numpy(v).permute(2, 0, 1)[None] for k, v in cameras.items()}
    views = VideoViewConfig(cameras=dict(head="cam_high", left="cam_left_wrist", right="cam_right_wrist"))
    expected = VideoResize()({"video": views.compose(frames, 1, viewpoint="concat_view")}, "256")
    assert prepared.dtype == torch.uint8 and prepared.device.type == "cpu"
    torch.testing.assert_close(prepared, expected["video"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(size, expected["image_size"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(image=np.zeros((3, 16, 16), dtype=np.float32)),
        dict(image_size=[17, 16, 16, 16]),
        dict(image_size=[16, 16, 17, 16]),
        dict(prompt=""),
        dict(tick=1),
        dict(revision=1),
    ],
)
def test_reset_rejects_invalid_observation_schema(overrides):
    with pytest.raises(ValueError):
        policy().infer(reset_request(**overrides))


def test_rtc_requires_a_bootstrap_and_server_opt_in_and_complete_actual_history():
    service = policy(enable_rtc=False)
    with pytest.raises(ValueError, match="Reset"):
        service.infer(predict_request())
    boot = service.infer(reset_request())["actions"]
    with pytest.raises(ValueError, match="enabled"):
        service.infer(predict_request(0, rtc=dict(previous_actions=boot, delay_steps=1)))
    with pytest.raises(ValueError, match="missing real"):
        service.infer(predict_request(history=[]))
    service.clear()
    assert service.builder is service.planner is None
