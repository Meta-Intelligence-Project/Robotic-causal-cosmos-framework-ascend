# SPDX-License-Identifier: OpenMDW-1.1
"""Live serving uses real observations and the same local sampling adapter."""

import copy
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.transforms import VideoResize
from cosmos_framework.data.generator.action.video_view import VideoViewConfig
from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.rtc_session import RTCObservation, make_model_planner
from cosmos_framework.inference.causal_action.serving_model import (
    ServingModelPolicy,
    adapter_from_metadata,
    load_policy,
    prepare_camera_observation,
)
from cosmos_framework.inference.causal_action.window_streaming_test import Tokenizer


class Model:
    tensor_kwargs = {"device": "cpu"}
    tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}

    def __init__(self):
        self.net = torch.nn.Linear(1, 1)
        self.tokenizer_vision_gen = Tokenizer()
        self.calls = []

    def generate_samples_from_batch(self, batch, **options):
        assert isinstance(options.get("seed"), list)
        assert all(isinstance(caption, str) for caption in batch["ai_caption"])
        self.calls.append((batch, options))
        metadata = batch["sequence_plan"][0].causal_action_metadata
        current = batch["causal_action_current_block"]
        generated = torch.zeros(len(metadata.action_frame_ids), dtype=torch.bool)
        generated[metadata.action_indexes_for_group(current)] = True
        result = dict(
            action=[torch.zeros_like(metadata.action_mask, dtype=torch.float32)], generated_action_mask=[generated]
        )
        if batch.get("causal_action_rtc") is not None:
            rtc = batch["causal_action_rtc"]
            result["rtc_committed_actions"] = [rtc.previous_actions[: rtc.delay_steps].clone()]
        return result


def policy(*, enable_rtc=True, model=None, **kwargs):
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
    if "history_steps" not in kwargs and "history_blocks" not in kwargs:
        kwargs["history_blocks"] = 1
    return ServingModelPolicy(
        Model() if model is None else model, statistics=stats, adapter=adapter, enable_rtc=enable_rtc, **kwargs
    )


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
    assert options == dict(causal_block_size=2, causal_history_blocks=3, num_steps=10, guidance=1.0, seed=[42])
    assert service.metadata["streaming_vae"] is True
    assert service.metadata["npu_replay_graph_requested"] is False
    assert batch["causal_action_npu_replay_graph"] is False
    assert "causal_action_precomputed_vision_prefix" in batch
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
        history_blocks=service.history_blocks,
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
    assert batch["sequence_plan"][0].causal_action_metadata.anchors[0, 0] == state[0]


def test_repeated_startup_requests_do_not_append_hidden_history_or_mutate_weights():
    service = policy()
    boot = service.infer(reset_request())["actions"]
    request = predict_request(0, rtc=dict(previous_actions=boot, delay_steps=1, beta=5))
    versions = [p._version for p in service.model.net.parameters()]
    one = service.infer(request)["actions"]
    two = service.infer(request)["actions"]
    np.testing.assert_array_equal(one, two)
    assert [p._version for p in service.model.net.parameters()] == versions
    assert service.builder.history_blocks == 1
    assert all(call[0]["causal_action_current_block"] == 3 for call in service.model.calls)


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


def test_repeat_first_has_full_history_at_reset_and_keeps_the_true_initial_image():
    service = policy(history_blocks=1, history_padding="repeat_first")
    service.infer(reset_request())
    batch, _ = service.model.calls[-1]
    assert batch["causal_action_current_block"] == 3
    assert batch["action"][0].shape == (32, 55)
    assert batch["sequence_plan"][0].causal_action_metadata.states.shape == (1, 55)
    assert batch["rtc_observation_ticks"].tolist() == list(range(-32, 1, 4))
    assert batch["rtc_observation_source_ticks"].tolist() == [0] * 9
    request = predict_request(16, history=[dict(tick=t, image=image(t)) for t in range(0, 16, 4)])
    service.infer(request)
    batch, _ = service.model.calls[-1]
    assert batch["rtc_observation_ticks"].tolist() == list(range(-16, 17, 4))
    assert batch["rtc_observation_source_ticks"].tolist() == [0, 0, 0, 0, 0, 4, 8, 12, 16]
    np.testing.assert_array_equal(service.first_image.numpy(), image(0))


def test_streaming_rejects_legacy_available_padding_before_loading_a_tokenizer():
    with pytest.raises(ValueError, match="streaming_vae requires history_padding=repeat_first"):
        policy(model=object(), history_padding="available")


def test_serving_rejects_disabling_streaming_before_accessing_the_model():
    with pytest.raises(ValueError, match="requires streaming_vae=True"):
        policy(model=object(), streaming_vae=False)


@pytest.mark.parametrize("flag", ["--no-streaming-vae", "--no-streaming_vae"])
def test_cli_rejects_disabling_streaming(flag):
    from cosmos_framework.inference.causal_action.server import parser

    required = ["--checkpoint", "weights", "--processor", "processor", "--vae", "vae", "--statistics", "stats"]
    with pytest.raises(SystemExit) as error:
        parser().parse_args([*required, flag])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", [None, "--streaming-vae", "--streaming_vae"])
def test_cli_always_enables_streaming_and_keeps_rtc_independent(flag):
    from cosmos_framework.inference.causal_action.server import parser

    required = ["--checkpoint", "weights", "--processor", "processor", "--vae", "vae", "--statistics", "stats"]
    arguments = required + ([] if flag is None else [flag])
    args = parser().parse_args(arguments)
    assert args.streaming_vae is True
    assert args.enable_rtc is False
    assert parser().parse_args([*arguments, "--enable-rtc"]).enable_rtc is True


@pytest.mark.parametrize("flag", [None, "--npu-replay-graph", "--npu_replay_graph"])
def test_npu_graph_cli_is_opt_in_and_can_be_combined_with_rtc(flag):
    from cosmos_framework.inference.causal_action.server import parser

    required = ["--checkpoint", "weights", "--processor", "processor", "--vae", "vae", "--statistics", "stats"]
    arguments = required + ([] if flag is None else [flag])
    args = parser().parse_args(arguments)
    assert args.npu_replay_graph is (flag is not None)
    assert args.enable_rtc is False
    combined = parser().parse_args([*arguments, "--enable-rtc"])
    assert combined.npu_replay_graph is (flag is not None)
    assert combined.enable_rtc is True


@pytest.mark.parametrize("requested", [False, True])
def test_npu_graph_request_reaches_ordinary_and_guided_batches_without_disabling_rtc(requested):
    service = policy(npu_replay_graph=requested, enable_rtc=True)
    bootstrap = service.infer(reset_request())["actions"]
    assert service.metadata["npu_replay_graph_requested"] is requested
    ordinary, _ = service.model.calls[-1]
    assert ordinary["causal_action_npu_replay_graph"] is requested
    assert "causal_action_rtc" not in ordinary
    target = bootstrap[16:]
    result = service.infer(
        predict_request(
            16,
            history=[dict(tick=t, image=image(t)) for t in range(0, 16, 4)],
            rtc=dict(previous_actions=target, delay_steps=8, beta=5),
        )
    )
    guided, _ = service.model.calls[-1]
    assert guided["causal_action_npu_replay_graph"] is requested
    assert isinstance(guided["causal_action_rtc"], RTCPrefix)
    assert guided["causal_action_rtc"].delay_steps == 8
    np.testing.assert_array_equal(result["actions"][:8], target[:8])


def test_loader_rejects_disabling_streaming_before_loading_assets():
    with pytest.raises(ValueError, match="requires streaming_vae=True"):
        load_policy(SimpleNamespace(streaming_vae=False))


@pytest.mark.parametrize("graph_option", [None, False, True])
def test_loader_omitting_streaming_option_uses_streaming_without_loading_weights(monkeypatch, graph_option):
    from cosmos_framework.utils import distributed

    fixture = policy()
    calls = []

    def load_checkpoint(*args, **kwargs):
        calls.append((args, kwargs))
        return Model(), None

    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WAN_VAE_PATH", "unused")
    monkeypatch.setattr(distributed, "init", lambda: None)
    monkeypatch.setattr(BlockStatistics, "load", lambda path: fixture.statistics)
    monkeypatch.setattr(RobotwinAdapter, "from_statistics", lambda *args: fixture.adapter)
    monkeypatch.setitem(
        sys.modules,
        "cosmos_framework.utils.generator.model_loader",
        SimpleNamespace(load_model_from_checkpoint=load_checkpoint),
    )
    args = SimpleNamespace(
        vae="vae",
        statistics="stats",
        checkpoint="weights",
        processor="processor",
        history_steps=32,
        execute_steps=16,
        fps=30,
        resolution="256",
        steps=10,
        guidance=1,
        enable_rtc=False,
        seed=42,
    )
    if graph_option is not None:
        args.npu_replay_graph = graph_option
    service = load_policy(args)
    assert service.metadata["streaming_vae"] is True
    assert service.window_encoder is not None
    assert service.enable_rtc is False
    assert service.npu_replay_graph is (graph_option is True)
    service.infer(reset_request())
    assert service.model.calls[-1][0]["causal_action_npu_replay_graph"] is (graph_option is True)
    assert len(calls) == 1
    assert calls[0][1]["keys_to_skip_loading"] == ["net_ema."]


@pytest.mark.parametrize("past,horizon,duration", [(48, 32, "2s"), (1024, 32, "35s"), (48, 48, "3s")])
def test_compact_caption_matches_the_full_training_video_window(past, horizon, duration):
    service = policy(history_steps=past, action_horizon=horizon, history_padding="repeat_first")
    service.infer(reset_request())
    batch, _ = service.model.calls[-1]
    assert json.loads(batch["ai_caption"][0])["duration"] == duration
    assert batch["action"][0].shape == (horizon, 55)
    assert batch["sequence_plan"][0].causal_action_metadata.states.shape == (1, 55)


def test_cli_history_units_remain_independent_of_action_horizon_and_legacy_alias():
    from cosmos_framework.inference.causal_action.server import parser

    required = ["--checkpoint", "weights", "--processor", "processor", "--vae", "vae", "--statistics", "stats"]
    args = parser().parse_args(required + ["--history_steps", "48", "--action-horizon", "48"])
    assert args.history_steps == 48 and args.action_horizon == 48 and args.history is None
    legacy = parser().parse_args(required + ["--history", "32", "--action_horizon", "48"])
    service = policy(
        history_blocks=legacy.history, action_horizon=legacy.action_horizon, history_padding="repeat_first"
    )
    assert service.metadata["history_steps"] == 1024
    assert service.metadata["action_horizon"] == 48
    with pytest.raises(ValueError, match="not both"):
        policy(history_blocks=32, history_steps=1024)


def test_async_schedule_metadata_and_cli_do_not_restrict_direct_rpc_horizon():
    from cosmos_framework.inference.causal_action.server import parser

    required = ["--checkpoint", "weights", "--processor", "processor", "--vae", "vae", "--statistics", "stats"]
    args = parser().parse_args(required + ["--execute-steps", "16", "--handoff-steps", "26"])
    assert args.execute_steps == 16 and args.handoff_steps == 26
    assert policy().metadata["handoff_steps"] == 24
    assert policy(handoff_steps=26).metadata["handoff_steps"] == 26
    # S=H可直接调用模型；只有客户端选择async时才需要S<R<=H。
    short = policy(history_steps=0, action_horizon=16, execute_steps=16, history_padding="repeat_first")
    assert short.metadata["handoff_steps"] is None
    assert short.infer(reset_request())["actions"].shape == (16, 55)
    with pytest.raises(ValueError, match="handoff_steps"):
        policy(handoff_steps=33)


def test_streaming_prefix_is_prepared_during_observation_delivery_and_passed_to_the_model():
    service = policy()
    service.infer(reset_request())
    service.prepare_window(dict(anchor_tick=16))
    assert service.window_encoder.count == 5
    for tick in range(1, 17):
        service.observe(dict(tick=tick, image=image(tick)))
    assert len(service.model.calls) == 1
    request = predict_request(16, history=[dict(tick=t, image=image(t)) for t in range(0, 16, 4)])
    result = service.infer(request)
    batch, _ = service.model.calls[-1]
    prefix = batch["causal_action_precomputed_vision_prefix"][0]
    assert prefix.shape == (1, 3, 1, 1)
    assert result["streaming_timing"]["sampled_frames_encoded_before_predict"] == 9
    assert not result["streaming_timing"]["streaming_window_rebuilt"]
    assert "causal_action_rtc" not in batch
