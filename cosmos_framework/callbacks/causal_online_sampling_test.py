# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample, causal_preview_action_mse
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


@pytest.mark.parametrize("scenario", ["normal", "partial", "multiple", "empty", "failure"])
def test_causal_callback_uses_ti2v_without_mutating_training_batch(monkeypatch, scenario):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    video = (torch.arange(17.0) + 42).reshape(1, 1, 17, 1, 1)
    original_plan = SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
    batch = {"video": [video], "ai_caption": ["test"], "sequence_plan": [original_plan]}
    if scenario == "multiple":
        batch = {key: value * 2 for key, value in batch.items()}
    if scenario == "empty":
        batch["ai_caption"] = [""]
    calls = []
    model = SimpleNamespace(
        net=torch.nn.Linear(1, 1),
        config=SimpleNamespace(causal_training_strategy="teacher_forcing"),
        input_video_key="video",
        input_caption_key="ai_caption",
        tokenizer_vision_gen=SimpleNamespace(
            get_latent_num_frames=lambda frames: 1 + (frames - 1) // 4,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
        get_data_and_condition=lambda data: SimpleNamespace(
            raw_state_vision=data["video"], x0_tokens_vision=data["video"]
        ),
        decode=lambda latent: latent,
    )

    def generate(data, **kwargs):
        calls.append((data, kwargs))
        assert data["sequence_plan"][0].condition_frame_indexes_vision == [0]
        assert not model.net.training
        assert len(data["video"]) == len(data["sequence_plan"]) == 1
        assert torch.count_nonzero(data["video"][0][..., 1:, :, :]) == 0
        torch.testing.assert_close(data["video"][0][..., :1, :, :], video[..., :1, :, :])
        if scenario == "failure":
            raise RuntimeError("test sampling failure")
        return {"vision": data["video"]}

    model.generate_samples_from_batch = generate
    callback = EveryNDrawSample(
        1,
        n_viz_sample=2 if scenario == "multiple" else 1,
        guidance=[1.0],
        causal_num_blocks=2,
        causal_block_size=3 if scenario == "partial" else 1,
    )
    monkeypatch.setattr(callback, "run_save", lambda rows, count, name: None)
    if scenario == "failure":
        with pytest.raises(RuntimeError, match="test sampling failure"):
            callback.sample(None, model, batch, None, None, 1)
    else:
        callback.sample(None, model, batch, None, None, 1)
    assert model.net.training
    assert batch["video"][0] is video and video.shape[-3] == 17
    assert original_plan.condition_frame_indexes_vision == []
    if scenario == "empty":
        assert not calls
        return
    assert calls[0][0]["video"][0].shape[-3] == (17 if scenario == "partial" else 9)
    assert calls[0][1]["causal_num_blocks"] == 2
    assert calls[0][1]["seed"] == [1]
    if scenario == "multiple":
        assert len(calls) == 2
        assert calls[1][1]["seed"] == [2]


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_causal_action_preview_skips_cfg(monkeypatch, mode):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    video = torch.zeros(1, 1, 5, 2, 2)
    metadata = SimpleNamespace(block_size=2, history_blocks=4)
    batch = {
        "video": [video],
        "ai_caption": ["pick up the cup"],
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True, causal_action_metadata=metadata)],
        "causal_action_mode": [mode],
    }
    calls = []
    model = SimpleNamespace(
        net=torch.nn.Linear(1, 1),
        config=SimpleNamespace(causal_training_strategy="teacher_forcing"),
        input_video_key="video",
        input_caption_key="ai_caption",
        get_data_and_condition=lambda data: SimpleNamespace(
            raw_state_vision=data["video"], x0_tokens_vision=data["video"]
        ),
        decode=lambda latent: latent,
    )

    def generate(data, **kwargs):
        calls.append(kwargs)
        return {"vision": data["video"]}

    model.generate_samples_from_batch = generate
    callback = EveryNDrawSample(1, guidance=[3.0])
    monkeypatch.setattr(callback, "run_save", lambda rows, count, name: None)
    callback.sample(None, model, batch, None, None, 1)
    assert [call["guidance"] for call in calls] == [1.0]


def test_causal_preview_action_mse_uses_generated_valid_channels():
    predicted = torch.tensor([[0.0, 1.0], [2.0, 9.0]])
    target = torch.zeros(2, 2)
    action_mask = torch.tensor([[True, True], [True, False]])
    # The second step was not denoised, and its invalid channel is excluded either way.
    generated_steps = torch.tensor([True, False])
    assert causal_preview_action_mse(predicted, target, action_mask, generated_steps) == 0.5


def test_causal_preview_action_mse_skips_empty_mask():
    value = torch.zeros(1, 2)
    assert causal_preview_action_mse(value, value, torch.zeros(1, 2, dtype=torch.bool), None) is None


def test_action_preview_logs_mse_only_when_action_is_predicted(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    messages = []
    monkeypatch.setattr(
        "cosmos_framework.callbacks.every_n_draw_sample.log.info",
        lambda message, rank0_only=True: messages.append(message),
    )
    video = torch.zeros(1, 1, 4, 2, 2)
    action = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
    target = torch.zeros(2, 2)
    metadata = SimpleNamespace(
        block_size=2,
        history_blocks=4,
        action_mask=torch.tensor([[True, False], [True, True]]),
    )
    batch = {
        "video": [video],
        "ai_caption": ["pick up the cup"],
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True, causal_action_metadata=metadata)],
        "causal_action_mode": ["policy"],
    }
    model = SimpleNamespace(
        net=torch.nn.Linear(1, 1),
        config=SimpleNamespace(causal_training_strategy="teacher_forcing"),
        input_video_key="video",
        input_caption_key="ai_caption",
        get_data_and_condition=lambda data: SimpleNamespace(
            raw_state_vision=data["video"], x0_tokens_vision=data["video"], x0_tokens_action=[target]
        ),
        decode=lambda latent: latent,
        generate_samples_from_batch=lambda data, **kwargs: {
            "vision": data["video"],
            "action": [action],
            "generated_action_mask": [torch.tensor([True, True])],
        },
    )
    callback = EveryNDrawSample(1, guidance=[1.0])
    monkeypatch.setattr(callback, "run_save", lambda rows, count, name: None)
    callback.sample(None, model, batch, None, None, 7)
    preview = [message for message in messages if message.startswith("CAUSAL_ACTION_PREVIEW")]
    assert len(preview) == 1
    record = json.loads(preview[0].removeprefix("CAUSAL_ACTION_PREVIEW "))
    # Valid elements are 1, 0 and 4. Mean square is (1 + 0 + 16) / 3.
    assert record["iteration"] == 7
    assert record["mode"] == "policy"
    assert record["sample"] == 0
    assert record["guidance"] == 1.0
    assert record["action_mse"] == pytest.approx(17 / 3)

    messages.clear()
    batch["causal_action_mode"] = ["forward_dynamics"]
    callback.sample(None, model, batch, None, None, 7)
    assert not any(message.startswith("CAUSAL_ACTION_PREVIEW") for message in messages)


@pytest.mark.parametrize("kwargs", [{"causal_num_blocks": 0}, {"causal_block_size": 0}, {"causal_history_blocks": 17}])
def test_causal_callback_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        EveryNDrawSample(1, **kwargs)


@torch.no_grad()
def test_ti2v_prefix_encoding_is_independent_of_future_ground_truth():
    encoded_lengths = []

    def encode(video, **kwargs):
        encoded_lengths.append(video.shape[-3])
        return video[..., ::4, :, :].clone()

    model = SimpleNamespace(
        tokenizer_vision_gen=SimpleNamespace(
            is_causal=True,
            get_latent_num_frames=lambda frames: 1 + (frames - 1) // 4,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
        _encode_vision_item=encode,
    )
    first = torch.randn(1, 3, 17, 2, 2)
    second = first.clone()
    second[..., 1:, :, :] = 100
    outputs = [OmniMoTModel._encode_vision_x0_tokens(model, [video], None, [[0]])[0] for video in (first, second)]
    torch.testing.assert_close(outputs[0], outputs[1])
    assert encoded_lengths == [1, 1]
    assert outputs[0].shape[-3] == 5
    assert torch.count_nonzero(outputs[0][..., 1:, :, :]) == 0


@pytest.mark.parametrize(("latent_frames", "valid"), [(3, False), (4, True), (5, True), (6, False)])
def test_causal_request_accepts_only_nonempty_partial_final_block(latent_frames, valid):
    model = SimpleNamespace(
        config=SimpleNamespace(
            action_gen=False,
            sound_gen=False,
            teacher_forcing_block_size_min=1,
            teacher_forcing_block_size_max=4,
            teacher_forcing_history_blocks_min=1,
            teacher_forcing_history_blocks_max=64,
        ),
        parallel_dims=None,
        input_caption_key="ai_caption",
        tokenizer_vision_gen=SimpleNamespace(
            is_causal=True,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
    )
    data = SimpleNamespace(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 3, latent_frames, 2, 2)],
        raw_state_vision=[torch.zeros(1, 3, 1 + (latent_frames - 1) * 4, 2, 2)],
    )
    request = dict(
        data_batch={"ai_caption": ["test"]},
        sequence_plans=[SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])],
        gen_data_clean=data,
        has_noisy_actions=False,
        has_velocity_postprocess_builder=False,
        causal_num_blocks=2,
        causal_block_size=2,
        causal_history_blocks=16,
    )
    if valid:
        assert OmniMoTCausalModel._validate_causal_inference_request(model, **request) == latent_frames
    else:
        with pytest.raises(ValueError, match="prepared latent length"):
            OmniMoTCausalModel._validate_causal_inference_request(model, **request)
