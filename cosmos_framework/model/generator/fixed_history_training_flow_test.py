# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-window samples retain their loss boundaries through real training plumbing."""

import pytest
import torch
from torch.utils.data import DataLoader

from cosmos_framework.data.generator.action.fixed_history_sample import build_fixed_history_sample
from cosmos_framework.data.generator.action.fixed_history_sample_test import sample
from cosmos_framework.data.generator.action.history_window import build_history_window
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.causal_multitask_test import make_model
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.precomputed_vision_prefix_test import _fixed_model, _track_sessions


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _training_sample(history_blocks, mode):
    raw, template, planner, stats = sample()
    frames = 1 + (history_blocks + 1) * 8
    raw["history_window"] = build_history_window(16, history_blocks=history_blocks, history_padding="repeat_first")
    raw["video"] = torch.randint(
        256, (3, frames, 32, 32), generator=torch.Generator().manual_seed(71), dtype=torch.uint8
    )
    encoded, metadata = build_fixed_history_sample(raw, template=template, planner=planner, statistics=stats)
    observed = 1 + 2 * history_blocks
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=list(range(observed + (2 if mode == "inverse_dynamics" else 0))),
        condition_frame_indexes_action=list(range(32 if mode == "forward_dynamics" else 0)),
        action_start_frame_offset=history_blocks * 32 + 1,
        causal_action_metadata=metadata,
    )
    return dict(
        video=encoded["video"],
        action=encoded["action"],
        domain_id=torch.tensor(0),
        text_token_ids=torch.tensor([11, 12]),
        sequence_plan=plan,
        conditioning_fps=encoded["conditioning_fps"],
        conditioning_fps_action=encoded["conditioning_fps_action"],
    )


def _loader(training_sample):
    return PackingDataLoader(
        dataloader=DataLoader([training_sample], batch_size=1, collate_fn=custom_collate_fn),
        dataset_name="fixed_robot",
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=1,
        max_sequence_length=65536,
        max_samples_per_batch=None,
    )


@pytest.mark.parametrize("history_blocks", [1, 32])
@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
def test_collated_fixed_window_training_only_supervises_current_future(monkeypatch, history_blocks, mode):
    training_sample = _training_sample(history_blocks, mode)
    loader = _loader(training_sample)
    batch = next(iter(loader))
    assert batch["sequence_plan"] == [training_sample["sequence_plan"]]
    assert batch["sequence_plan"][0].causal_action_metadata.include_visual_prefix
    model = make_model()
    model.tensor_kwargs = model.tensor_kwargs_fp32
    model.parallel_dims = None
    model.config.max_action_dim = 55
    model.config.diffusion_expert_config.patch_spatial = 1
    model.config.diffusion_expert_config.vision_temporal_position_mode = "latent_index"
    model.config.rectified_flow_training_config.shift = 1
    model.config.teacher_forcing_history_blocks_max = history_blocks
    model.input_video_key, model.input_image_key = "video", "images"
    model.tokenizer_vision_gen = _fixed_model(torch.float32).tokenizer_vision_gen
    model.tokenizer_sound_gen = None
    model.log_enc_time_every_n = 100
    model.llm_special_tokens = dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4)
    flow = RectifiedFlow(velocity_field=None, device="cpu")
    model.rectified_flow_image = model.rectified_flow_video = model.rectified_flow_action = flow
    monkeypatch.setattr(model, "_update_train_stats", lambda data: None)
    monkeypatch.setattr(model, "_load_and_tokenize_text_data", lambda data, iteration: [[11, 12]])
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    sessions = _track_sessions(monkeypatch, model.tokenizer_vision_gen)
    captured = {}

    def predict(*, data_batch_packed, memory):
        captured["packed"] = data_batch_packed
        output = dict(
            preds_vision=[torch.ones_like(t, requires_grad=True) for t in data_batch_packed.vision.tokens],
            preds_action=[torch.ones_like(t, requires_grad=True) for t in data_batch_packed.action.tokens],
        )
        captured["predictions"] = output
        return output

    monkeypatch.setattr(model, "denoise", predict)
    output, loss = model.training_step(batch, iteration=1)
    assert torch.isfinite(loss)
    loss.backward()
    observed = 1 + history_blocks * 2
    vg = captured["predictions"]["preds_vision"][0].grad
    ag = captured["predictions"]["preds_action"][0].grad
    assert not vg[:, :, :observed].any()
    assert ag.shape == (32, 55)  # 不存在历史action槽，所有anchor都指向当前state。
    valid = training_sample["sequence_plan"].causal_action_metadata.action_mask[-32:]
    assert not ag[-32:][~valid].any()
    assert bool(vg[:, :, observed:].any()) is (mode != "inverse_dynamics")
    assert bool(ag[-32:][valid].any()) is (mode != "forward_dynamics")
    assert bool(output["flow_matching_loss_vision"] > 0) is (mode != "inverse_dynamics")
    assert bool(output["flow_matching_loss_action"] > 0) is (mode != "forward_dynamics")
    assert output["sigma"].unique().numel() == 1  # 每个决策窗口只有一个联合sigma。
    assert len(sessions) == 1 and sessions[0].frame_count == 1 + (history_blocks + 1) * 8
    assert sessions[0].closed
    layout = captured["packed"].teacher_forcing.layout
    assert layout.geometry.decision_windows == (True,)
    assert layout.state_indexes.numel() == 1
    assert layout.clean_token_indexes.numel() == 0  # 没有额外复制的clean视频流。
    assert int((layout.stream_ids == 0).sum()) == observed * 4
    # The real estimator includes clean video and state rows and never drops this sample.
    estimated = loader._compute_num_tokens_per_sample(
        {
            **training_sample,
            "video": [training_sample["video"]],
            "action": [training_sample["action"]],
            "text_token_ids": [training_sample["text_token_ids"]],
        }
    )
    expected = 2 + 1 + 2 + (observed + 2) * 4 + 32 + 1
    assert estimated == expected  # 一份完整视觉、32 action、1 state，含文本标记。
    assert captured["packed"].sequence_length == len(captured["packed"].text_indexes) + (observed + 2) * 4 + 32 + 1
    assert captured["packed"].sequence_length <= estimated < 65536
