# SPDX-License-Identifier: OpenMDW-1.1
"""Use metadata valid lengths at the real model loss entry without allocating a backbone."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.omni_mot_causal_action_model import OmniMoTCausalActionModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


@pytest.fixture
def model():
    model = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
    torch.nn.Module.__init__(model)
    model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
    model.config = SimpleNamespace(
        vision_gen=True,
        action_gen=True,
        sound_gen=False,
        causal_action_log_loss_by_mode=True,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            image_loss_scale=None,
            loss_scale=2.0,
            action_loss_weight=3.0,
        ),
    )
    return model


def loss_batch(real_actions=6, output_actions=64, mode="policy"):
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    frames = geometry.num_latent_frames(output_actions + 1)
    ac = torch.ones(output_actions, 1) if mode == "forward_dynamics" else torch.zeros(output_actions, 1)
    vc = torch.ones(frames, 1, 1) if mode == "inverse_dynamics" else torch.zeros(frames, 1, 1)
    vc[0] = 1
    channels = torch.ones(output_actions, 3, dtype=torch.bool)
    channels[:, -1] = False
    an, vn = torch.where(~ac.flatten().bool())[0], torch.where(~vc.flatten().bool())[0]
    return PackedSequence(
        action=ModalityData(
            condition_mask=[ac],
            tokens=[torch.ones(output_actions, 3)],
            token_shapes=[(output_actions,)],
            noisy_frame_indexes=[an],
            mse_loss_indexes=an,
            raw_action_dim=None,
        ),
        vision=ModalityData(
            condition_mask=[vc],
            tokens=[torch.ones(2, frames, 2, 3)],
            token_shapes=[(frames, 2, 3)],
            noisy_frame_indexes=[vn],
            mse_loss_indexes=vn,
        ),
        causal_action_metadata=[
            SimpleNamespace(
                action_mask=channels,
                real_action_steps=real_actions,
                num_valid_video_latents=1 + (real_actions // 4 + 3) // 4,
                action_frame_ids=torch.arange(1, frames).repeat_interleave(16),
            )
        ],
    )


def calculate(model, packed, *, mode="policy", weighted=False, predictions=None):
    flow = SimpleNamespace(train_time_weight=lambda ts, kwargs: ts + 1 if weighted else torch.ones_like(ts))
    model.rectified_flow_video = model.rectified_flow_action = flow
    if predictions is None:
        predictions = {
            name: torch.full_like(getattr(packed, name).tokens[0], 3.0, requires_grad=True)
            for name in ("action", "vision")
        }
    noised = SimpleNamespace(
        vt_target_vision=[torch.ones_like(predictions["vision"])],
        vt_target_action=[torch.ones_like(predictions["action"])],
    )
    total, losses = model._compute_losses(
        {f"preds_{name}": [pred] for name, pred in predictions.items()},
        packed,
        noised,
        torch.arange(packed.vision.token_shapes[0][0], dtype=torch.float32)[None],
        False,
        causal_action_modes=[mode],
        action_sample_indices=[0],
    )
    return total, losses, predictions


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
@pytest.mark.parametrize("real_actions,valid_latents", [(1, 1), (6, 2), (20, 3), (64, 5)])
def test_padding_predictions_do_not_affect_loss_or_gradients(model, real_actions, valid_latents, mode):
    packed = loss_batch(real_actions=real_actions, mode=mode)
    original_conditions = {name: getattr(packed, name).condition_mask[0].clone() for name in ("action", "vision")}
    total, losses, predictions = calculate(model, packed, mode=mode)
    changed = {}
    for name, pred in predictions.items():
        condition = original_conditions[name]
        active = (1 - condition).expand_as(pred).bool().clone()
        if name == "action":
            active[real_actions:] = False
            active[:, -1] = False
        else:
            active[:, valid_latents:] = False
        value = losses[f"flow_matching_loss_{name}"]
        torch.testing.assert_close(value, torch.tensor(4.0 if active.any() else 0.0))
        value.backward()
        assert not pred.grad[~active].any()
        if active.any():
            assert pred.grad[active].abs().sum() > 0
        changed[name] = pred.detach().masked_fill(~active, 100).requires_grad_()
        torch.testing.assert_close(getattr(packed, name).condition_mask[0], condition)
    changed_total, changed_losses, _ = calculate(model, packed, mode=mode, predictions=changed)
    torch.testing.assert_close(changed_total, total)
    for name in predictions:
        changed_losses[f"flow_matching_loss_{name}"].backward()
        torch.testing.assert_close(changed[name].grad, predictions[name].grad)


@pytest.mark.parametrize("weighted", [False, True])
def test_appending_pure_padding_preserves_loss_and_active_gradients(model, weighted):
    results = []
    for length in (32, 96):
        packed = loss_batch(output_actions=length)
        total, losses, predictions = calculate(model, packed, weighted=weighted)
        total.backward()
        results.append((total.detach(), losses, predictions))
    torch.testing.assert_close(results[0][0], results[1][0])
    for name in ("action", "vision"):
        key = f"flow_matching_loss_{name}"
        torch.testing.assert_close(results[0][1][key], results[1][1][key])
        before, after = results[0][2][name].grad, results[1][2][name].grad
        torch.testing.assert_close(before, after[:32] if name == "action" else after[:, :3])
        torch.testing.assert_close(results[0][1][key], torch.tensor(8.0 if weighted else 4.0))


def test_loss_view_leaves_inputs_and_layout_unchanged(model, monkeypatch):
    packed = loss_batch()
    packed.action.condition_mask[0][:2] = 1
    packed.vision.condition_mask[0][1] = 1
    captured = []
    original_compute = OmniMoTModel._compute_losses

    def capture(self, out_net, loss_packed, *args, **kwargs):
        captured.append(loss_packed)
        return original_compute(self, out_net, loss_packed, *args, **kwargs)

    monkeypatch.setattr(OmniMoTModel, "_compute_losses", capture)
    total, losses, predictions = calculate(model, packed)
    assert losses["flow_matching_loss_vision"] == 0
    total.backward()
    assert not predictions["vision"].grad.any()
    view = captured[0]
    assert view is not packed
    assert (1 - view.action.condition_mask[0]).sum() == 8
    for name in ("action", "vision"):
        original, masked = getattr(packed, name), getattr(view, name)
        assert masked.tokens is original.tokens
        assert masked.noisy_frame_indexes is original.noisy_frame_indexes
        assert masked.mse_loss_indexes is original.mse_loss_indexes
    assert not packed.action.condition_mask[0][6:].any()
    assert not packed.vision.condition_mask[0][2:].any()


def test_legacy_metadata_without_valid_lengths_keeps_previous_supervision(model):
    packed = loss_batch(real_actions=64)
    metadata = packed.causal_action_metadata[0]
    metadata.real_action_steps = metadata.num_valid_video_latents = None
    _, losses, _ = calculate(model, packed)
    assert losses["flow_matching_loss_action"] == losses["flow_matching_loss_vision"] == 4


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_full_range_reader_lengths_reach_actual_model_loss(model, monkeypatch, tmp_path, mode):
    from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset_test import full_sft_sample

    result = full_sft_sample(monkeypatch, tmp_path, mode)
    plan = result["sequence_plan"]
    packed = loss_batch(real_actions=33, mode=mode)
    packed.action.tokens = [result["action"]]
    packed.causal_action_metadata = [plan.causal_action_metadata]
    assert (
        torch.where(packed.action.condition_mask[0].flatten().bool())[0].tolist() == plan.condition_frame_indexes_action
    )
    assert (
        torch.where(packed.vision.condition_mask[0].flatten().bool())[0].tolist() == plan.condition_frame_indexes_vision
    )
    total, losses, predictions = calculate(model, packed, mode=mode)
    for name in ("action", "vision"):
        inactive = mode == ("forward_dynamics" if name == "action" else "inverse_dynamics")
        torch.testing.assert_close(losses[f"flow_matching_loss_{name}"], torch.tensor(0.0 if inactive else 4.0))
    total.backward()
    assert not predictions["action"].grad[33:].any()
    assert not predictions["action"].grad[:, ~plan.causal_action_metadata.action_mask[0]].any()
    assert not predictions["vision"].grad[:, 0].any()
    assert not predictions["vision"].grad[:, 3:].any()


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
@pytest.mark.parametrize("real", [1, 33, 96])
def test_frame_windows_use_shared_padding_loss_and_gradients(model, monkeypatch, tmp_path, mode, real):
    from cosmos_framework.data.generator.action.datasets.frame_window_test import frame_window_adapter, make_reader

    reader, _ = make_reader(
        monkeypatch, lengths=(real + 1,), root=tmp_path, table_backend="parquet", maximum=96, window_mode="frame_window"
    )
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64))
    result = frame_window_adapter(reader, mode=mode)[0]
    metadata = result["sequence_plan"].causal_action_metadata
    packed = loss_batch(real_actions=real, output_actions=len(result["action"]), mode=mode)
    packed.action.tokens = [result["action"]]
    packed.causal_action_metadata = [metadata]
    total, losses, predictions = calculate(model, packed, mode=mode)
    expected_action = 0.0 if mode == "forward_dynamics" else 4.0
    expected_video = 0.0 if mode == "inverse_dynamics" or metadata.num_valid_video_latents == 1 else 4.0
    torch.testing.assert_close(losses["flow_matching_loss_action"], torch.tensor(expected_action))
    torch.testing.assert_close(losses["flow_matching_loss_vision"], torch.tensor(expected_video))
    total.backward()
    assert not predictions["action"].grad[real:].any()
    assert not predictions["action"].grad[:, ~metadata.action_mask[0]].any()
    assert not predictions["vision"].grad[:, metadata.num_valid_video_latents :].any()
