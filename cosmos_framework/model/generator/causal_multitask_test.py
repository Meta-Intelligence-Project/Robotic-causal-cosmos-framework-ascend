# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise robot-action and video-only paths through a shared causal network."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
from cosmos_framework.model.generator.causal_teacher_forcing_test import _packed_noisy_video
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.mot.causal_action_template_test import packed_batch
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.teacher_forcing_vertical_slice_test import _make_network, _make_sequence
from cosmos_framework.model.generator.omni_mot_causal_action_model import OmniMoTCausalActionModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    from cosmos_framework.model.attention import attention

    def run(*args, backend, **kwargs):
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", run)


def make_model():
    model = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
    nn.Module.__init__(model)
    model.config = OmniMoTModelConfig(
        action_gen=True,
        causal_training_strategy="teacher_forcing",
        joint_attn_implementation="teacher_forcing",
        teacher_forcing_block_size_min=2,
        teacher_forcing_block_size_max=2,
        teacher_forcing_history_blocks_min=1,
        teacher_forcing_history_blocks_max=1,
    )
    model.precision = torch.float32
    model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
    flow = SimpleNamespace(train_time_weight=lambda ts, kwargs: torch.ones_like(ts))
    model.rectified_flow_image = model.rectified_flow_video = model.rectified_flow_action = flow
    return model


def make_network(add_action_flag=False):
    video_network = _make_network()
    config = Cosmos3VFMNetworkConfig(
        **{**video_network.config.to_dict(), "action_gen": True, "action_dim": 55, "add_action_flag": add_action_flag}
    )
    net = CausalActionNetwork(video_network.language_model, config)
    net._init_action_interfaces()
    return net


@pytest.mark.parametrize("add_action_flag", [False, True])
@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics", "video_only"])
def test_shared_network_loss_and_backward_for_each_task(mode, add_action_flag):
    torch.manual_seed(42)
    model, net = make_model(), make_network(add_action_flag)
    if mode == "video_only":
        packed, _, _ = _make_sequence()
    else:
        original = packed_batch(55, mode)
        original.vision.tokens = [torch.randn(1, 2, t, h, w) for t, h, w in original.vision.token_shapes]
        packed = expand_action_sequence(original, original.vision.tokens, CausalActionGeometry((2, 2), (1, 1)))
        packed.action.raw_action_dim = None
    output = net(packed)
    targets = SimpleNamespace(
        vt_target_vision=[torch.ones_like(v) for v in output["preds_vision"]],
        vt_target_action=[torch.ones_like(a) for a in output["preds_action"]],
    )
    original_masks = None if packed.action is None else packed.action.condition_mask
    loss, metrics = model._compute_losses(
        output, packed, targets, torch.full((len(packed.vision.tokens), 128), 0.5), False
    )
    assert torch.isfinite(loss)
    if mode in {"video_only", "forward_dynamics"}:
        assert metrics["flow_matching_loss_action"] == 0
    else:
        assert metrics["flow_matching_loss_action"] > 0
    if mode == "inverse_dynamics":
        assert metrics["flow_matching_loss_vision"] == 0
    if packed.action is not None:
        assert packed.action.condition_mask is original_masks
    loss.backward()
    assert net.language_model.o_proj.weight.grad.abs().sum() > 0
    if mode == "video_only":
        for name, parameter in net.named_parameters():
            if name.startswith(("action", "state", "llm2action")):
                assert parameter.grad is not None, name
                assert not parameter.grad.any(), name


def test_video_geometry_and_packing_do_not_require_action_metadata():
    model = make_model()
    plans = [SequencePlan(has_text=True, has_vision=True)]
    geometry = model.prepare_teacher_forcing_geometry_from_plans([3], plans)
    assert geometry.block_sizes == (2,)
    assert model.config.action_gen is True
    noisy = _packed_noisy_video()
    clean = GenerationDataClean(batch_size=1, is_image_batch=False, x0_tokens_vision=[torch.randn(1, 1, 3, 1, 1)])
    packed = model.post_noise_packing_hook(noisy, clean, geometry)
    assert packed.action is None
    assert packed.causal_action_metadata is None
    assert not hasattr(packed.teacher_forcing, "state_tokens")
    assert packed.sequence_length == noisy.sequence_length + 3


@pytest.mark.parametrize("video_only", [False, True])
def test_latent_packing_noise_sampling_and_loss_for_both_tasks(video_only):
    model, net = make_model(), make_network()
    model.tensor_kwargs = model.tensor_kwargs_fp32
    model.parallel_dims = None
    model.config.max_action_dim = 55
    model.config.diffusion_expert_config.patch_spatial = 1
    model.config.rectified_flow_training_config.shift = 1
    model.tokenizer_vision_gen = SimpleNamespace(temporal_compression_factor=4)
    model.llm_special_tokens = dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4)
    flow = RectifiedFlow(velocity_field=None, device="cpu")
    model.rectified_flow_image = model.rectified_flow_video = model.rectified_flow_action = flow
    if video_only:
        plans = [SequencePlan(has_text=True, has_vision=True)]
        clean = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[torch.randn(1, 2, 3, 1, 1)],
            fps_vision=torch.tensor([7.5]),
        )
    else:
        source = packed_batch(55, "policy")
        plans = [
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=[0],
                action_start_frame_offset=1,
                causal_action_metadata=m,
            )
            for m in source.causal_action_metadata
        ]
        clean = GenerationDataClean(
            batch_size=2,
            is_image_batch=False,
            x0_tokens_vision=[torch.randn(1, 2, t, h, w) for t, h, w in source.vision.token_shapes],
            x0_tokens_action=source.action.tokens,
            action_domain_id=source.action.domain_id,
            fps_vision=torch.full((2,), 7.5),
            fps_action=torch.full((2,), 30.0),
        )
    frames = [v.shape[2] for v in clean.x0_tokens_vision]
    geometry = model.prepare_teacher_forcing_geometry_from_plans(frames, plans)
    timesteps, sigmas = model._get_train_noise_level_vision(
        batch_size=clean.batch_size,
        is_image_batch=False,
        num_vision_latent_frames=frames,
        teacher_forcing_geometry=geometry,
    )
    packed = model._pack_input_sequence(plans, [[11, 12]] * clean.batch_size, clean, timesteps)
    noised = model._add_noise_to_input(clean, packed, sigmas)
    assert (noised.xt_tokens_action is None) == video_only
    model._replace_clean_with_noised(packed, noised)
    packed = model.post_noise_packing_hook(packed, clean, geometry)
    output = net(packed)
    loss, metrics = model._compute_losses(output, packed, noised, timesteps, False)
    assert torch.isfinite(loss)
    assert (metrics["flow_matching_loss_action"] == 0) == video_only
    loss.backward()
    assert net.language_model.o_proj.weight.grad.abs().sum() > 0


def test_mixed_modalities_within_one_pack_are_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="separate training steps"):
        model.prepare_teacher_forcing_geometry_from_plans(
            [3, 3],
            [
                SequencePlan(has_text=True, has_vision=True),
                SequencePlan(has_text=True, has_vision=True, has_action=True),
            ],
        )


def test_alternating_tasks_and_learned_mask_checkpoint_round_trip():
    torch.manual_seed(11)
    model, net = make_model(), make_network(add_action_flag=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, foreach=False)
    for video_only in (False, True, False):
        if video_only:
            packed, _, _ = _make_sequence()
        else:
            source = packed_batch(55, "policy")
            source.vision.tokens = [torch.randn(1, 2, t, h, w) for t, h, w in source.vision.token_shapes]
            packed = expand_action_sequence(source, source.vision.tokens, CausalActionGeometry((2, 2), (1, 1)))
            packed.action.raw_action_dim = None
        output = net(packed)
        targets = SimpleNamespace(
            vt_target_vision=[torch.ones_like(x) for x in output["preds_vision"]],
            vt_target_action=[torch.ones_like(x) for x in output["preds_action"]],
        )
        loss, _ = model._compute_losses(
            output, packed, targets, torch.full((len(packed.vision.tokens), 128), 0.5), False
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert net.action_mask_embed.weight.abs().sum() > 0
    restored = make_network(add_action_flag=True)
    restored.load_state_dict(net.state_dict())
    with torch.no_grad():
        expected, actual = net(packed), restored(packed)
    for a, b in zip(expected["preds_action"], actual["preds_action"], strict=True):
        torch.testing.assert_close(a, b)
