# SPDX-License-Identifier: OpenMDW-1.1
"""Template-width action/state interfaces through causal packing and masked loss."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.data.generator.sequence_packing.causal_action import (
    CausalActionGeometry,
    dense_action_mask,
    expand_action_sequence,
)
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork


class SmallActionNetwork(CausalActionNetwork):
    """Use production interfaces without allocating a pretrained language backbone."""

    def __init__(self, width, add_action_flag=False):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(add_action_flag=add_action_flag)
        self.action_dim, self.hidden_size = width, 16
        self.timestep_scale = 1.0
        self._create_action_interfaces()
        self._init_action_interfaces()
        self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))
        self.time_embedder = nn.Linear(1, self.hidden_size)

    def _embed_packed_timesteps(self, timesteps, packed_seq):
        return self.time_embedder(timesteps[:, None])


def packed_batch(width, mode):
    """Two independent segments, with one and three fixed action blocks."""
    lengths = [32, 96]
    vision, action = ModalityData(), ModalityData()
    metadata, text, positions, sample_lens = [], [], [], []
    vindexes, aindexes, vloss, aloss = [], [], [], []
    offset = 0
    for n in lengths:
        t = 1 + n // 16
        vi = torch.arange(offset + 1, offset + 1 + t)
        ai = torch.arange(offset + 1 + t, offset + 1 + t + n)
        text.append(offset)
        vindexes.append(vi)
        aindexes.append(ai)
        vc = torch.ones(t, 1, 1) if mode == "inverse_dynamics" else torch.zeros(t, 1, 1)
        vc[0] = 1
        ac = torch.ones(n, 1) if mode == "forward_dynamics" else torch.zeros(n, 1)
        vn = torch.where(~vc.flatten().bool())[0]
        an = torch.where(~ac.flatten().bool())[0]
        vloss.append(vi[vn])
        aloss.append(ai[an])
        vision.token_shapes.append((t, 1, 1))
        vision.tokens.append(torch.zeros(1, 1, t, 1, 1))
        vision.condition_mask.append(vc)
        vision.noisy_frame_indexes.append(vn)
        action.token_shapes.append((n,))
        mask = torch.ones(n, width, dtype=torch.bool)
        mask[:, -1] = False
        action.tokens.append(torch.randn(n, width).masked_fill(~mask, 0))
        action.condition_mask.append(ac)
        action.noisy_frame_indexes.append(an)
        action.domain_id.append(torch.zeros(1, dtype=torch.long))
        states = torch.randn(n // 32, width)
        sm = torch.ones_like(states, dtype=torch.bool)
        sm[:, -1] = False
        metadata.append(
            SimpleNamespace(
                action_frame_ids=torch.arange(1, t).repeat_interleave(16),
                block_size=2,
                history_blocks=1,
                states=states,
                state_mask=sm,
                action_mask=mask,
                state_action_indexes=torch.arange(0, n, 32),
                state_latent_indexes=torch.arange(0, n // 16, 2),
            )
        )
        pos = torch.zeros(3, 1 + t + n)
        pos[0, 1 : 1 + t] = torch.arange(t)
        pos[0, 1 + t :] = torch.arange(1, n + 1) / 16
        positions.append(pos)
        sample_lens.append(pos.shape[1])
        offset += pos.shape[1]
    for mod, indexes, losses in ((vision, vindexes, vloss), (action, aindexes, aloss)):
        mod.sequence_indexes = torch.cat(indexes)
        mod.mse_loss_indexes = torch.cat(losses)
        mod.timesteps = torch.full((mod.mse_loss_indexes.numel(),), 0.5)
    return PackedSequence(
        sample_lens=sample_lens,
        split_lens=sample_lens,
        attn_modes=["full", "full"],
        sequence_length=offset,
        text_ids=torch.tensor([1, 2]),
        text_indexes=torch.tensor(text),
        position_ids=torch.cat(positions, dim=1),
        vision=vision,
        action=action,
        causal_action_metadata=metadata,
    )


@pytest.mark.parametrize("add_action_flag", [False, True])
@pytest.mark.parametrize("width", [7, 55])
@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
def test_interfaces_packing_loss_and_backward(width, mode, add_action_flag):
    torch.manual_seed(7)
    net = SmallActionNetwork(width, add_action_flag=add_action_flag)
    original = packed_batch(width, mode)
    packed = expand_action_sequence(original, original.vision.tokens, CausalActionGeometry((2, 2), (1, 1)))
    layout = packed.teacher_forcing.layout
    assert packed.teacher_forcing.state_tokens[1].shape == (3, width)
    visible = dense_action_mask(layout)
    q = layout.gen_query_indexes
    cross_sample = layout.sample_ids[q, None] != layout.sample_ids[None, :]
    assert not visible[cross_sample].any()
    encoded = torch.zeros(packed.sequence_length, net.hidden_size)
    net._encode_action(packed, encoded, torch.float32)
    expected = net.state2llm(torch.cat([m.states.masked_fill(~m.state_mask, 0) for m in packed.causal_action_metadata]))
    expected = expected + net.state_modality_embed + net.time_embedder(torch.zeros(4, 1))
    torch.testing.assert_close(encoded[layout.state_indexes], expected)
    # Small dense attention exercises gradients from action predictions into current block state.
    hidden = encoded.clone()
    hidden[q] = F.scaled_dot_product_attention(
        encoded[q][None, None], encoded[None, None], encoded[None, None], attn_mask=visible
    )[0, 0]
    result = {}
    net._decode_action(packed, hidden, result)
    preds = result["preds_action"]
    assert [list(x.shape) for x in preds] == [[32, width], [96, width]]
    assert all(not x[:, -1].any() for x in preds)
    masks = [1 - (1 - c) * m.action_mask for c, m in zip(packed.action.condition_mask, packed.causal_action_metadata)]
    target = [torch.ones_like(p) for p in preds]
    loss, _ = compute_flow_matching_loss(
        preds,
        target,
        masks,
        torch.full((2, 96), 0.5),
        mode != "forward_dynamics",
        SimpleNamespace(train_time_weight=lambda ts, kwargs: torch.ones_like(ts)),
        dict(device="cpu", dtype=torch.float32),
        normalize_by_active=True,
    )
    assert torch.isfinite(loss)
    if mode == "forward_dynamics":
        assert loss == 0
    else:
        reference = torch.stack([(p[:, :-1] - 1).square().mean() for p in preds]).mean()
        torch.testing.assert_close(loss, reference)
    loss.backward()
    assert net.llm2action.weight.grad is not None
    if mode != "forward_dynamics":
        assert net.state2llm.weight.grad.abs().sum() > 0
        assert not net.state2llm.weight.grad[:, -1].any()
        assert not net.llm2action.weight.grad[-1].any()
        if add_action_flag:
            assert net.action_mask_embed.weight.grad.abs().sum() > 0
            assert net.state_mask_embed.weight.grad.abs().sum() > 0


def test_mask_conditioning_is_zero_initialized_and_can_distinguish_valid_zero_channels():
    torch.manual_seed(17)
    legacy = SmallActionNetwork(55, add_action_flag=False)
    net = SmallActionNetwork(55, add_action_flag=True)
    missing, unexpected = net.load_state_dict(legacy.state_dict(), strict=False)
    assert set(missing) == {"action_mask_embed.weight", "state_mask_embed.weight"}
    assert not unexpected
    packed = packed_batch(55, "policy")
    packed = expand_action_sequence(packed, packed.vision.tokens, CausalActionGeometry((2, 2), (1, 1)))
    reference = torch.zeros(packed.sequence_length, net.hidden_size)
    encoded = torch.zeros_like(reference)
    legacy._encode_action(packed, reference, torch.float32)
    net._encode_action(packed, encoded, torch.float32)
    torch.testing.assert_close(encoded, reference)
    assert not net.action_mask_embed.weight.any()
    assert not net.state_mask_embed.weight.any()

    # Change only validity, keeping the channel's action/state values at zero.
    with torch.no_grad():
        net.action_mask_embed.weight[:, -1] = 2
        net.state_mask_embed.weight[:, -1] = 3
    net._encode_action(packed, encoded, torch.float32)
    before = encoded.clone()
    for metadata, state in zip(packed.causal_action_metadata, packed.teacher_forcing.state_tokens, strict=True):
        metadata.action_mask[:, -1] = True
        metadata.state_mask[:, -1] = True
        state[:, -1] = 0
    net._encode_action(packed, encoded, torch.float32)
    torch.testing.assert_close(
        encoded[packed.action.sequence_indexes] - before[packed.action.sequence_indexes],
        torch.full((128, net.hidden_size), 2.0),
    )
    indexes = packed.teacher_forcing.layout.state_indexes
    torch.testing.assert_close(encoded[indexes] - before[indexes], torch.full((4, net.hidden_size), 3.0))


@pytest.mark.parametrize("add_action_flag", [False, True])
def test_video_only_step_keeps_action_interfaces_in_zero_gradient_graph(add_action_flag):
    net = SmallActionNetwork(55, add_action_flag=add_action_flag)
    packed = SimpleNamespace(action=None)
    hidden = torch.randn(3, net.hidden_size, requires_grad=True)
    encoded = hidden.clone()
    net._encode_action(packed, encoded, torch.float32)
    torch.testing.assert_close(encoded, hidden)
    output = {}
    net._decode_action(packed, hidden, output)
    video_loss = hidden.square().mean()
    loss = video_loss + 0 * sum(x.sum() for x in output["preds_action"])
    loss.backward()
    assert hidden.grad.abs().sum() > 0
    for name, parameter in net.named_parameters():
        if name.startswith("time_embedder"):
            continue
        assert parameter.grad is not None, name
        assert not parameter.grad.any(), name


def test_checkpoint_dimensions_are_not_silently_converted():
    net = SmallActionNetwork(55)
    net.load_state_dict(SmallActionNetwork(55).state_dict())
    # strict=False permits missing keys, but must still reject incompatible tensor shapes.
    with pytest.raises(RuntimeError, match="size mismatch"):
        net.load_state_dict(SmallActionNetwork(80).state_dict(), strict=False)


def test_dcp_warm_start_requires_explicit_projection_skip(tmp_path):
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.checkpoint.dcp import CustomLoadPlanner

    old = SmallActionNetwork(80)
    checkpoint = str(tmp_path / "checkpoint")
    dcp.save({"net": old.state_dict()}, checkpoint_id=checkpoint, no_dist=True)
    new = SmallActionNetwork(55)
    initial = {name: value.clone() for name, value in new.state_dict().items()}
    reader = dcp.FileSystemReader(checkpoint)
    planner = CustomLoadPlanner(allow_partial_load=True)
    planner.set_up_planner({"net": new.state_dict()}, reader.read_metadata())
    with pytest.raises(ValueError, match="[Ss]ize mismatch"):
        planner.create_local_plan()
    skipped = ["action2llm", "llm2action", "state2llm", "state_modality_embed", "action_mask_embed", "state_mask_embed"]
    dcp.load(
        {"net": new.state_dict()},
        checkpoint_id=checkpoint,
        no_dist=True,
        planner=CustomLoadPlanner(keys_to_skip_loading=skipped),
    )
    for name, value in new.state_dict().items():
        expected = initial[name] if any(key in name for key in skipped) else old.state_dict()[name]
        torch.testing.assert_close(value, expected)
