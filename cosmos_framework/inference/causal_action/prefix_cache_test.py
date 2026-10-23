# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-prefix reuse through real causal encoders, MoT layers, and output heads."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action.prefix_cache import action_prefix_cache
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLTextModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


class _SmallLanguageModel(nn.Module):
    def __init__(self, config, *, normalized_und_keys):
        super().__init__()
        self.model = Qwen3VLTextModel(
            config,
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=normalized_und_keys,
        )

    def forward(self, pack, **kwargs):
        return self.model(pack, **kwargs)


@pytest.fixture(autouse=True)
def _use_cpu_attention(monkeypatch):
    from cosmos_framework.model.attention import attention

    def cpu_attention(*args, backend, **kwargs):
        del backend
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)


def _network(*, normalized_und_keys=False):
    torch.manual_seed(71)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
    )
    language_model = _SmallLanguageModel(config, normalized_und_keys=normalized_und_keys)
    network = CausalActionNetwork(
        language_model,
        Cosmos3VFMNetworkConfig(
            vlm_config=SimpleNamespace(text_config=config),
            vision_gen=True,
            action_gen=True,
            sound_gen=False,
            action_dim=7,
            latent_patch_size=1,
            latent_channel_size=2,
            max_latent_h=2,
            max_latent_w=2,
            max_latent_t=7,
            joint_attn_implementation="teacher_forcing",
            teacher_forcing_dense_mode="grouped_tnd",
            timestep_scale=1.0,
        ),
    )
    if normalized_und_keys:
        with torch.no_grad():
            for layer in language_model.model.layers:
                layer.self_attn.k_norm_und_for_gen.weight.copy_(torch.linspace(0.5, 2.0, config.head_dim))
    return network.eval()


def _sequence(block_index=2, *, num_text=2, mode="policy", partial_conditions=False):
    rng = torch.Generator().manual_seed(19)
    t, h, w, num_actions = 7, 2, 2, 12
    nv = t * h * w
    vi = torch.arange(num_text, num_text + nv)
    ai = torch.arange(num_text + nv, num_text + nv + num_actions)
    observed = 1 + 2 * block_index
    vcondition = torch.zeros(t, 1, 1)
    vcondition[:observed] = 1
    acondition = torch.zeros(num_actions, 1)
    if mode == "inverse_dynamics":
        vcondition[:] = 1
    if mode == "forward_dynamics":
        acondition[:] = 1
    if partial_conditions:
        vcondition[observed] = 1
        acondition[4 * block_index] = 1
    vnoisy = torch.where(~vcondition.flatten().bool())[0]
    anoisy = torch.where(~acondition.flatten().bool())[0]
    clean_video = torch.randn(1, 2, t, h, w, generator=rng)
    clean_video[:, :, observed:] = 0
    video = torch.randn(1, 2, t, h, w, generator=rng)
    video[:, :, :observed] = clean_video[:, :, :observed]
    action_mask = torch.ones(num_actions, 7, dtype=torch.bool)
    action_mask[:, -1] = False
    states = torch.randn(3, 7, generator=rng)
    state_mask = torch.ones_like(states, dtype=torch.bool)
    state_mask[:, -1] = False
    metadata = SimpleNamespace(
        block_size=2,
        action_frame_ids=torch.arange(1, t).repeat_interleave(2),
        state_action_indexes=torch.arange(0, num_actions, 4),
        state_latent_indexes=torch.tensor([0, 2, 4]),
        states=states,
        state_mask=state_mask,
        action_mask=action_mask,
    )
    vision = ModalityData(
        sequence_indexes=vi,
        timesteps=torch.full((len(vnoisy) * h * w,), 0.9),
        mse_loss_indexes=vi.reshape(t, h * w)[vnoisy].flatten(),
        token_shapes=[(t, h, w)],
        tokens=[video],
        condition_mask=[vcondition],
        noisy_frame_indexes=[vnoisy],
    )
    action = ModalityData(
        sequence_indexes=ai,
        timesteps=torch.full((len(anoisy),), 0.9),
        mse_loss_indexes=ai[anoisy],
        token_shapes=[(num_actions,)],
        tokens=[torch.randn(num_actions, 7, generator=rng).masked_fill(~action_mask, 0)],
        condition_mask=[acondition],
        noisy_frame_indexes=[anoisy],
        domain_id=[torch.zeros(1, dtype=torch.long)],
    )
    length = num_text + nv + num_actions
    position_ids = torch.zeros(3, length, dtype=torch.long)
    position_ids[:, :num_text] = torch.arange(num_text)
    position_ids[:, vi] = torch.stack(
        torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    ).flatten(1)
    position_ids[0, ai] = metadata.action_frame_ids
    original = PackedSequence(
        sample_lens=[length],
        split_lens=[num_text, nv + num_actions],
        attn_modes=["causal", "full"],
        sequence_length=length,
        text_ids=torch.arange(11, 11 + num_text),
        text_indexes=torch.arange(num_text),
        position_ids=position_ids,
        vision=vision,
        action=action,
        causal_action_metadata=[metadata],
    )
    return expand_action_sequence(original, [clean_video], CausalActionGeometry((2,), (2,)))


def _next_step(sequence):
    result = copy.deepcopy(sequence)
    result.teacher_forcing.layout = sequence.teacher_forcing.layout
    result.vision.timesteps.fill_(0.2)
    result.action.timesteps.fill_(0.2)
    result.vision.tokens[0][:, :, result.vision.noisy_frame_indexes[0]] *= -0.7
    # Past noisy action embeddings also change their diffusion timestep. They
    # cannot be reused as fixed conditions, although no current query reads them.
    result.action.tokens[0][result.action.noisy_frame_indexes[0]] *= -0.4
    return result


def _assert_current_outputs(actual, expected, sequence, block_index):
    layout = sequence.teacher_forcing.layout
    active = (layout.stream_ids == 1) & (layout.block_ids == block_index)
    torch.testing.assert_close(
        actual["last_hidden_state"][active], expected["last_hidden_state"][active], atol=2e-6, rtol=1e-5
    )
    start, end = 1 + 2 * block_index, 3 + 2 * block_index
    torch.testing.assert_close(
        actual["preds_vision"][0][:, :, start:end],
        expected["preds_vision"][0][:, :, start:end],
        atol=2e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        actual["preds_action"][0][4 * block_index : 4 * (block_index + 1)],
        expected["preds_action"][0][4 * block_index : 4 * (block_index + 1)],
        atol=2e-6,
        rtol=1e-5,
    )
    assert actual["last_hidden_state"].shape == expected["last_hidden_state"].shape
    assert not actual["preds_action"][0][:, -1].any()


@pytest.mark.parametrize("normalized_und_keys", [False, True])
@pytest.mark.parametrize("block_index", [0, 1, 2])
@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics", "partial"])
def test_prefix_cache_matches_current_velocity_across_timesteps(normalized_und_keys, block_index, mode):
    network = _network(normalized_und_keys=normalized_und_keys)
    first = _sequence(block_index, mode=mode, partial_conditions=mode == "partial")
    second = _next_step(first)
    with torch.no_grad():
        expected_first = network(first)
        expected_second = network(second)
        with action_prefix_cache(network) as controller:
            controller["key"] = (block_index, "cond")
            actual_first = network(first)
            actual_second = network(second)
    _assert_current_outputs(actual_first, expected_first, first, block_index)
    _assert_current_outputs(actual_second, expected_second, second, block_index)
    # Older action predictions may be stale in a target-only inference result;
    # the native sampler consumes only this block's velocity.
    if block_index and mode == "policy":
        assert not torch.allclose(
            expected_first["preds_action"][0][: 4 * block_index],
            expected_second["preds_action"][0][: 4 * block_index],
        )


def test_replay_projects_and_runs_ffn_only_for_current_tokens():
    network = _network(normalized_und_keys=True)
    sequence = _sequence()
    rows = {}
    handles = []
    for index, layer in enumerate(network.language_model.model.layers):
        for name, module in (
            ("q", layer.self_attn.q_proj_moe_gen),
            ("k", layer.self_attn.k_proj_moe_gen),
            ("v", layer.self_attn.v_proj_moe_gen),
            ("o", layer.self_attn.o_proj_moe_gen),
            ("ffn", layer.mlp_moe_gen),
        ):
            key = (index, name)
            rows[key] = []
            handles.append(
                module.register_forward_pre_hook(lambda module, args, key=key: rows[key].append(len(args[0])))
            )
    try:
        with torch.no_grad(), action_prefix_cache(network) as controller:
            controller["key"] = (2, "cond")
            network(sequence)
            network(_next_step(sequence))
    finally:
        for handle in handles:
            handle.remove()
    layout = sequence.teacher_forcing.layout
    active_count = int(((layout.stream_ids == 1) & (layout.block_ids == 2)).sum())
    assert active_count == 12  # 2 frames x 2x2 patches plus 4 action tokens.
    assert all(len(values) == 2 and values[0] > active_count and values[1] == active_count for values in rows.values())


def test_cfg_branches_are_isolated_and_clear_rebuilds_changed_conditions():
    network = _network(normalized_und_keys=True)
    cond = _sequence()
    uncond = _sequence(num_text=1)
    uncond.text_ids.fill_(43)
    cond_next, uncond_next = _next_step(cond), _next_step(uncond)
    changed = copy.deepcopy(cond_next)
    changed.teacher_forcing.clean_vision_tokens[0][:, :, :5] += 2
    changed.teacher_forcing.state_tokens[0][2] += 1
    with torch.no_grad():
        references = {name: network(sequence) for name, sequence in (("cond", cond_next), ("uncond", uncond_next))}
        changed_reference = network(changed)
        with action_prefix_cache(network) as controller:
            for name, sequence in (("cond", cond), ("uncond", uncond), ("cond", cond_next), ("uncond", uncond_next)):
                controller["key"] = (2, name)
                actual = network(sequence)
                if sequence is cond_next or sequence is uncond_next:
                    _assert_current_outputs(actual, references[name], sequence, 2)
            controller["entries"].clear()
            controller["key"] = (2, "cond")
            _assert_current_outputs(network(changed), changed_reference, changed, 2)
            _assert_current_outputs(network(changed), changed_reference, changed, 2)


def test_context_restores_forwards_on_exception_and_does_not_affect_gradients():
    network = _network()
    model = network.language_model.model
    original_model_forward = model.forward
    original_lm_forward = network.language_model.forward
    original_dispatches = [layer.self_attn.dispatch_attention_fn for layer in model.layers]
    with pytest.raises(RuntimeError, match="intentional"):
        with torch.no_grad(), action_prefix_cache(network) as controller:
            controller["key"] = (2, "cond")
            network(_sequence())
            raise RuntimeError("intentional")
    assert model.forward == original_model_forward
    assert network.language_model.forward == original_lm_forward
    assert [layer.self_attn.dispatch_attention_fn for layer in model.layers] == original_dispatches
    output = network(_sequence())
    output["preds_action"][0][8:].square().mean().backward()
    assert network.llm2action.weight.grad is not None
    assert model.layers[0].self_attn.k_proj_moe_gen.weight.grad.abs().sum() > 0


def test_prefix_cache_rejects_grad_enabled_execution():
    network = _network()
    with pytest.raises(RuntimeError, match="inference-only"):
        with action_prefix_cache(network) as controller:
            controller["key"] = (2, "cond")
            network(_sequence())
