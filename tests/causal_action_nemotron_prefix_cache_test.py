# SPDX-License-Identifier: OpenMDW-1.1
"""Prefix replay through the Nemotron architecture used by the 0930 checkpoint.

These CPU tests cover the real encoder, RMSNorm, default mRoPE, ReLU-squared
MLP, and output heads in FP32 and BF16. They do not substitute for NPU BF16
validation: different matrix shapes may select different device kernels.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from cosmos_framework.inference.causal_action.prefix_cache import action_prefix_cache
from cosmos_framework.inference.causal_action.prefix_cache_test import _assert_current_outputs, _next_step, _sequence
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLTextModel
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)


class _SmallNemotronLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Nemotron3DenseVLTextModel(
            config,
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=True,
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


def _network(dtype, *, depth=3):
    torch.manual_seed(71)
    config = Nemotron3DenseVLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=144,
        num_hidden_layers=depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        mrope_section=[2, 1, 1],
        rope_theta=100_000_000.0,
        layer_norm_epsilon=1e-5,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
    )
    language_model = _SmallNemotronLanguageModel(config)
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
    ).to(dtype=dtype)
    # Diffusion time embedding is computed in FP32 in production as well.
    network.time_embedder.float()
    with torch.no_grad():
        for layer in language_model.model.layers:
            layer.self_attn.k_norm_und_for_gen.weight.copy_(torch.linspace(0.5, 2.0, config.head_dim))
    return network.eval()


def _assert_outputs(actual, expected, sequence, block_index, dtype):
    # Keep the FP32 gate used by the Qwen tests. BF16 is required to match
    # bitwise here; a wider dtype-specific tolerance would hide regressions.
    _assert_current_outputs(actual, expected, sequence, block_index)
    if dtype == torch.bfloat16:
        layout = sequence.teacher_forcing.layout
        active = (layout.stream_ids == 1) & (layout.block_ids == block_index)
        torch.testing.assert_close(
            actual["last_hidden_state"][active], expected["last_hidden_state"][active], atol=0, rtol=0
        )
        start, end = 1 + 2 * block_index, 3 + 2 * block_index
        torch.testing.assert_close(
            actual["preds_vision"][0][:, :, start:end],
            expected["preds_vision"][0][:, :, start:end],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual["preds_action"][0][4 * block_index : 4 * (block_index + 1)],
            expected["preds_action"][0][4 * block_index : 4 * (block_index + 1)],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_index", [0, 1, 2])
@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics", "partial"])
def test_nemotron_prefix_replay_matches_current_outputs(dtype, block_index, mode):
    network = _network(dtype)
    first = _sequence(block_index, mode=mode, partial_conditions=mode == "partial")
    second = _next_step(first)
    third = _next_step(second)
    with torch.no_grad():
        expected = [network(sequence) for sequence in (first, second, third)]
        with action_prefix_cache(network) as controller:
            controller["key"] = (block_index, "cond")
            for sequence, reference in zip((first, second, third), expected, strict=True):
                _assert_outputs(network(sequence), reference, sequence, block_index, dtype)
        assert controller["stats"]["prefill_calls"] == 1
        assert controller["stats"]["replay_calls"] == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nemotron_full_depth_cfg_branches(dtype):
    network = _network(dtype, depth=28)
    cond, uncond = _sequence(2), _sequence(2, num_text=1)
    uncond.text_ids.fill_(43)
    cond_next, uncond_next = _next_step(cond), _next_step(uncond)
    with torch.no_grad():
        references = {"cond": network(cond_next), "uncond": network(uncond_next)}
        with action_prefix_cache(network) as controller:
            for branch, sequence in (("cond", cond), ("uncond", uncond), ("cond", cond_next), ("uncond", uncond_next)):
                controller["key"] = (2, branch)
                output = network(sequence)
                if sequence is cond_next or sequence is uncond_next:
                    _assert_outputs(output, references[branch], sequence, 2, dtype)
        assert controller["stats"]["prefill_calls"] == 2
        assert controller["stats"]["replay_calls"] == 2


@pytest.mark.parametrize("unsupported", ["dynamic_rope", "moe"])
def test_nemotron_prefix_cache_rejects_unsupported_configuration_without_patching(unsupported):
    network = _network(torch.float32)
    model = network.language_model.model
    original_forward = network.language_model.forward
    original_dispatch = [layer.self_attn.dispatch_attention_fn for layer in model.layers]
    if unsupported == "dynamic_rope":
        model.rotary_emb.rope_type = "dynamic"
        message = "fixed default RoPE"
    else:
        model.config.num_experts = 8
        message = "dense language models"
    with pytest.raises(ValueError, match=message):
        with action_prefix_cache(network):
            pytest.fail("Unsupported configuration must fail before entering the cache context")
    assert network.language_model.forward == original_forward
    assert [layer.self_attn.dispatch_attention_fn for layer in model.layers] == original_dispatch


def test_nemotron_prefix_cache_restores_patches_after_prefill_exception():
    network = _network(torch.float32)
    model = network.language_model.model
    original_forward = network.language_model.forward
    original_dispatch = [layer.self_attn.dispatch_attention_fn for layer in model.layers]
    sequence = _sequence(2)
    with torch.no_grad():
        reference = network(sequence)
        # Fail midway through real prefill, after layer KV entries were written.
        with patch.object(model.layers[1].mlp_moe_gen, "forward", side_effect=RuntimeError("intentional")):
            with pytest.raises(RuntimeError, match="intentional"):
                with action_prefix_cache(network) as controller:
                    controller["key"] = (2, "cond")
                    network(sequence)
        assert controller["entries"] == {}
        assert network.language_model.forward == original_forward
        assert [layer.self_attn.dispatch_attention_fn for layer in model.layers] == original_dispatch
        _assert_outputs(network(sequence), reference, sequence, 2, torch.float32)
