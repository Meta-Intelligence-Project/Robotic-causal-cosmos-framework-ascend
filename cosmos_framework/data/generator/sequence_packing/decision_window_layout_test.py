# SPDX-License-Identifier: OpenMDW-1.1
"""Single-copy decision windows share causal TND and both prefix cache paths."""

import copy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader
from cosmos_framework.data.generator.sequence_packing.causal_action import (
    CausalActionGeometry,
    CausalActionLayout,
    dense_action_mask,
    expand_action_sequence,
)
from cosmos_framework.data.generator.sequence_packing.causal_action_prefix_visibility_test import _tnd_mask
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action.prefix_cache import action_prefix_cache
from cosmos_framework.inference.causal_action.prefix_cache_test import _network, _next_step
from cosmos_framework.inference.causal_action.rtc_prefix_cache import action_rtc_prefix_cache


@pytest.fixture(autouse=True)
def _cpu_attention(monkeypatch):
    from cosmos_framework.model.attention import attention

    def cpu_attention(*args, backend, **kwargs):
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)


def _pack(*, modes=("policy",), windows=(True,), observed=5, future=2, per_latent=2, spatial=(2, 2)):
    vision, action = ModalityData(), ModalityData()
    lengths, positions, metadata, clean_tokens, text, ti = [], [], [], [], [], []
    vi, ai, vl, al = [], [], [], []
    offset = 0
    h, w = spatial
    t = observed + future
    for sample, (mode, window) in enumerate(zip(modes, windows, strict=True)):
        u = 4
        n = future * per_latent if window else (t - 1) * per_latent
        nv = t * h * w
        vindex, aindex = torch.arange(offset + u, offset + u + nv), torch.arange(offset + u + nv, offset + u + nv + n)
        vc, ac = torch.zeros(t, 1, 1), torch.zeros(n, 1)
        vc[:observed] = 1
        if mode == "inverse_dynamics":
            vc[:] = 1
        if mode == "forward_dynamics":
            ac[:] = 1
        vn, an = torch.where(~vc.flatten().bool())[0], torch.where(~ac.flatten().bool())[0]
        clean = torch.randn(1, 2, t, h, w, generator=torch.Generator().manual_seed(31 + sample), requires_grad=True)
        video = torch.randn(1, 2, t, h, w, generator=torch.Generator().manual_seed(51 + sample))
        with torch.no_grad():
            video[:, :, :observed] = clean[:, :, :observed]
        state_count = 1 if window else (t - 1) // future
        states = torch.randn(state_count, 7)
        sm, am = torch.ones_like(states, dtype=torch.bool), torch.ones(n, 7, dtype=torch.bool)
        sm[:, -1] = False
        am[:, -1] = False
        common = dict(
            block_size=1 if window else future,
            history_blocks=1 if window else 2,
            action_frame_ids=(observed + torch.arange(n) // per_latent)
            if window
            else torch.arange(1, t).repeat_interleave(per_latent),
            action_mask=am,
            states=states,
            state_mask=sm,
            state_action_indexes=torch.tensor([0]) if window else torch.arange(state_count) * future * per_latent,
            anchors=states.clone(),
            state_latent_indexes=torch.tensor([observed - 1]) if window else torch.arange(state_count) * future,
        )
        m = BlockStateMetadata(
            **common,
            **(
                dict(
                    layout_mode="decision_window",
                    history_steps=(observed - 1) * per_latent,
                    action_horizon=n,
                    observed_latents=observed,
                    future_latents=future,
                    actions_per_latent_frame=per_latent,
                )
                if window
                else {}
            ),
        )
        metadata.append(m)
        clean_tokens.append(clean)
        vi.append(vindex)
        ai.append(aindex)
        vl.append(vindex.reshape(t, h * w)[vn].flatten())
        al.append(aindex[an])
        vision.tokens.append(video)
        vision.token_shapes.append((t, h, w))
        vision.condition_mask.append(vc)
        vision.noisy_frame_indexes.append(vn)
        action.tokens.append(torch.randn(n, 7).masked_fill(~am, 0))
        action.token_shapes.append((n,))
        action.condition_mask.append(ac)
        action.noisy_frame_indexes.append(an)
        action.domain_id.append(torch.zeros(1, dtype=torch.long))
        text.append(torch.arange(11, 11 + u))
        ti.append(torch.arange(offset, offset + u))
        length = u + nv + n
        pos = torch.zeros(3, length, dtype=torch.float32)
        pos[:, :u] = torch.arange(u)
        pos[:, u : u + nv] = torch.stack(
            torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        ).flatten(1)
        pos[0, u + nv :] = m.action_frame_ids
        positions.append(pos)
        lengths.append(length)
        offset += length
    for mod, indexes, losses in ((vision, vi, vl), (action, ai, al)):
        mod.sequence_indexes = torch.cat(indexes)
        mod.mse_loss_indexes = torch.cat(losses)
        mod.timesteps = torch.full((mod.mse_loss_indexes.numel(),), 0.9)
    source = PackedSequence(
        sample_lens=lengths,
        split_lens=[v for length in lengths for v in (4, length - 4)],
        attn_modes=[v for _ in lengths for v in ("causal", "full")],
        sequence_length=sum(lengths),
        text_ids=torch.cat(text),
        text_indexes=torch.cat(ti),
        position_ids=torch.cat(positions, dim=1),
        vision=vision,
        action=action,
        causal_action_metadata=metadata,
    )
    geometry = CausalActionGeometry(
        (future,) * len(windows),
        tuple(observed if window else 2 for window in windows),
        (False,) * len(windows),
        tuple(windows),
    )
    return source, clean_tokens, expand_action_sequence(source, clean_tokens, geometry)


@pytest.mark.parametrize("observed,future,per_latent", [(1, 2, 16), (5, 3, 16), (65, 2, 16)])
def test_decision_window_has_single_copy_of_visuals_horizon_actions_and_one_state(observed, future, per_latent):
    source, _, packed = _pack(observed=observed, future=future, per_latent=per_latent)
    layout = packed.teacher_forcing.layout
    assert isinstance(layout, CausalActionLayout) and layout.decision_window
    assert packed.sequence_length == 4 + (observed + future) * 4 + future * per_latent + 1
    assert not layout.clean_token_indexes.numel()
    assert len(layout.state_indexes) == 1
    indexes = torch.cat((packed.text_indexes, packed.vision.sequence_indexes, packed.action.sequence_indexes))
    torch.testing.assert_close(layout.source_sequence_indexes[indexes], torch.arange(source.sequence_length))
    assert indexes.unique().numel() == source.sequence_length
    assert packed.vision.sequence_indexes.numel() == (observed + future) * 4
    assert packed.action.sequence_indexes.numel() == future * per_latent
    state = layout.state_indexes[0]
    boundary = source.vision.sequence_indexes[(observed - 1) * 4]
    torch.testing.assert_close(packed.position_ids[0, state], source.position_ids[0, boundary])
    groups = layout.block_ids[packed.vision.sequence_indexes].reshape(observed + future, 4)
    torch.testing.assert_close(groups[:, 0], torch.tensor(list(range(observed)) + [observed] * future))
    assert layout.block_ids[state] == observed


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
@pytest.mark.parametrize("windows", [(True,), (True, False), (False, True)])
def test_history_latents_are_causal_and_tnd_matches_dense_without_cross_sample_visibility(mode, windows):
    _, _, packed = _pack(modes=(mode,) * len(windows), windows=windows)
    layout = packed.teacher_forcing.layout
    mask = dense_action_mask(layout)
    torch.testing.assert_close(_tnd_mask(layout), mask)
    qi = layout.gen_query_indexes
    assert not mask[layout.sample_ids[qi, None] != layout.sample_ids[None, :]].any()
    for sample, window in enumerate(windows):
        if not window:
            continue
        sample_keys = layout.sample_ids == sample
        history = sample_keys & (layout.stream_ids == 0)
        history_queries = (layout.sample_ids[qi] == sample) & (layout.stream_ids[qi] == 0)
        future = layout.block_ids[None, :] > layout.block_ids[qi, None]
        assert not mask[history_queries][future[history_queries]].any()
        assert not mask[history_queries][
            :, (layout.stream_ids == 1) | (layout.stream_ids == 2) | (layout.stream_ids == 3)
        ].any()
        current = (layout.sample_ids[qi] == sample) & (layout.block_ids[qi] == 5) & (layout.stream_ids[qi] != 3)
        assert current.any() and mask[current][:, history].all()
        state = (
            layout.state_indexes[sample] if all(windows) else torch.where(sample_keys & (layout.stream_ids == 3))[0][0]
        )
        state_query = torch.where(qi == state)[0][0]
        expected = sample_keys & ((layout.stream_ids == -1) | (torch.arange(len(sample_keys)) == state))
        torch.testing.assert_close(mask[state_query], expected)


@pytest.mark.parametrize("windows", [(True,), (True, False), (False, True)])
def test_network_projects_visuals_once_per_window_and_never_consumes_its_gt_clean_copy(monkeypatch, windows):
    _, clean, packed = _pack(modes=("policy",) * len(windows), windows=windows)
    network = _network()
    calls = []
    handle = network.vae2llm.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    try:
        output = network(packed)
    finally:
        handle.remove()
    expected = [28 * len(windows)] + ([28 * windows.count(False)] if not all(windows) else [])
    assert calls == expected
    loss = sum(x.square().sum() for x in output["preds_action"])
    loss.backward()
    for window, target in zip(windows, clean, strict=True):
        if window:
            assert target.grad is None
        else:
            assert target.grad is not None and target.grad.abs().sum() > 0


def _current_output(network, sequence):
    result = network(sequence)
    return torch.cat((result["preds_vision"][0][:, :, 5:].flatten(), result["preds_action"][0].flatten()))


def test_compact_layout_reuses_ordinary_prefix_cache_across_two_denoising_steps():
    _, _, first = _pack()
    steps = [first, _next_step(first)]
    network = _network(normalized_und_keys=True)
    with torch.no_grad():
        expected = [_current_output(network, sequence) for sequence in steps]
        with action_prefix_cache(network) as cache:
            cache["key"] = (5, "cond")
            actual = [_current_output(network, sequence) for sequence in steps]
            assert cache["stats"]["prefill_calls"] == 1 and cache["stats"]["replay_calls"] == 1
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=3e-6, rtol=2e-5)


def test_compact_layout_keeps_real_joint_input_vjps_with_gradient_safe_prefix_cache():
    _, _, sequence = _pack()
    network = _network(normalized_und_keys=True).requires_grad_(False)
    nv = sequence.vision.tokens[0][:, :, 5:].numel()
    noise = torch.cat((sequence.vision.tokens[0][:, :, 5:].flatten(), sequence.action.tokens[0].flatten())).detach()

    def evaluate(x):
        changed = copy.deepcopy(sequence)
        changed.teacher_forcing.layout = sequence.teacher_forcing.layout
        changed.vision.tokens[0][:, :, 5:] = x[:nv].reshape(1, 2, 2, 2, 2)
        changed.action.tokens = [x[nv:].reshape(4, 7)]
        return _current_output(network, changed)

    references = []
    for scale in (1.0, -0.4):
        x = (noise * scale).requires_grad_(True)
        expected = evaluate(x)
        weight = torch.linspace(-0.7, 0.8, len(expected))
        expected_vjp = torch.autograd.grad(expected, x, grad_outputs=weight)[0]
        references.append((scale, expected.detach(), expected_vjp))
    with action_rtc_prefix_cache(network) as cache:
        cache["key"] = (5, "cond")
        for scale, expected, expected_vjp in references:
            x = (noise * scale).requires_grad_(True)
            actual = evaluate(x)
            actual_vjp = torch.autograd.grad(actual, x, grad_outputs=weight)[0]
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
            torch.testing.assert_close(actual_vjp, expected_vjp, atol=3e-6, rtol=2e-5)
        assert cache["stats"]["prefill_calls"] == 1 and cache["stats"]["grad_compact_calls"] == 2
    assert expected_vjp[:nv].abs().sum() > 0 and expected_vjp[nv:].abs().sum() > 0


def test_packing_token_estimate_matches_the_actual_compact_layout():
    source, _, packed = _pack(observed=65, per_latent=16)
    loader = PackingDataLoader.__new__(PackingDataLoader)
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 1
    loader.sound_latent_fps = 0
    loader.uniae_chunk_frames = None
    sample = dict(
        video=[torch.zeros(3, 265, 32, 32)],
        text_token_ids=[torch.tensor([11])],
        action=source.action.tokens,
        sequence_plan=SimpleNamespace(causal_action_metadata=source.causal_action_metadata[0]),
    )
    assert loader._compute_num_tokens_per_sample(sample) == packed.sequence_length == 305
