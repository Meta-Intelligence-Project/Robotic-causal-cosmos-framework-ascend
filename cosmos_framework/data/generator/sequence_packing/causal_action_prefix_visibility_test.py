# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-window visual-prefix visibility agrees with production TND scheduling."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import (
    CausalActionGeometry,
    build_action_layout,
    dense_action_mask,
)
from cosmos_framework.model.generator.mot.causal_action_tnd import build_action_tnd_plan


def _layout(histories, flags=()):
    """Use a full history window and both noisy and conditioned current tokens."""
    shapes, actions, vision_conditions, action_conditions, metadata = [], [], [], [], []
    for history in histories:
        blocks = history + 1
        frames, num_actions = 1 + 2 * blocks, 32 * blocks
        shapes.append((frames, 1, 2))
        actions.append(num_actions)
        vc = torch.ones(frames, 1, 1, dtype=torch.bool)
        vc[-2] = False
        ac = torch.ones(num_actions, 1, dtype=torch.bool)
        ac[-32:-1] = False
        vision_conditions.append(vc)
        action_conditions.append(ac)
        metadata.append(
            SimpleNamespace(
                block_size=2,
                action_frame_ids=torch.arange(1, frames).repeat_interleave(16),
                states=torch.zeros(blocks, 1),
                state_action_indexes=torch.arange(blocks) * 32,
            )
        )
    return build_action_layout(
        und_counts=[2 + i for i in range(len(histories))],
        vision_shapes=shapes,
        action_lengths=actions,
        vision_conditions=vision_conditions,
        action_conditions=action_conditions,
        geometry=CausalActionGeometry((2,) * len(histories), tuple(histories), tuple(flags)),
        metadata=metadata,
    )


def _tnd_mask(layout):
    plan = build_action_tnd_plan(layout, device=torch.device("cpu"), max_kv_tokens=96)
    visible = torch.zeros(plan.num_queries, plan.num_keys, dtype=torch.bool)
    visits = torch.zeros(plan.num_queries, dtype=torch.long)
    for chunk in plan.chunks:
        query_start = key_start = 0
        for query_end, key_end in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            queries = chunk.query_indexes[query_start:query_end]
            keys = chunk.kv_indexes[key_start:key_end]
            # A prefix already inside the ordinary H window must not be added twice.
            assert len(keys) == len(keys.unique())
            visible[queries[:, None], keys[None, :]] = True
            visits[queries] += 1
            query_start, key_start = query_end, key_end
    assert (visits == 1).all()
    assert len(plan.chunks) > 1
    return visible


@pytest.mark.parametrize("history", [1, 2, 32])
def test_fixed_window_prefix_is_visible_to_all_current_visual_and_action_roles(history):
    layout = _layout((history,), (True,))
    visible = dense_action_mask(layout)
    prefix = (layout.stream_ids == 0) & (layout.block_ids == -1)
    queries = layout.gen_query_indexes
    for role in (0, 1, 2):
        current = (layout.stream_ids[queries] == role) & (layout.block_ids[queries] == history)
        assert current.any()
        assert visible[current][:, prefix].all()
    torch.testing.assert_close(_tnd_mask(layout), visible)


@pytest.mark.parametrize("history", [1, 2, 32])
def test_including_prefix_preserves_state_isolation_and_clean_history_state_freedom(history):
    layout = _layout((history,), (True,))
    visible = dense_action_mask(layout)
    queries = layout.gen_query_indexes
    for row in torch.where(layout.stream_ids[queries] == 3)[0]:
        state = int(queries[row])
        expected = torch.where((layout.stream_ids == -1) | (torch.arange(len(layout.stream_ids)) == state))[0]
        assert torch.equal(torch.where(visible[row])[0], expected)
    clean = layout.stream_ids[queries] == 0
    assert not visible[clean][:, layout.stream_ids == 3].any()
    # Only text and clean visual tokens can ever enter the historical representation.
    assert not visible[clean][:, (layout.stream_ids == 1) | (layout.stream_ids == 2)].any()
    torch.testing.assert_close(_tnd_mask(layout), visible)


@pytest.mark.parametrize("history", [1, 2, 32])
def test_legacy_available_visibility_is_unchanged_and_prefix_is_the_only_new_dependency(history):
    legacy = _layout((history,))
    explicit_false = replace(legacy, geometry=CausalActionGeometry((2,), (history,), (False,)))
    fixed = replace(legacy, geometry=CausalActionGeometry((2,), (history,), (True,)))
    old, new = dense_action_mask(legacy), dense_action_mask(fixed)
    torch.testing.assert_close(dense_action_mask(explicit_false), old)
    torch.testing.assert_close(_tnd_mask(legacy), old)
    torch.testing.assert_close(_tnd_mask(explicit_false), old)
    queries = legacy.gen_query_indexes
    current_nonstate = (legacy.block_ids[queries] == history) & (legacy.stream_ids[queries] != 3)
    prefix = (legacy.stream_ids == 0) & (legacy.block_ids == -1)
    assert not old[current_nonstate][:, prefix].any()
    expected_added = current_nonstate[:, None] & prefix[None, :]
    assert torch.equal(new & ~old, expected_added)
    assert not (old & ~new).any()


@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True), (False, False)])
def test_prefix_flags_are_sample_specific_and_tnd_never_crosses_samples(flags):
    histories = (2, 1)
    layout = _layout(histories, flags)
    visible = dense_action_mask(layout)
    queries = layout.gen_query_indexes
    for sample, (history, include) in enumerate(zip(histories, flags, strict=True)):
        current = (
            (layout.sample_ids[queries] == sample)
            & (layout.block_ids[queries] == history)
            & (layout.stream_ids[queries] != 3)
        )
        prefix = (layout.sample_ids == sample) & (layout.stream_ids == 0) & (layout.block_ids == -1)
        assert bool(visible[current][:, prefix].all()) is include
        assert not visible[layout.sample_ids[queries] == sample][:, layout.sample_ids != sample].any()
    torch.testing.assert_close(_tnd_mask(layout), visible)
