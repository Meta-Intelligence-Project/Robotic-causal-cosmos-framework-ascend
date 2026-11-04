# SPDX-License-Identifier: OpenMDW-1.1
"""Episode-relative padding and current-anchored history geometry."""

import pytest

from cosmos_framework.data.generator.action.history_window import build_history_window


@pytest.mark.parametrize("tick,blocks,start", [(0, 0, 0), (16, 0, 16), (32, 1, 0), (48, 1, 16), (96, 2, 32)])
def test_available_preserves_complete_block_warmup(tick, blocks, start):
    window = build_history_window(tick, history_blocks=2)
    assert window.history_blocks == blocks
    assert window.window_start_tick == start
    assert window.logical_ticks == window.observation_ticks == tuple(range(start, tick + 1, 4))


@pytest.mark.parametrize("tick", [0, 1, 8, 16, 31, 32, 48, 64, 80])
def test_fixed_window_pads_only_negative_logical_times(tick):
    window = build_history_window(tick, history_blocks=2, history_padding="repeat_first")
    assert window.history_blocks == 2
    assert window.window_start_tick == tick - 64
    assert len(window.logical_ticks) == 17
    assert window.logical_ticks[-1] == tick
    assert window.observation_ticks == tuple(max(0, t) for t in window.logical_ticks)
    assert len(window.required_observation_ticks) == len(set(window.observation_ticks))
    assert all(
        actual == logical for logical, actual in zip(window.logical_ticks, window.observation_ticks) if logical >= 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tick": -1},
        {"tick": True},
        {"history_blocks": 0},
        {"history_blocks": 1.5},
        {"video_stride": 3},
        {"history_padding": "earliest_available"},
    ],
)
def test_invalid_window_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_history_window(**dict(tick=0, history_blocks=2) | kwargs)


@pytest.mark.parametrize("history_steps", [0, 48, 64, 1024])
@pytest.mark.parametrize("tick", [0, 1, 17, 80])
def test_explicit_history_steps_are_independent_of_prediction_block(history_steps, tick):
    window = build_history_window(tick, history_steps=history_steps, history_padding="repeat_first")
    assert window.history_steps == history_steps
    assert window.window_start_tick == tick - history_steps
    assert len(window.logical_ticks) == history_steps // 4 + 1
    assert window.observation_ticks == tuple(max(0, t) for t in window.logical_ticks)
