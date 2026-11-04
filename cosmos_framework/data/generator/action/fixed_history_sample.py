# SPDX-License-Identifier: OpenMDW-1.1
"""One decision sample: visual history, one measured state and future actions."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from cosmos_framework.data.generator.action.block_state import build_block_sample
from cosmos_framework.data.generator.action.history_window import HistoryWindow


def build_fixed_history_sample(raw, *, template, planner, statistics):
    """Keep history visual-only; encode real targets against the fresh current state."""
    window = raw.get("history_window")
    if not isinstance(window, HistoryWindow):
        raise ValueError("Fixed history training requires an explicit episode-relative history window")
    geometry = planner.geometry
    horizon = raw.get("action_horizon", geometry.actions_per_block)
    geometry = replace(geometry, actions_per_block=horizon)
    history_steps = window.history_steps
    per_latent = geometry.actions_per_latent_frame
    if history_steps % per_latent:
        raise ValueError("history_steps must align with video_stride * VAE temporal compression")
    observed = 1 + history_steps // per_latent
    video = raw.get("video")
    if len(window.logical_ticks) != 1 + history_steps // geometry.video_stride:
        raise ValueError("History window sampling differs from the prediction geometry")
    if len(raw["action_target"]) != horizon or len(raw["state_trajectory"]) != horizon + 1:
        raise ValueError("Decision samples require exactly action_horizon real future actions")
    if (
        video is None
        or video.ndim != 4
        or video.shape[1] != len(window.logical_ticks) + horizon // geometry.video_stride
    ):
        raise ValueError("Decision video must contain the sampled history and real future horizon")
    # The existing encoder validates physical anchors, target offsets and masks.
    current_raw = dict(raw, video=None)
    current_raw.pop("video_observation_indexes", None)
    result, current = build_block_sample(
        current_raw,
        template=template,
        planner=SimpleNamespace(geometry=geometry),
        history_blocks=1,
        statistics=statistics,
    )
    # 复用当前anchor的delta/quantile编码，只平移视频时间位置；不制造历史动作槽。
    metadata = replace(
        current,
        action_frame_ids=observed + torch.arange(horizon) // per_latent,
        state_latent_indexes=torch.tensor([observed - 1]),
        include_visual_prefix=True,
        layout_mode="decision_window",
        history_steps=history_steps,
        action_horizon=horizon,
        observed_latents=observed,
        future_latents=geometry.latent_frames_per_block,
        actions_per_latent_frame=per_latent,
    )
    result.update(video=video)
    return result, metadata
