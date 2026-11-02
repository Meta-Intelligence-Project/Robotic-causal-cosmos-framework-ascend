# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.generator.action.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    pad_action_to_max_dim,
)
from cosmos_framework.data.generator.action.domain_utils import EMBODIMENT_TO_RAW_ACTION_DIM, get_domain_id
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
)
from cosmos_framework.data.generator.action.video_view import VideoViewConfig
from cosmos_framework.data.generator.utils import ACTION_RES_SIZE_INFO
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.vision import read_media_frames


def _load_actions(
    action_path: Path | str | None,
    model_mode: ModelMode,
    action_chunk_size: int,
    max_action_dim: int,
    raw_action_dim: int | None,
) -> torch.Tensor:
    """Load actions from JSON (or zeros for policy mode and inverse dynamics mode). Returns padded action tensor."""
    match model_mode:
        case ModelMode.FORWARD_DYNAMICS:
            assert action_path is not None, "action_path is required for forward_dynamics mode"
            p = Path(str(action_path))
            raw = torch.tensor(json.loads(p.read_text()), dtype=torch.float32)
            raw_dim = raw.shape[-1]
            assert raw_dim == raw_action_dim, (
                f"Raw action dimension from file ({raw_dim}) does not match expected dimension ({raw_action_dim})"
            )
            return pad_action_to_max_dim(raw, max_action_dim)
        case ModelMode.POLICY | ModelMode.INVERSE_DYNAMICS:
            assert raw_action_dim is not None, "raw_action_dim is required for policy and inverse_dynamics modes"
            return torch.zeros(action_chunk_size, max_action_dim, dtype=torch.float32)
        case _:
            raise ValueError(f"Unsupported action model_mode: {model_mode}")


def _format_prompt(
    prompt: str,
    view_point: str,
    video: torch.Tensor,
    action: torch.Tensor,
    fps: torch.Tensor,
    image_size: torch.Tensor,
    additional_view_description: str = "",
) -> str:
    """Helper function to build the action prompt with optional duration and resolution info."""
    data_dict = {
        "viewpoint": view_point,
        "ai_caption": prompt.strip(),
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
        "additional_view_description": additional_view_description,
    }
    prompt_json_formatter = ActionPromptJsonFormatter()
    ai_caption = prompt_json_formatter(data_dict)[prompt_json_formatter.caption_key]
    if isinstance(ai_caption, dict):
        ai_caption = json.dumps(ai_caption)
    return ai_caption


def build_action_batch(
    *,
    video: torch.Tensor,
    action: torch.Tensor,
    raw_action_dim: int,
    prompt: str,
    view_point: str,
    domain_name: str,
    model_mode: ModelMode,
    action_chunk_size: int,
    fps: int,
    input_video_key: str,
    additional_view_description: str = "",
    batch_size: int = 1,
    device: Any = "cuda",
) -> dict:
    """Build an Action batch from spatially prepared video and action tensors."""
    target_frames = action_chunk_size + 1
    _, num_frames, h, w = video.shape

    if num_frames < target_frames:
        pad = video[:, -1:].repeat(1, target_frames - num_frames, 1, 1)
        video = torch.cat([video, pad], dim=1)
    elif num_frames > target_frames:
        video = video[:, :target_frames]

    image_size = torch.tensor([h, w, h, w], dtype=torch.float32)

    sequence_plan = build_sequence_plan_from_mode(
        mode=model_mode.value,
        video_length=target_frames,
        action_length=action_chunk_size,
        has_text=True,
    )

    ai_caption = _format_prompt(
        prompt=prompt,
        view_point=view_point,
        video=video,
        action=action,
        fps=torch.tensor(fps, dtype=torch.long),
        image_size=image_size,
        additional_view_description=additional_view_description,
    )

    action_processing_record = ActionProcessingRecord(
        raw_action_dim=raw_action_dim,
        action_normalizer=None,
    )

    return {
        input_video_key: [[video]] * batch_size,
        "action": [[action]] * batch_size,
        **make_batched_action_processing_fields(action_processing_record, batch_size),
        "mode": [model_mode.value] * batch_size,
        "ai_caption": [ai_caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(fps, dtype=torch.long)] * batch_size,
        "image_size": image_size.unsqueeze(0).repeat(batch_size, 1).to(device=device),
        "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }


def get_action_sample_data(
    model_config: Any,
    *,
    batch_size: int,
    prompt: str,
    vision_path: Path,
    model_mode: ModelMode,
    action_path: Path | None,
    domain_name: str,
    view_point: str = "ego_view",
    resolution: str,
    action_video_prepared: bool = False,
    action_chunk_size: int,
    max_action_dim: int,
    fps: int,
    device: Any,
) -> dict:
    """Load observation image/video + optional actions and build an Action inference batch."""
    domain_name = domain_name.lower().strip()
    if domain_name not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"invalid domain_name {domain_name!r}; expected one of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )

    raw_action_dim = EMBODIMENT_TO_RAW_ACTION_DIM[domain_name]
    frames, _ = read_media_frames(Path(vision_path), max_frames=action_chunk_size + 1)
    if frames.shape[1] == 0:
        raise ValueError("Action input media must contain at least one frame")
    view = VideoViewConfig(cameras={role: role for role in ("head", "left", "right")}, resolution=resolution)
    cameras = frames.permute(1, 0, 2, 3)
    if view_point == "ego_view":
        cameras = {"head": cameras}
    elif view_point == "concat_view":
        height, width = frames.shape[-2:]
        if height % 3 or width % 2:
            raise ValueError("AgiBot concat input requires three equal-height bands and two equal-width wrist views")
        head_height = height // 3 * 2
        cameras = {
            "head": cameras[:, :, :head_height],
            "left": cameras[:, :, head_height:, : width // 2],
            "right": cameras[:, :, head_height:, width // 2 :],
        }
    else:
        raise ValueError(f"Unsupported action viewpoint: {view_point}")
    if action_video_prepared:
        height, width = cameras["head"].shape[-2:]
        if (width, height) not in ACTION_RES_SIZE_INFO[str(resolution)].values():
            raise ValueError(f"Prepared action input does not match resolution {resolution}: {width}x{height}")
    else:
        frames = view.compose(cameras, frames.shape[1], viewpoint=view_point)
    assert action_path is not None or raw_action_dim is not None, (
        "Either action_path or raw_action_dim must be provided"
    )
    action = _load_actions(action_path, model_mode, action_chunk_size, max_action_dim, raw_action_dim)

    return build_action_batch(
        video=frames,
        action=action,
        raw_action_dim=raw_action_dim,
        prompt=prompt,
        view_point=view_point,
        domain_name=domain_name,
        model_mode=model_mode,
        action_chunk_size=action_chunk_size,
        fps=fps,
        additional_view_description=view.describe(viewpoint=view_point),
        input_video_key=model_config.input_video_key,
        batch_size=batch_size,
        device=device,
    )
