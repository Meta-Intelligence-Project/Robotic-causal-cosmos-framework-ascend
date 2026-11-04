# SPDX-License-Identifier: OpenMDW-1.1
"""Template-driven causal training adapter for segmented readers."""

import math
import random
from dataclasses import replace

import torch

from cosmos_framework.data.generator.action.sample_contract import ActionSampleReadError

from .action_sft_dataset import ActionSFTDataset

MODES = ("forward_dynamics", "inverse_dynamics", "policy")


def validate_joint_weights(weights=None):
    weights = dict.fromkeys(MODES, 1.0) if weights is None else dict(weights)
    if set(weights) != set(MODES):
        raise ValueError(f"joint_mode_weights requires exactly {MODES}")
    values = tuple(float(weights[m]) for m in MODES)
    if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("joint_mode_weights must be finite, nonnegative, and have positive sum")
    return values


class CausalActionSFTDataset(ActionSFTDataset):
    """Any source implementing the measured-state contract can share this path."""

    def __init__(
        self,
        base,
        *,
        mode,
        statistics_path=None,
        statistics=None,
        allow_mock_statistics=False,
        history_blocks_min=1,
        history_blocks_max=8,
        history_padding="available",
        history_steps=None,
        action_horizon=None,
        joint_mode_weights=None,
        use_state=True,
    ):
        super().__init__(base._dataset, base._transform, base._resolution)
        if mode not in (*MODES, "joint"):
            raise ValueError(f"Unsupported causal action mode: {mode}")
        if not use_state or (statistics_path is None and statistics is None):
            raise ValueError("Measured block states and separate state/delta statistics are required")
        if not 1 <= history_blocks_min <= history_blocks_max:
            raise ValueError("Invalid history sampling range")
        if statistics_path is not None and statistics is not None:
            raise ValueError("Provide statistics or statistics_path, not both")
        self.template = self._dataset.template
        self.planner = self._dataset.planner
        self.mode = mode
        self.history_range = (history_blocks_min, history_blocks_max)
        if history_padding not in ("available", "repeat_first"):
            raise ValueError("history_padding must be available or repeat_first")
        if history_padding == "repeat_first" and not callable(getattr(self._dataset, "read_prediction_window", None)):
            raise ValueError("repeat_first training requires a reader with true episode-relative prediction windows")
        self.history_padding = history_padding
        if (history_steps is None) != (action_horizon is None):
            raise ValueError("history_steps and action_horizon must be supplied together")
        if history_steps is not None and history_padding != "repeat_first":
            raise ValueError("Decision windows require repeat_first history padding")
        self.history_steps = (
            history_steps if history_steps is not None else history_blocks_max * self.planner.geometry.actions_per_block
        )
        self.action_horizon = action_horizon if action_horizon is not None else self.planner.geometry.actions_per_block
        per_latent = self.planner.geometry.actions_per_latent_frame
        if history_padding == "repeat_first" and (
            type(self.history_steps) is not int
            or self.history_steps < 0
            or type(self.action_horizon) is not int
            or self.action_horizon < 1
            or self.history_steps % per_latent
            or self.action_horizon % per_latent
        ):
            raise ValueError(
                "history_steps and action_horizon must be nonnegative/positive multiples of actions per latent"
            )
        self.statistics_path = statistics_path
        self.statistics = statistics
        self._statistics_validated = False
        self.allow_mock_statistics = allow_mock_statistics
        self.weights = validate_joint_weights(joint_mode_weights)
        self.debug_fixed_index = None

    def __getitem__(self, idx):
        from cosmos_framework.data.generator.action.block_state import BlockStatistics, build_block_sample

        idx = idx if self.debug_fixed_index is None else self.debug_fixed_index
        rng = random.getstate()
        try:
            if self.debug_fixed_index is not None:
                random.seed(42)
            raw = (
                self._dataset.read_prediction_window(
                    idx, history_steps=self.history_steps, action_horizon=self.action_horizon
                )
                if self.history_padding == "repeat_first"
                else self._dataset[idx]
            )
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except Exception as error:
            # 只标记原始读取错误；统计校验、编码和 transform 错误仍直接抛出。
            raise ActionSampleReadError(
                f"Failed to read source={str(self._dataset.root)!r}, sample_index={idx}: "
                f"{type(error).__name__}: {error}"
            ) from error
        finally:
            if self.debug_fixed_index is not None:
                random.setstate(rng)
        if self.statistics is None:
            self.statistics = BlockStatistics.load(self.statistics_path)
        if not self._statistics_validated:
            self.statistics.validate(
                self.template.width, raw["state_mask"], raw["action_mask"], allow_mock=self.allow_mock_statistics
            )
            self._statistics_validated = True
        mode = random.choices(MODES, weights=self.weights, k=1)[0] if self.mode == "joint" else self.mode
        if self.history_padding == "repeat_first":
            from cosmos_framework.data.generator.action.fixed_history_sample import build_fixed_history_sample

            raw, metadata = build_fixed_history_sample(
                raw, template=self.template, planner=self.planner, statistics=self.statistics
            )
        else:
            raw, metadata = build_block_sample(
                raw,
                template=self.template,
                planner=self.planner,
                history_blocks=random.randint(*self.history_range),
                statistics=self.statistics,
            )
        for key in (
            "state_trajectory",
            "action_target",
            "state_mask",
            "action_mask",
            "action_state_indexes",
            "state_timestamps",
            "action_timestamps",
            "storage_fps",
            "source",
            "split",
            "read_options",
            "history_window",
            "history_episode_id",
        ):
            raw.pop(key, None)
        raw["mode"] = mode
        result = self._transform(raw, self._resolution, action_normalizer=None)
        length = result["action"].shape[0]
        if result["action"].shape[-1] != self.template.width:
            raise ValueError("The shared action transform must use template.width without extra padding")
        action_conditions = length if mode == "forward_dynamics" else 0
        plan = result["sequence_plan"]
        vision_conditions = plan.condition_frame_indexes_vision
        if self.history_padding == "repeat_first":
            vision_conditions = sorted(set(vision_conditions) | set(range(metadata.observed_latents)))
        result["sequence_plan"] = replace(
            plan,
            condition_frame_indexes_action=list(range(action_conditions)),
            condition_frame_indexes_vision=vision_conditions,
            # offset以action源时间步计；FPS位置与当前视频latent对齐。
            action_start_frame_offset=self.history_steps + 1 if metadata.layout_mode == "decision_window" else 1,
            causal_action_metadata=metadata,
        )
        result["raw_action_dim"] = None
        result["causal_action_mode"] = mode
        return result
