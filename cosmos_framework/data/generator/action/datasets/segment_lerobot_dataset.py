# SPDX-License-Identifier: OpenMDW-1.1
"""Template-independent, variable-length LeRobot segment reader."""

import json
import logging
import math
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
    LeRobotDatasetMetadata,
    split_episode_ids,
)
from cosmos_framework.data.generator.action.datasets.direct_parquet import build_file_starts, read_window
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.action.video_view import VideoViewConfig
from lerobot.datasets.video_utils import get_safe_default_codec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionTrainingRange:
    """episode 内的原始 observation 半开范围，不含切片扩展；caption=None 沿用任务文本。"""

    start: int
    stop: int
    caption: str | None = None
    segment_id: str | None = None


class SegmentLeRobotDataset(BaseActionLeRobotDataset):
    """一个实例对应一个 LeRobot 根目录；返回尚未编码、归一化的绝对量。

    数值数据连续读取并保持源 FPS；可在读取时按 geometry 抽视频帧，显式携带 observation 索引。
    相机组合通过 video_view 配置；未配置时只读取数值。
    """

    def __init__(
        self,
        *,
        root: str | Path,
        template: ActionStateTemplate,
        planner: SegmentPlanner,
        window_mode: str = "segmented",
        read_options: ActionReadOptions = ActionReadOptions(),
        use_subtask: bool = False,  # frame_window 只取起点文本，segmented 按 action_config 切分。
        split: str = "full",
        split_seed: int = 0,
        split_val_ratio: float = 0.0,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        table_backend: str = "parquet",
        early_video_sampling: bool = False,
        video_view: VideoViewConfig | None = None,
        viewpoint: str | None = None,
    ):
        if table_backend not in ("parquet", "hf"):
            raise ValueError("table_backend must be parquet or hf")
        if window_mode not in ("segmented", "frame_window"):
            raise ValueError("window_mode must be segmented or frame_window")
        if window_mode == "frame_window":
            if planner.max_action_steps <= 0:
                raise ValueError("frame_window requires positive max_action_steps")
            planner.geometry.validate_observation_frames(planner.max_action_steps + 1)
        self.window_mode = window_mode
        self.root = Path(root)
        self.table_backend = table_backend
        self.early_video_sampling = early_video_sampling
        meta = LeRobotDatasetMetadata(repo_id="local", root=root, revision="local", use_hf_cache=table_backend == "hf")
        self._direct_meta = meta if table_backend == "parquet" else None
        self._file_starts = build_file_starts(meta) if table_backend == "parquet" else None
        self.video_backend = video_backend or get_safe_default_codec()
        self._pyav_resize = video_backend == "pyav_resize"
        self.video_view = video_view
        if video_view is not None:
            video_view.validate_features(meta.features, viewpoint=viewpoint)
        fps = float(meta.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS must be finite and positive")
        super().__init__(
            fps=fps,
            chunk_length=None if planner.max_action_steps == -1 else planner.max_action_steps,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=None,  # Reader 只读取数据，FD/ID/Policy 由后续训练适配层选择。
            embodiment_type=None,
            viewpoint=viewpoint,
            tolerance_s=tolerance_s,
        )
        if type(use_subtask) is not bool:
            raise TypeError("use_subtask must be a bool")
        self.use_subtask = use_subtask
        self.template = template
        self.read_options = read_options
        self._episode_fps, self._subtask_ranges = self._load_episode_metadata(Path(root), meta)
        self.unannotated_observation_frames = 0
        # 索引计数属于当前 Reader，不修改调用方或其他 Reader 的 planner。
        self.planner = SegmentPlanner(
            max_action_steps=planner.max_action_steps,
            overlap_action_steps=0 if window_mode == "frame_window" else planner.overlap_action_steps,
            geometry=planner.geometry,
        )
        self._segments: list[tuple[int, int, int, int]] = []  # ds、episode、源表起点、action 数
        # frame_window 每个可采 episode 仅一条：ds、episode、源表 episode 起点、合法 observation [first, stop)。
        self._frame_ranges: list[tuple[int, int, int, int, int]] = []
        self.skipped_window_episodes = 0
        # 每个保留范围只存一次文本；累计终点同时用于 shuffle 和片段到范围的查找。
        self._range_captions: list[str | None] = []
        # 每个保留 subtask 的 (标注起点, 标注终点, 读取起点, 读取终点)，不按窗口重复存储。
        self._range_bounds: list[tuple[int, int, int, int]] = []
        self._masks: tuple[torch.Tensor, torch.Tensor] | None = None
        self._register_source(
            root=str(root),
            delta_timestamps={},
            tolerance_s=tolerance_s,
            video_backend=video_backend,
            prefetched_meta=meta,
        )

        if window_mode == "segmented":
            # 旧路径建完索引只需范围 caption，释放临时标注边界。
            self._subtask_ranges.clear()
        else:
            # 新路径保留可采 episode 的原始标注供起点查找，不缓存完整 JSONL 或逐帧文本。
            kept = {episode_id for _, episode_id, _, _, _ in self._frame_ranges}
            self._subtask_ranges = {k: v for k, v in self._subtask_ranges.items() if k in kept}
            logger.info(
                "Frame windows source=%s: episodes=%s, candidates=%s, skipped_empty_episodes=%s; "
                "max_action_steps=%s, overlap_action_steps ignored",
                self.root,
                len(self._frame_ranges),
                len(self),
                self.skipped_window_episodes,
                planner.max_action_steps,
            )

    def _load_episode_metadata(self, root, meta):
        """流式读取 FPS 和可选 subtask；episode ID 从 0 连续编号。"""
        total_episodes = meta.total_episodes
        values = np.full(total_episodes, np.nan, dtype=np.float32)
        ranges = {}
        path = root / "meta" / "episodes.jsonl"
        if path.is_file():
            with path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    if not self.use_subtask and "source_fps" not in episode:
                        continue
                    episode_id = episode["episode_index"]
                    if type(episode_id) is not int or not 0 <= episode_id < total_episodes:
                        raise ValueError(f"episode_index out of range in {path}: {episode_id}")
                    if "source_fps" in episode:
                        fps = episode["source_fps"]
                        if (
                            isinstance(fps, bool)
                            or not isinstance(fps, (int, float))
                            or not math.isfinite(fps)
                            or fps <= 0
                        ):
                            raise ValueError(f"Invalid source_fps for episode {episode_id} in {path}: {fps}")
                        values[episode_id] = fps
                    if self.use_subtask:
                        if episode_id in ranges:
                            logger.error(
                                "Duplicate episode %s in %s: overwriting previous subtask annotations", episode_id, path
                            )
                        ep = meta.episodes[episode_id]
                        length = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
                        annotations = episode.get("action_config")
                        try:
                            ranges[episode_id] = self._parse_subtask_annotations(annotations, length)
                        except ValueError as error:
                            fallback = "task text" if self.window_mode == "frame_window" else "episode mode"
                            warnings.warn(
                                f"Episode {episode_id} in {self.root}: falling back to {fallback}: {error}",
                                UserWarning,
                                stacklevel=2,
                            )
                            # 无有效标注；由窗口路径决定范围或文本回退，不冒充原始标注。
                            ranges[episode_id] = []
        values.setflags(write=False)
        return values, ranges

    @staticmethod
    def _parse_subtask_annotations(annotations, episode_length):
        """校验并只保留原始范围及文本，不决定窗口边界；任一标注非法则整组失败。"""
        if not isinstance(annotations, list) or not annotations:
            raise ValueError("action_config must be a non-empty list")
        parsed, previous_stop = [], 0
        for i, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"subtask {i} must be an object")
            start, stop = annotation.get("start_frame"), annotation.get("end_frame")
            caption = annotation.get("action_text")
            if type(start) is not int or type(stop) is not int:
                raise ValueError(f"subtask {i} frame bounds must be integers")
            if not previous_stop <= start < stop <= episode_length:
                raise ValueError(f"subtask {i} bounds must be ordered, non-overlapping and within the episode")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"subtask {i} action_text must be non-empty text")
            parsed.append(ActionTrainingRange(start, stop, caption, str(i)))
            previous_stop = stop
        return parsed

    @staticmethod
    def _find_subtask_annotation(ranges: list[ActionTrainingRange], frame: int) -> ActionTrainingRange | None:
        """在已校验的原始标注 [start, stop) 中查找；间隙返回 None，不使用扩展读取范围。"""
        index = bisect_right(ranges, frame, key=lambda training_range: training_range.start) - 1
        if index >= 0 and frame < ranges[index].stop:
            return ranges[index]
        return None

    def _read_source_fps(self, episode_id):
        """按 episode ID 直接索引；NaN 表示未声明，回退到 meta.fps。"""
        fps = self._episode_fps[episode_id]
        return self.fps if np.isnan(fps) else float(fps)

    def _iter_training_ranges(self, episode_id, episode_length):
        """返回已校验的范围；缺少元数据记录时也回退到 episode。"""
        if not self.use_subtask:
            yield ActionTrainingRange(0, episode_length)
            return
        ranges = self._subtask_ranges.get(episode_id)
        if ranges is None:
            warnings.warn(
                f"Episode {episode_id} in {self.root}: falling back to episode mode: missing metadata",
                UserWarning,
                stacklevel=2,
            )
        ranges = ranges or [ActionTrainingRange(0, episode_length)]
        covered = 0
        for training_range in ranges:
            covered += training_range.stop - training_range.start
            yield training_range
        self.unannotated_observation_frames += episode_length - covered

    def _caption_for_index(self, idx):
        """按范围查找文本，避免为重叠片段重复保存 caption。"""
        if self.window_mode == "frame_window":
            _, episode_id, start, _ = self._resolve_index(idx)
            if idx < 0:
                idx += len(self)
            _, _, begin, _, _ = self._frame_ranges[bisect_right(self._episode_cum_ends, idx)]
            annotation = self._find_subtask_annotation(self._subtask_ranges.get(episode_id, []), start - begin)
            return annotation.caption if annotation is not None else None
        if idx < 0:
            idx += len(self._segments)
        return self._range_captions[bisect_right(self._episode_cum_ends, idx)]

    def _subtask_read_bounds(self, training_range, episode_length):
        """覆盖标注内的合法动作起点，优先向后补真实帧，不足再向前补。"""
        offset = self.read_options.action_time_offset_steps
        episode_first = max(0, -offset)
        episode_stop = max(episode_first, min(episode_length, episode_length + 1 - offset))
        first = max(training_range.start, episode_first)
        # 包含标注末帧动作的终点，避免已整除的 subtask 仍漏掉衔接动作。
        stop = max(first, min(training_range.stop + 1, episode_stop))
        if stop == first or self.planner.max_action_steps == -1:
            return first, stop

        block_actions = self.planner.geometry.actions_per_block
        actions = stop - first - 1
        required_frames = max(1, (actions + block_actions - 1) // block_actions) * block_actions + 1
        missing = required_frames - (stop - first)
        after = min(missing, episode_stop - stop)
        stop += after
        first -= min(missing - after, first - episode_first)
        # 整个 episode 也无法补齐时，由 planner 的重叠尾窗覆盖剩余动作。
        return first, stop

    def _append_index_records(self, *, meta, ds_idx, dataset_label=None):
        """先排除目标偏移导致的越界，再在对齐网格上规划片段。"""
        options = self.read_options
        for key in {options.state_key, options.target_key, options.state_mask_key, options.target_mask_key}:
            if key not in meta.features:
                raise ValueError(f"Missing required field {key!r} in {meta.root}")
        for key in {options.state_key, options.target_key, options.state_mask_key, options.target_mask_key}:
            if tuple(meta.features[key]["shape"]) != (self.template.width,):
                raise ValueError(f"Field {key!r} must declare shape [{self.template.width}]")
        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )
        offset = options.action_time_offset_steps
        for episode_id in episode_ids:
            ep = meta.episodes[episode_id]
            begin, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            count = end - begin
            if self.window_mode == "frame_window":
                # first/stop 是 episode 内合法 observation 区间 [first, stop)，begin 是源表中的 episode 起点。
                # action 的目标行 = action 起点 + offset；负 offset 时跳过开头，避免目标落到 episode 前。
                first = max(0, -offset)
                # 最后一个 action 从 stop-2 开始，其目标 stop-2+offset 必须 < count，
                # 因此 stop <= count+1-offset；observation 本身还要求 stop <= count。
                # max(first, ...) 将无可读数据的情况表示为空区间。
                stop = max(first, min(count, count + 1 - offset))
                # 可用 action 数 = observation 数 - 1；连一步真实 action 都没有时跳过。
                available_actions = max(0, stop - first - 1)
                if not available_actions:
                    self.skipped_window_episodes += 1
                    continue
                # 长 episode 只保留能读满 L 的起点：first .. stop-L-1。
                # 整个合法范围不足 L 时，只保留 first 一条，读取全部真实数据后补齐到最近的 block。
                candidates = max(1, available_actions - self.planner.max_action_steps + 1)
                # 每个 episode 只存一条区间；抽样时再计算具体起点，避免按帧展开索引。
                self._frame_ranges.append((ds_idx, episode_id, begin, first, stop))
                self._num_valid_indices += candidates
                # 保存累计候选数，供 _resolve_index 二分定位 episode。
                # 例如 L=96、两个 episode 各有 100/50 个可用 action，候选数为 5/1，累计终点为 [5, 6]。
                self._episode_cum_ends.append(self._num_valid_indices)
                if self.use_subtask:
                    # 新模式的 subtask 只用于起点文本；缺失标注不会减少上面已计入的候选。
                    ranges = self._subtask_ranges.get(episode_id)
                    if ranges is None:
                        warnings.warn(
                            f"Episode {episode_id} in {self.root}: falling back to task text: missing metadata",
                            UserWarning,
                            stacklevel=2,
                        )
                    # 仅统计未标注的 observation 数，不据此过滤窗口或改变 episode 边界。
                    self.unannotated_observation_frames += count - sum(r.stop - r.start for r in ranges or [])
                # 新模式的 episode 索引已建好，跳过下面旧路径的 subtask/episode 切片规划。
                continue
            for training_range in self._iter_training_ranges(episode_id, count):
                lo, hi = training_range.start, training_range.stop
                if not 0 <= lo <= hi <= count:
                    raise ValueError(f"Episode {episode_id}: training range [{lo}, {hi}) outside [0, {count})")
                is_subtask = training_range.segment_id is not None
                if is_subtask:
                    first, stop = self._subtask_read_bounds(training_range, count)
                else:
                    # episode 模式及统计保持原索引与截尾规则。
                    first = lo + max(0, -offset)
                    stop = max(first, min(hi, hi + 1 - offset))
                planned = self.planner.plan(
                    stop - first,
                    observation_start=first,
                    source_id=str(meta.root),
                    episode_id=episode_id,
                    segment_id=training_range.segment_id,
                    preserve_tail=is_subtask,
                )
                for start, actions in planned:
                    self._segments.append((ds_idx, episode_id, begin + start, actions))
                if planned:
                    # 父类 get_shuffle_blocks 据此分组；默认 episode 模式的分组不变。
                    self._episode_cum_ends.append(len(self._segments))
                    self._range_captions.append(training_range.caption)
                    if is_subtask:
                        self._range_bounds.append((lo, hi, first, stop))
        if self.window_mode == "segmented":
            self._num_valid_indices = len(self._segments)

    def _resolve_index(self, idx):
        """按窗口方式将候选编号映射到 ds、episode、源表起点和真实 action 数。"""
        if self.window_mode == "frame_window":
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"Frame window index {idx} out of range for size {len(self)}")
            span = bisect_right(self._episode_cum_ends, idx)
            ds_idx, episode_id, begin, first, stop = self._frame_ranges[span]
            previous = self._episode_cum_ends[span - 1] if span else 0
            start = first + idx - previous
            return ds_idx, episode_id, begin + start, min(self.planner.max_action_steps, stop - start - 1)
        return self._segments[idx]

    def _read_masks(self, row, *, ds=None, values=None):
        """每个子数据集从一行加载 mask，缓存后不再逐帧查询。"""
        if self._masks is None:
            keys = {self.read_options.state_mask_key, self.read_options.target_mask_key}
            if values is None:
                values = ds._query_hf_dataset({key: [row] for key in keys})
            masks = []
            for key in (self.read_options.state_mask_key, self.read_options.target_mask_key):
                mask = self.template.validate_valid_mask(values[key][0]).clone()
                masks.append(mask)
            self._masks = tuple(masks)
        return self._masks

    def _compute_idle_frames(self, raw_action):
        """绝对目标尚未编码，不能套用旧 action 的静止检测规则。"""
        return None

    def _read_video(self, meta, episode_id, timestamps, *, viewpoint):
        """按给定视频时间戳读取并组合视角。"""
        return (
            self.video_view.read(
                meta,
                episode_id,
                timestamps,
                viewpoint=viewpoint,
                tolerance_s=self._tolerance_s,
                video_backend=self.video_backend,
            )
            if self.video_view is not None
            else None
        )

    def _convert_video(self, video_tchw):
        """The opt-in resized reader already produces uint8; preserve legacy validation otherwise."""
        if not self._pyav_resize:
            return super()._convert_video(video_tchw)
        if self._skip_video_loading or video_tchw is None:
            return None
        if video_tchw.ndim != 4 or video_tchw.shape[1] != 3:
            raise ValueError("pyav_resize expected video with shape [T,3,H,W]")
        if video_tchw.dtype != torch.uint8:
            raise TypeError("pyav_resize expected uint8 video")
        return video_tchw.permute(1, 0, 2, 3)

    def _pad_to_action_steps(self, state, target, times, *, real_actions, output_actions):
        """保留真实行，重复末尾补到指定 action 数；时间轴按存储 FPS 延长。"""
        if not 1 <= real_actions <= output_actions:
            raise ValueError("Padding requires 1 <= real_actions <= output_actions")
        padding = output_actions - real_actions
        if padding:
            state = torch.cat((state, state[-1:].expand(padding, -1)))
            target = torch.cat((target, target[-1:].expand(padding, -1)))
            times = torch.cat((times, times[-1] + torch.arange(1, padding + 1, dtype=times.dtype) / self.fps))
        return state, target, times

    def __getitem__(self, idx):
        """按实际长度批量读取，目标偏移只在这里执行一次。"""
        ds_idx, episode_id, start, real_actions = self._resolve_index(idx)
        # 按当前 episode 获取原始 FPS；只有字段缺失时才回退到 meta.fps。
        source_fps = self._read_source_fps(episode_id)
        options = self.read_options
        if self.table_backend == "parquet":
            meta = self._direct_meta
            state, target, state_mask, action_mask, times, task = self._read_parquet_segment(
                episode_id, start, real_actions
            )
        else:
            ds = self._get_dataset(ds_idx)
            meta = ds.meta
            ds._ensure_hf_dataset_loaded()
            rows = list(range(start, start + real_actions + 1))
            target_rows = [row + options.action_time_offset_steps for row in rows[:-1]]
            # 不经过 LeRobot 的边界 clamp，也不改共享 delta_timestamps。
            observations = ds._query_hf_dataset({options.state_key: rows})
            if options.action_from_state and options.action_time_offset_steps in (0, 1):
                offset = options.action_time_offset_steps
                target = observations[options.state_key][offset : offset + real_actions]
            else:
                target = ds._query_hf_dataset({options.target_key: target_rows})[options.target_key]
            state_mask, action_mask = self._read_masks(rows[0], ds=ds)
            # 绕开 LeRobot 默认 torch.tensor(float) 的 float32 转换，保留长轨迹时间精度。
            timestamps = ds.hf_dataset.select_columns(["timestamp"]).with_format(None)[rows]["timestamp"]
            times = torch.tensor(timestamps, dtype=torch.float64)
            task = ds._query_hf_dataset({"task_index": [rows[0]]})["task_index"][0].item()
            state = observations[options.state_key]
        # 先用存储 FPS 检查连续采样，再将输出 FPS 切换到真实训练时间尺度。
        if not torch.allclose(times[1:] - times[:-1], torch.full_like(times[1:], 1 / self.fps), atol=1e-5, rtol=1e-4):
            raise ValueError(f"Episode {episode_id}: timestamp intervals must match meta.fps={self.fps}")
        actions = real_actions  # 输出长度与真实读取长度分开，正数切片保持原长度。
        if self.window_mode == "frame_window" or self.planner.max_action_steps == -1:
            # Preserve every real row, then repeat the tail to complete a block.
            actions += (-real_actions) % self.planner.geometry.actions_per_block
        state, target, times = self._pad_to_action_steps(
            state, target, times, real_actions=real_actions, output_actions=actions
        )
        # action 时间标记区间起点；目标实际读取时刻由 read_options 声明。
        # 当前使用默认布局；后续布局增强在此选择本次样本的 viewpoint。
        viewpoint = self._viewpoint
        video_indexes = (
            torch.arange(0, actions + 1, self.planner.geometry.video_stride) if self.early_video_sampling else None
        )
        video_times = times if video_indexes is None else times[video_indexes]
        if actions > real_actions:
            # Padded timeline positions reuse the last real video frame.
            video_times = video_times.clamp(max=times[real_actions])
        video = self._read_video(meta, episode_id, video_times.tolist(), viewpoint=viewpoint)
        caption = self._caption_for_index(idx)
        if caption is None:
            caption = str(meta.tasks.iloc[int(task)].name)
        # 沿用父类字典与 uint8 视频格式；action 暂为绝对目标，C06 再写入编码结果。
        sample = self._build_result(
            mode=None,
            video=video.permute(1, 0, 2, 3) if video is not None else None,
            action=target,
            state_trajectory=state,
            action_target=target,
            state_mask=state_mask.clone(),
            action_mask=action_mask.clone(),
            state_timestamps=times,
            action_timestamps=times[:-1].clone(),
            action_state_indexes=torch.arange(actions),
            real_action_steps=real_actions,
            source=str(self.root),
            split=self.split,
            # 存储时间戳仍搭配 storage_fps；训练 FPS 在读取完成后统一切换。
            storage_fps=torch.tensor(self.fps, dtype=torch.float32),
            conditioning_fps=torch.tensor(source_fps, dtype=torch.float32),
            source_fps=torch.tensor(source_fps, dtype=torch.float32),
            read_options=options,
            ai_caption=caption,
            viewpoint=viewpoint,
            additional_view_description=self.video_view.describe(viewpoint=viewpoint) if self.video_view else "",
        )
        if video is not None and video_indexes is not None:
            sample["video_observation_indexes"] = video_indexes
        if self.planner.max_action_steps == -1:
            sample["full_range"] = (start, start + real_actions + 1)
        return sample

    def _read_parquet_segment(self, episode_id, start, actions):
        options = self.read_options
        target_start = start + options.action_time_offset_steps
        begin = min(start, target_start)
        end = max(start + actions + 1, target_start + actions)
        columns = [options.state_key, options.target_key, "timestamp", "task_index"]
        if self._masks is None:
            columns.extend([options.state_mask_key, options.target_mask_key])
        # 合并所有字段为一次读取，兼容 action 来源和 episode 内的正负偏移。
        values = read_window(self._direct_meta, episode_id, begin, end, columns, file_starts=self._file_starts)
        state_offset, target_offset = start - begin, target_start - begin
        state = torch.tensor(values[options.state_key][state_offset : state_offset + actions + 1])
        target = torch.tensor(values[options.target_key][target_offset : target_offset + actions])
        mask_values = None
        if self._masks is None:
            mask_values = {
                key: [torch.tensor(values[key][state_offset])]
                for key in {options.state_mask_key, options.target_mask_key}
            }
        state_mask, action_mask = self._read_masks(start, values=mask_values)
        times = torch.tensor(values["timestamp"][state_offset : state_offset + actions + 1], dtype=torch.float64)
        return state, target, state_mask, action_mask, times, values["task_index"][state_offset]
