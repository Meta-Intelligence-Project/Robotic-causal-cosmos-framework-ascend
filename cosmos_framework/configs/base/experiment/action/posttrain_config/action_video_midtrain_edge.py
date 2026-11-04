# SPDX-License-Identifier: OpenMDW-1.1
"""One shared causal model alternating robot-action and human-video training steps."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_causal_midtrain_edge import (
    cfg as action_cfg,
)
from cosmos_framework.data.generator.causal_multitask_dataloader import CausalMultitaskDataLoader
from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import get_sft_dataset_from_lerobot
from cosmos_framework.utils.lazy_config import LazyCall as L

cfg = copy.deepcopy(action_cfg)
cfg.job.group = "causal_multitask"
cfg.job.name = "action_video_midtrain_edge"
robot_dataset = cfg.dataloader_train.dataloader.datasets.robots.dataset
# robot使用独立P/H决策窗口；human视频仍走自己的AR几何，不构造state/action。
cfg.dataloader_train = L(CausalMultitaskDataLoader)(
    audio_sample_rate=48000,
    max_samples_per_batch=None,
    max_sequence_length=54000,
    patch_spatial=2,
    sound_latent_fps=0,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    seed=42,
    lookahead_limit=1,
    dataloader=dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        datasets=dict(
            robots=dict(ratio="${data_setting.human_video.robot_ratio}", dataset=robot_dataset),
            video=dict(
                ratio="${data_setting.human_video.video_ratio}",
                dataset=L(get_sft_dataset_from_lerobot)(
                    dataset_path="${data_setting.human_video.dataset_path}",
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    conditioning_config={0: 1.0},
                    conditioning_fps=-1,
                    cfg_dropout_rate=0.1,
                    max_caption_tokens=2048,
                    caption_key="task",
                    video_feature_key=None,
                    video_feature_keywords=["head", "top", "main"],
                    resolution_tiers=["256", "480"],
                    use_multi_fps=False,
                    min_video_frames=33,
                    max_video_duration_s=61.0,
                    long_video_policy="split",
                    video_window_overlap_s=5.0,
                    video_backend="pyav",
                    max_video_fps=30.0,
                    video_tolerance_s=1e-4,
                ),
            ),
        ),
    ),
)

ConfigStore.instance().store(group="experiment", package="_global_", name="action_video_midtrain_edge", node=cfg)
