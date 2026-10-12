# SPDX-License-Identifier: OpenMDW-1.1
"""Thin adapter for already-template-mapped RoboTwin exports."""

from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig

_DEFAULT_READ_OPTIONS = ActionReadOptions(
    state_key="state_unified",
    action_key="action_unified",
    state_mask_key="mask_state",
    action_mask_key="mask_action",
    action_from_state=False,
    action_time_offset_steps=0,
)
_CAMERA_MAPPING = {
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}


class RoboTwinSegmentLeRobotDataset(SegmentLeRobotDataset):
    """处理后 RoboTwin：直接读取原始绝对命令目标，不重复移位。

    state 是命令目标代理，未确认是真实测量 qpos；缺失维度由来源 mask 屏蔽。
    template/planner 由调用方传入，编码和归一化由公共训练链路完成。
    """

    default_read_options = _DEFAULT_READ_OPTIONS

    def __init__(
        self,
        *,
        viewpoint="concat_view",
        video_view: VideoViewConfig | None = None,
        read_options=_DEFAULT_READ_OPTIONS,
        **kwargs,
    ):
        if video_view is None:
            video_view = VideoViewConfig(cameras=_CAMERA_MAPPING.copy())
        super().__init__(read_options=read_options, video_view=video_view, viewpoint=viewpoint, **kwargs)
