# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned physical layouts, independent of dataset I/O and normalization.

55D EEF quaternions use xyzw. Relative scalars are target minus block anchor;
Gripper and dexterous-hand targets remain absolute in both encoding and decoding.
rotation deltas are inverse(anchor) * target. Decoding composes anchor * delta.
Quaternion results are unit length with a deterministic sign (q and -q are
the same rotation). No angle wrapping or coordinate-frame conversion is implicit.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class ValidMaskSpec:
    width: int
    dtype: str = "bool"
    true_means: str = "valid"
    alignment: str = "template_dimension"
    granularity: str = "dataset"
    version: str = "v1"


class ActionStateTemplate(ABC):
    template_id: str
    width: int
    fields: Mapping[str, slice]
    rotation_groups: tuple[tuple[int, ...], ...]
    valid_mask_spec: ValidMaskSpec

    @property
    def action_names(self):
        """字段名来自模板，通用 reader 不再生成特定机器人的槽位名。"""
        names = [f"channel_{i}" for i in range(self.width)]
        for name, slots in self.fields.items():
            for i, slot in enumerate(range(*slots.indices(self.width))):
                names[slot] = f"{name}_{i}"
        return names

    @abstractmethod
    def sanitize(self, values, valid_mask): ...

    def mock_quantiles(self, state_stats, delta_stats, valid_mask):
        """返回模板表示的模拟范围；具体字段映射只能由模板定义。"""
        raise ValueError("This template does not define an AgiBot mock statistics mapping")

    def required_anchor_mask(self, action_mask):
        """默认每个有效目标都需要 anchor；含绝对字段的模板可覆盖此规则。"""
        return self.validate_valid_mask(action_mask)

    @abstractmethod
    def validate_valid_mask(self, valid_mask, *, template_id=None) -> torch.Tensor:
        """校验数据集共享 mask，返回布尔向量。"""
        ...

    @abstractmethod
    def validate(self, state, action, valid_mask) -> None:
        """检查 state/action 数值和形状是否符合模板。"""
        ...

    @abstractmethod
    def encode_action_delta(self, absolute_action, anchor_state, valid_mask) -> torch.Tensor:
        """按模板逐字段策略编码目标；相对字段使用块起点 state。"""
        ...

    @abstractmethod
    def decode_action_delta(self, action_delta, anchor_state, valid_mask) -> torch.Tensor:
        """使用编码时的同一 anchor，将相对 action 还原为绝对量。"""
        ...


class ActionStateTemplate55(ActionStateTemplate):
    """55D layout for processed state/action data.

    No dataset masks or quantiles are stored on the template. A dataset owns
    one [55] mask, broadcast at runtime over any leading tensor dimensions.
    Invalid values are cleared by sanitize/encode/decode, including NaNs.
    """

    template_id = "unified55-xyzw-absolute-gripper-hand-v4"
    width = 55
    fields = MappingProxyType(
        {
            "left_arm_joint": slice(0, 7),
            "right_arm_joint": slice(7, 14),
            "left_eef": slice(14, 21),
            "right_eef": slice(21, 28),
            "left_gripper": slice(28, 29),
            "right_gripper": slice(29, 30),
            "left_hand": slice(30, 36),
            "right_hand": slice(36, 42),
            "waist": slice(42, 46),
            "head": slice(46, 48),
            "mobile": slice(48, 51),
            "reserved": slice(51, 55),
        }
    )
    # 左 EEF 四元数占 17:21，右 EEF 四元数占 24:28，均按 xyzw 排列；每组四维必须整体有效或无效。
    rotation_groups = (tuple(range(17, 21)), tuple(range(24, 28)))
    valid_mask_spec = ValidMaskSpec(width=55)
    absolute_fields = ("left_gripper", "right_gripper", "left_hand", "right_hand")

    def required_anchor_mask(self, action_mask):
        """夹爪与灵巧手是绝对目标，不要求对应的 state 维度有效。"""
        mask = self.validate_valid_mask(action_mask).clone()
        for name in self.absolute_fields:
            mask[self.fields[name]] = False
        return mask

    def validate_valid_mask(self, valid_mask, *, template_id=None):
        """校验 55D mask、模板版本、保留位及四元数组完整性。"""
        if template_id is not None and template_id != self.template_id:
            raise ValueError(f"Expected template_id={self.template_id}, got {template_id}")
        # Preserve Python float precision before checking exact 0/1 values.
        mask = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.as_tensor(valid_mask, dtype=torch.float64)
        if mask.shape != (self.width,):
            raise ValueError("valid_mask must be a dataset-level [55] vector")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("valid_mask must contain only bool or exact 0/1 values")
        mask = mask.bool()
        if mask[self.fields["reserved"]].any():
            raise ValueError("Reserved dimensions 51:55 must be invalid")
        for group in self.rotation_groups:
            enabled = mask[list(group)]
            if enabled.any() and not enabled.all():
                raise ValueError("A quaternion rotation group must be entirely valid or invalid")
        return mask

    def sanitize(self, values, valid_mask):
        """清零无效维，拒绝非法数值，并统一 state/target 四元数的长度和符号。"""
        mask = self.validate_valid_mask(valid_mask)
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise ValueError("Template values must be floating-point tensors")
        if values.ndim < 1 or values.shape[-1] != self.width:
            raise ValueError("Template values must have last dimension 55")
        mask = mask.to(values.device)
        result = values.masked_fill(~mask, 0)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite value in a valid template dimension")
        for group in self.rotation_groups:
            if mask[list(group)].all():
                q = result[..., list(group)]
                if (torch.linalg.vector_norm(q, dim=-1) < 1e-8).any():
                    raise ValueError("Active quaternion must be nonzero")
                # state 也进入模型；与 action 编码共用单位四元数及确定性符号约定。
                result[..., list(group)] = self._unit_quaternion(q)
        return result

    def validate(self, state, action, valid_mask):
        """校验有效数值，并确认 state/action 可通过广播对齐。"""
        state, action = self.sanitize(state, valid_mask), self.sanitize(action, valid_mask)
        try:
            torch.broadcast_shapes(state.shape, action.shape)
        except RuntimeError as exc:
            raise ValueError("State and action shapes are not broadcastable") from exc

    @staticmethod
    def _unit_quaternion(q):
        """归一化 xyzw 四元数并统一符号，消除 q 与 -q 的表示歧义。"""
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        # w >= 0; at exactly 180 degrees use the largest xyz component.
        pivot = torch.where(q[..., 3:4] != 0, q[..., 3:4], q.gather(-1, q.abs().argmax(-1, keepdim=True)))
        return torch.where(pivot < 0, -q, q)

    @staticmethod
    def _multiply(a, b):
        """计算 xyzw 四元数乘积 a * b，保持旋转复合顺序。"""
        av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
        return torch.cat(
            (aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1
        )

    def _convert(self, values, anchor, valid_mask, *, decode):
        """相对标量做加减，夹爪与灵巧手绝对保留，四元数复合，清零无效维。"""
        self.validate(anchor, values, valid_mask)
        values = self.sanitize(values, valid_mask)
        anchor = self.sanitize(anchor, valid_mask)
        values, anchor = torch.broadcast_tensors(values, anchor)
        result = values + anchor if decode else values - anchor
        # 目标时间由读取选项决定；夹爪和灵巧手编解码均不加减 block anchor。
        for name in self.absolute_fields:
            result[..., self.fields[name]] = values[..., self.fields[name]]
        mask = self.validate_valid_mask(valid_mask).to(values.device)
        for group in self.rotation_groups:
            indexes = list(group)
            if mask[indexes].all():
                a, b = self._unit_quaternion(anchor[..., indexes]), self._unit_quaternion(values[..., indexes])
                if not decode:
                    a = torch.cat((-a[..., :3], a[..., 3:]), -1)
                result[..., indexes] = self._unit_quaternion(self._multiply(a, b))
        return result.masked_fill(~mask, 0)

    def encode_action_delta(self, absolute_action, anchor_state, valid_mask):
        """编码块相对目标；旋转采用 inverse(anchor) * target。"""
        return self._convert(absolute_action, anchor_state, valid_mask, decode=False)

    def decode_action_delta(self, action_delta, anchor_state, valid_mask):
        """还原绝对目标；旋转采用 anchor * delta，输入应已反归一化。"""
        return self._convert(action_delta, anchor_state, valid_mask, decode=True)

    def mock_quantiles(self, state_stats, delta_stats, valid_mask):
        """AgiBot 具名统计映射到本模板；未确认的旋转顺序不猜测。"""
        mask = self.validate_valid_mask(valid_mask)
        sl, sh = torch.full((self.width,), -1.0), torch.ones(self.width)
        al, ah = sl.clone(), sh.clone()
        origin = {name: "placeholder" for name in self.fields}
        # 所有数字切片均属于此模板，统计加载器只接收最终 D 维范围。
        mappings = {
            "left_arm_joint": ("observation.states.joint.position", slice(0, 7)),
            "right_arm_joint": ("observation.states.joint.position", slice(7, 14)),
            "left_gripper": ("observation.states.effector.position", slice(0, 1)),
            "right_gripper": ("observation.states.effector.position", slice(1, 2)),
            "head": ("observation.states.head.position", slice(0, 2)),
        }

        def bounds(record, slots, width):
            if not record or "q01" not in record or "q99" not in record:
                return None
            lo = torch.as_tensor(record["q01"], dtype=torch.float32).flatten()[slots]
            hi = torch.as_tensor(record["q99"], dtype=torch.float32).flatten()[slots]
            if lo.shape != (width,) or hi.shape != (width,):
                return None
            if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (hi < lo).any():
                raise ValueError("Invalid AgiBot q01/q99 bounds")
            return lo, hi

        for name, (key, source_slots) in mappings.items():
            slots = self.fields[name]
            pair = bounds(state_stats.get(key), source_slots, slots.stop - slots.start)
            if pair is None:
                continue
            sl[slots], sh[slots] = pair
            if name in self.absolute_fields:
                al[slots], ah[slots] = pair
            else:
                span = pair[1] - pair[0]
                al[slots], ah[slots] = -span, span
            origin[name] = key
        for side, source_slots in (("left", slice(0, 3)), ("right", slice(3, 6))):
            name = f"{side}_eef"
            slots = self.fields[name]
            position = slice(slots.start, slots.start + 3)
            pair = bounds(state_stats.get("observation.states.end.position"), source_slots, 3)
            if pair is not None:
                sl[position], sh[position] = pair
                span = pair[1] - pair[0]
                al[position], ah[position] = -span, span
                origin[name] = "position:state_span; quaternion:unit_component_range"
            key = f"action.{side}_arm_eef_position.delta_sub"
            pair = bounds(delta_stats.get(key), slice(None), 3)
            if pair is not None:
                al[position], ah[position] = pair
                origin[name] = f"position:{key}; quaternion:unit_component_range"
        for values in (sl, sh, al, ah):
            values.masked_fill_(~mask, 0)
        return (sl, sh), (al, ah), origin


def resolve_action_template(template=None):
    """支持实例、类或完整类路径；更换模板无需改通用 dataset。"""
    if template is None:
        return ActionStateTemplate55()
    if isinstance(template, str):
        module, name = template.rsplit(".", 1)
        template = getattr(importlib.import_module(module), name)
    if isinstance(template, type):
        template = template()
    if not isinstance(template, ActionStateTemplate):
        raise TypeError("Expected an ActionStateTemplate instance or class path")
    return template
