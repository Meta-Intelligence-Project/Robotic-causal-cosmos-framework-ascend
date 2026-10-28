# SPDX-License-Identifier: OpenMDW-1.1
"""Build raw online policy batches exclusively from measured RTC observations."""

import copy
import math
from dataclasses import asdict, replace
from numbers import Integral, Real

import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.device.type != "cpu":
            raise ValueError(f"{name} must be one CPU scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _single(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one sample")
        return value[0]
    return value


class RTCObservedBatchBuilder:
    """Reuse a reference sample's schema, never its pixels, states or targets.

    ``reference_batch`` is one unprocessed CPU uint8 dataset sample, either a
    single-sample collated batch or the corresponding uncollated sample. Its
    template, statistics, fixed task caption/tokenization and spatial metadata
    define the model interface. Images must already have the same concatenated
    camera layout and resized/padded ``[C,H,W]`` shape as that reference.

    A planning request at tick t ends its observed window exactly at t. It uses
    h=min(t//32, history_blocks) complete preceding blocks, sampling real images
    at t-32h, t-32h+4, ..., t, then appends eight zero future placeholders. This
    also handles requests between old block boundaries, e.g. t=48 starts at16.
    Missing measurements are errors; no frame is interpolated or copied from
    the reference. The session attaches ``rtc`` separately after construction.
    """

    def __init__(
        self,
        reference_batch,
        *,
        controller_hz,
        actions_per_block=32,
        video_stride=4,
        temporal_compression_factor=4,
        history_blocks=8,
        video_key="video",
        caption_key="ai_caption",
    ):
        if not isinstance(reference_batch, dict):
            raise ValueError("reference_batch must be a raw single-sample dictionary")
        self.controller_hz = _scalar(controller_hz, "controller_hz")
        self.history_blocks = _integer(history_blocks, "history_blocks", 1)
        geometry = CausalBlockGeometry(
            actions_per_block=_integer(actions_per_block, "actions_per_block", 1),
            video_stride=_integer(video_stride, "video_stride", 1),
            temporal_compression_factor=_integer(temporal_compression_factor, "temporal_compression_factor", 1),
        )
        if (geometry.actions_per_block, geometry.video_stride, geometry.temporal_compression_factor) != (32, 4, 4):
            raise ValueError("RTCObservedBatchBuilder currently supports the 32-action/stride4/VAE4 recipe")
        self.geometry, self.video_key, self.caption_key = geometry, video_key, caption_key
        video_payload = reference_batch.get(video_key)
        uncollated = isinstance(video_payload, torch.Tensor) and video_payload.ndim == 4
        video = _single(video_payload, video_key)
        if isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 4
            or video.dtype != torch.uint8
            or video.device.type != "cpu"
            or video.shape[0] != 3
            or min(video.shape[1:]) < 1
            or reference_batch.get("is_preprocessed", False)
        ):
            raise ValueError("reference video must be raw CPU uint8 [3,T,H,W], not preprocessed latents")
        self.image_shape = (int(video.shape[0]), int(video.shape[2]), int(video.shape[3]))
        plan = _single(reference_batch.get("sequence_plan"), "sequence_plan")
        metadata = getattr(plan, "causal_action_metadata", None)
        if (
            metadata is None
            or metadata.template is None
            or metadata.statistics is None
            or not plan.has_vision
            or not plan.has_action
            or not plan.has_text
            or plan.has_sound
        ):
            raise ValueError("reference needs a text/video/action SequencePlan with template and statistics")
        self.template = copy.deepcopy(metadata.template)
        self._statistics = copy.deepcopy(metadata.statistics)
        provenance = self._statistics.provenance
        if provenance.get("geometry") != asdict(geometry):
            raise ValueError("reference statistics geometry differs from the RTC recipe")
        if provenance.get("template_id", self.template.template_id) != self.template.template_id:
            raise ValueError("reference statistics template differs from its SequencePlan")
        if metadata.block_size != geometry.latent_frames_per_block:
            raise ValueError("reference latent block size differs from the RTC recipe")
        ids = metadata.action_frame_ids
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long or ids.device.type != "cpu":
            raise ValueError("reference action_frame_ids must be a CPU int64 vector")
        if not len(ids) or len(ids) % geometry.actions_per_block:
            raise ValueError("reference must contain complete 32-action blocks")
        count = len(ids) // geometry.actions_per_block
        expected = torch.arange(1, 1 + count * geometry.latent_frames_per_block).repeat_interleave(
            geometry.actions_per_latent_frame
        )
        if not torch.equal(ids, expected) or video.shape[1] != 1 + len(ids) // geometry.video_stride:
            raise ValueError("reference action/video geometry does not match its declared statistics")
        for name, expected_fps in (
            ("conditioning_fps", self.controller_hz / geometry.video_stride),
            ("conditioning_fps_action", self.controller_hz),
        ):
            if name not in reference_batch or not math.isclose(
                _scalar(reference_batch[name], name), expected_fps, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("reference FPS differs from controller_hz and video_stride")
        for quantiles in (self._statistics.state, self._statistics.action):
            quantiles.__post_init__()
            if quantiles.low.shape != (self.template.width,) or any(
                value.device.type != "cpu" for value in (quantiles.low, quantiles.high, quantiles.valid)
            ):
                raise ValueError("reference statistics must have CPU template-width bounds")
        self._plan = replace(
            plan,
            condition_frame_indexes_vision=[],
            condition_frame_indexes_action=[],
            condition_frame_indexes_sound=[],
            causal_action_metadata=None,
            action_start_frame_offset=1,
        )
        static_keys = (
            caption_key,
            f"neg_{caption_key}",
            "text_token_ids",
            "neg_text_token_ids",
            "image_size",
            "domain_id",
            "action_processing_record",
        )
        self._static = {}
        for key in static_keys:
            if key in reference_batch:
                value = copy.deepcopy(reference_batch[key])
                self._static[key] = [value] if uncollated else value
        if caption_key not in self._static:
            raise ValueError("reference must supply the fixed task caption")
        record = _single(self._static.get("action_processing_record"), "action_processing_record")
        if record is not None and (
            record.raw_action_dim != self.template.width or record.action_normalizer is not None
        ):
            raise ValueError("reference must use template statistics without a second action normalizer or padding")
        cache_mode = reference_batch.get("causal_action_cache_mode", "full")
        if cache_mode not in ("off", "plans", "full"):
            raise ValueError("reference causal_action_cache_mode must be off, plans or full")
        self._cache_mode = cache_mode

    def _image(self, image, label):
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or image.dtype != torch.uint8
            or tuple(image.shape) != self.image_shape
        ):
            raise ValueError(f"{label} must be a real CPU uint8 image with shape {self.image_shape}")
        return image.detach()

    @torch.no_grad()
    def __call__(self, *, state, state_mask, image, action_mask, history=(), tick, revision, rtc=None):
        tick, revision = _integer(tick, "tick"), _integer(revision, "revision")
        if not isinstance(state, torch.Tensor) or state.device.type != "cpu" or state.shape != (self.template.width,):
            raise ValueError("state must be a fresh CPU [template.width] physical measurement")
        sm = self.template.validate_valid_mask(state_mask).detach().cpu().clone()
        am = self.template.validate_valid_mask(action_mask).detach().cpu().clone()
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("current actions require valid measured anchor channels")
        state = self.template.sanitize(state.detach().float(), sm)
        for mask, quantiles in ((sm, self._statistics.state), (am, self._statistics.action)):
            if (mask & ~quantiles.valid).any():
                raise ValueError("reference statistics do not cover the measured state/action masks")
        current_image = self._image(image, "image")
        observed = {}
        previous_tick = -1
        for record in history:
            record_tick = _integer(record.tick, "history observation tick")
            if record_tick <= previous_tick or record_tick >= tick:
                raise ValueError("history observations must have unique increasing ticks before the fresh image")
            observed[record_tick] = record.image
            previous_tick = record_tick
        observed[tick] = current_image
        geometry = self.geometry
        h = min(tick // geometry.actions_per_block, self.history_blocks)
        first_tick = tick - h * geometry.actions_per_block
        selected_ticks = list(range(first_tick, tick + 1, geometry.video_stride))
        missing = [t for t in selected_ticks if t not in observed]
        if missing:
            raise ValueError(f"missing real historical video observations at ticks {missing}")
        frames = [self._image(observed[t], f"history image at tick {t}") for t in selected_ticks]
        future_frames = geometry.actions_per_block // geometry.video_stride
        video = torch.zeros(
            (self.image_shape[0], len(frames) + future_frames, *self.image_shape[1:]), dtype=torch.uint8
        )
        video[:, : len(frames)] = torch.stack(frames, dim=1)

        blocks = h + 1
        action_steps, width = blocks * geometry.actions_per_block, self.template.width
        anchors = torch.zeros((blocks, width), dtype=torch.float32)
        anchors[h] = state
        state_masks = torch.zeros((blocks, width), dtype=torch.bool)
        state_masks[h] = sm
        statistics = copy.deepcopy(self._statistics)
        states, state_clip = statistics.state.normalize(anchors, state_masks)
        metadata = BlockStateMetadata(
            block_size=geometry.latent_frames_per_block,
            history_blocks=self.history_blocks,
            action_frame_ids=torch.arange(1, 1 + blocks * geometry.latent_frames_per_block).repeat_interleave(
                geometry.actions_per_latent_frame
            ),
            action_mask=am.expand(action_steps, -1).clone(),
            states=states,
            state_mask=state_masks,
            state_action_indexes=torch.arange(blocks) * geometry.actions_per_block,
            anchors=anchors,
            state_latent_indexes=torch.arange(blocks) * geometry.latent_frames_per_block,
            source=f"rtc:revision={revision}:ticks={first_tick}..{tick}",
            state_clip_fraction=state_clip,
            statistics=statistics,
            template=copy.deepcopy(self.template),
        )
        batch = copy.deepcopy(self._static)
        batch.update(
            {
                self.video_key: [video],
                "action": [torch.zeros((action_steps, width), dtype=torch.float32)],
                "action_raw": [torch.zeros((action_steps, width), dtype=torch.float32)],
                "sequence_plan": [
                    replace(
                        self._plan,
                        condition_frame_indexes_vision=list(range(1 + h * geometry.latent_frames_per_block)),
                        causal_action_metadata=metadata,
                    )
                ],
                "conditioning_fps": torch.tensor([self.controller_hz / geometry.video_stride], dtype=torch.float32),
                "conditioning_fps_action": torch.tensor([self.controller_hz], dtype=torch.float32),
                "raw_action_dim": [None],
                "causal_action_current_block": h,
                "causal_action_mode": ["policy"],
                "causal_action_cache_mode": self._cache_mode,
                "causal_action_preview": False,
                "is_preprocessed": False,
                "rtc_observation_tick": tick,
                "rtc_window_start_tick": first_tick,
                "rtc_observation_ticks": torch.tensor(selected_ticks, dtype=torch.long),
                "rtc_episode_revision": revision,
            }
        )
        return batch
