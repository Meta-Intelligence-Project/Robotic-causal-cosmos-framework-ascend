# SPDX-License-Identifier: OpenMDW-1.1
"""Stateless model requests using the validated local ordinary/RTC planner.

Client helpers import CPU preprocessing lazily and never load a checkpoint.
Only the live episode's prompt/schema are retained; video/KV values are local
to each request, including repeated startup calibration requests.
"""

import json
import math
import os
from dataclasses import asdict, replace

import numpy as np

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def prepare_camera_observation(cameras, resolution="480"):
    """Return the training concat-view CPU image and its exact padding metadata."""
    import torch

    from cosmos_framework.data.generator.action.transforms import VideoResize
    from cosmos_framework.data.generator.action.video_view import VideoViewConfig

    frames = {}
    for key in CAMERA_KEYS:
        value = np.asarray(cameras[key])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 2:
            raise ValueError(f"{key} must be a real RGB uint8 [H,W,3] observation")
        frames[key] = torch.from_numpy(value.copy()).permute(2, 0, 1).unsqueeze(0)
    view = VideoViewConfig(cameras=dict(zip(("head", "left", "right"), CAMERA_KEYS, strict=True)))
    video = view.compose(frames, 1, viewpoint="concat_view")
    prepared = VideoResize()({"video": video}, resolution)
    return prepared["video"][:, 0].contiguous(), prepared["image_size"].clone()


def adapter_from_metadata(metadata):
    """Rebuild only the CPU native state/action adapter advertised by the server."""
    from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
    from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter

    template = ActionStateTemplate55()
    if metadata.get("template_id") != template.template_id:
        raise ValueError("The remote policy does not use the unified55 action template")
    return RobotwinAdapter(
        template,
        state_mask=metadata["state_mask"],
        action_mask=metadata["action_mask"],
        **metadata["adapter"],
    )


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class ServingModelPolicy:
    """One episode's schema with full observed history supplied on every request."""

    def __init__(
        self,
        model,
        *,
        statistics,
        template=None,
        adapter=None,
        history_blocks=32,
        execute_steps=16,
        controller_hz=30.0,
        resolution="480",
        num_steps=10,
        guidance=1.0,
        cache_mode="full",
        enable_rtc=False,
        seed=42,
    ):
        from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
        from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
        from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter

        self.model, self.statistics = model, statistics
        self.template = template if template is not None else ActionStateTemplate55()
        self.adapter = adapter if adapter is not None else RobotwinAdapter.from_statistics(self.template, statistics)
        self.statistics.validate(self.template.width, self.adapter.state_mask, self.adapter.action_mask)
        self.geometry = CausalBlockGeometry(temporal_compression_factor=4)
        if statistics.provenance.get("geometry") != asdict(self.geometry):
            raise ValueError("Serving statistics must declare the same 32-action/stride4/VAE4 geometry")
        self.history_blocks = _integer(history_blocks, "history_blocks", 1)
        self.execute_steps = _integer(execute_steps, "execute_steps", 1)
        if not self.execute_steps < self.geometry.actions_per_block:
            raise ValueError("execute_steps must be smaller than the prediction horizon")
        if isinstance(controller_hz, bool) or not math.isfinite(controller_hz) or controller_hz <= 0:
            raise ValueError("controller_hz must be positive and finite")
        self.controller_hz = float(controller_hz)
        self.resolution, self.num_steps, self.guidance = resolution, _integer(num_steps, "num_steps", 1), guidance
        if isinstance(guidance, bool) or not math.isfinite(guidance):
            raise ValueError("guidance must be finite")
        if cache_mode not in ("off", "plans", "full"):
            raise ValueError("cache_mode must be off, plans or full")
        self.cache_mode, self.enable_rtc, self.seed = cache_mode, bool(enable_rtc), _integer(seed, "seed")
        model.net.eval().requires_grad_(False)
        self.clear()

    @property
    def metadata(self):
        adapter = dict(
            left_arm_dof=self.adapter.dofs["left"],
            right_arm_dof=self.adapter.dofs["right"],
            eef_pose_frame=self.adapter.eef_pose_frame,
            T_world_base=self.adapter.T_world_base,
        )
        return dict(
            actions_per_block=self.geometry.actions_per_block,
            execute_steps=self.execute_steps,
            controller_hz=self.controller_hz,
            fps=self.controller_hz,
            history_blocks=self.history_blocks,
            video_stride=self.geometry.video_stride,
            resolution=self.resolution,
            rtc_enabled=self.enable_rtc,
            state_mask=self.adapter.state_mask.numpy().copy(),
            action_mask=self.adapter.action_mask.numpy().copy(),
            adapter=adapter,
            template_id=self.template.template_id,
            available_control_modes=list(self.adapter.available_control_modes),
            seed=self.seed,
            num_steps=self.num_steps,
            guidance=self.guidance,
        )

    def clear(self):
        self.builder = self.planner = self._reference = None
        self.prompt = None

    def _measurement(self, request):
        import torch

        state = torch.as_tensor(request["state"], dtype=torch.float32).detach().cpu().clone()
        if state.shape != (self.template.width,):
            raise ValueError("state must be one physical unified55 measurement")
        state = self.template.sanitize(state, self.adapter.state_mask)
        image = torch.as_tensor(request["image"]).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError("image must be a real prepared CPU uint8 [3,H,W] observation")
        return state, image.clone()

    def _build_reference(self, state, image, image_size, prompt):
        import torch

        from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
        from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
        from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
        from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode
        from cosmos_framework.data.generator.action.video_view import VideoViewConfig

        image_size = torch.as_tensor(image_size, dtype=torch.float32).detach().cpu()
        if (
            image_size.shape != (4,)
            or not torch.isfinite(image_size).all()
            or (image_size <= 0).any()
            or not torch.equal(image_size, image_size.round())
            or tuple(image_size[:2].int().tolist()) != tuple(image.shape[1:])
            or (image_size[2:] > image_size[:2]).any()
        ):
            raise ValueError("image_size must declare the prepared image canvas and valid content region")
        g = self.geometry
        sm, am = self.adapter.state_mask, self.adapter.action_mask
        normalized, clip = self.statistics.state.normalize(state[None], sm[None])
        metadata = BlockStateMetadata(
            block_size=g.latent_frames_per_block,
            history_blocks=self.history_blocks,
            action_frame_ids=torch.arange(1, 1 + g.latent_frames_per_block).repeat_interleave(
                g.actions_per_latent_frame
            ),
            action_mask=am.expand(g.actions_per_block, -1).clone(),
            states=normalized,
            state_mask=sm[None].clone(),
            state_action_indexes=torch.zeros(1, dtype=torch.long),
            anchors=state[None].clone(),
            state_latent_indexes=torch.zeros(1, dtype=torch.long),
            statistics=self.statistics,
            template=self.template,
            state_clip_fraction=clip,
            source="live-policy-bootstrap-schema",
        )
        video = torch.zeros((3, 1 + g.actions_per_block // g.video_stride, *image.shape[1:]), dtype=torch.uint8)
        video[:, 0] = image
        sample = dict(
            video=video,
            action=torch.zeros(g.actions_per_block, self.template.width),
            action_raw=torch.zeros(g.actions_per_block, self.template.width),
            ai_caption=prompt,
            mode="policy",
            viewpoint="concat_view",
            conditioning_fps=torch.tensor(self.controller_hz / g.video_stride),
            conditioning_fps_action=torch.tensor(self.controller_hz),
            image_size=image_size.clone(),
            additional_view_description=VideoViewConfig(cameras={}).describe(viewpoint="concat_view"),
            domain_id=torch.tensor(0),
            action_processing_record=ActionProcessingRecord(self.template.width, None),
            causal_action_cache_mode=self.cache_mode,
        )
        sample = ActionPromptJsonFormatter()(sample)
        # Training's TextTokenizerTransform serializes this structure before
        # passing it to the chat template. Serving tokenizes inside the model.
        sample["ai_caption"] = json.dumps(sample["ai_caption"])
        sample["sequence_plan"] = replace(
            build_sequence_plan_from_mode(
                "policy", video_length=video.shape[1], action_length=g.actions_per_block, video_temporal_downsample=4
            ),
            causal_action_metadata=metadata,
        )
        return sample

    def infer(self, request):
        import torch

        from cosmos_framework.inference.causal_action.rtc import RTCPrefix
        from cosmos_framework.inference.causal_action.rtc_batch import RTCObservedBatchBuilder
        from cosmos_framework.inference.causal_action.rtc_session import RTCObservation, make_model_planner

        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        operation = request.get("operation")
        tick = _integer(request.get("tick", 0), "tick")
        revision = _integer(request.get("revision", 0), "revision")
        if operation == "reset":
            self.clear()
            prompt = request.get("prompt")
            if tick or revision or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Reset requires tick/revision zero and a nonempty task prompt")
            state, image = self._measurement(request)
            self._reference = self._build_reference(state, image, request["image_size"], prompt)
            self.builder = RTCObservedBatchBuilder(
                self._reference, controller_hz=self.controller_hz, history_blocks=self.history_blocks
            )
            self.planner = make_model_planner(
                self.model,
                self.builder,
                template=self.template,
                horizon=self.geometry.actions_per_block,
                controller_hz=self.controller_hz,
                history_blocks=self.history_blocks,
                num_steps=self.num_steps,
                guidance=self.guidance,
                seed=[self.seed],
            )
            self.prompt = prompt
        elif operation == "predict":
            if self.planner is None:
                raise ValueError("Reset with a fresh observation before requesting predictions")
            state, image = self._measurement(request)
        else:
            raise ValueError("operation must be reset or predict")
        rtc = None
        prefix = request.get("rtc")
        if prefix is not None:
            if operation == "reset" or not self.enable_rtc:
                raise ValueError("RTC predictions require an enabled server and an ordinary bootstrap first")
            rtc = RTCPrefix(
                torch.as_tensor(prefix["previous_actions"], dtype=torch.float32).detach().cpu().clone(),
                prefix["delay_steps"],
                prefix.get("beta", 5.0),
            )
        history = []
        for item in request.get("history", []):
            frame = torch.as_tensor(item["image"]).detach().cpu()
            history.append(RTCObservation(_integer(item["tick"], "history tick"), frame))
        inputs = dict(
            state=state,
            state_mask=self.adapter.state_mask,
            action_mask=self.adapter.action_mask,
            image=image,
            history=tuple(history),
            tick=tick,
            revision=revision,
            rtc=rtc,
        )
        actions = self.planner(**inputs)
        return dict(actions=actions.numpy(), tick=tick, revision=revision)


def load_policy(args):
    """Load one model rank; no training dataset or recorded observation is needed."""
    os.environ.setdefault("COSMOS_DEVICE", "npu")
    os.environ["WAN_VAE_PATH"] = args.vae
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The causal action RPC service supports one model rank")
    if os.environ["COSMOS_DEVICE"] == "npu":
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

    from cosmos_framework.data.generator.action.action_state_template import resolve_action_template
    from cosmos_framework.data.generator.action.block_state import BlockStatistics
    from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

    statistics = BlockStatistics.load(args.statistics)
    template = resolve_action_template(
        getattr(args, "template", "cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55")
    )
    adapter = RobotwinAdapter.from_statistics(template, statistics, getattr(args, "robot_config", None))
    distributed.init()
    model, _ = load_model_from_checkpoint(
        "action_causal_midtrain_edge",
        checkpoint_path=args.checkpoint,
        parallelism_config=dict(
            data_parallel_shard_degree=1,
            data_parallel_replicate_degree=1,
            context_parallel_shard_degree=1,
            cfg_parallel_shard_degree=1,
            enable_inference_mode=True,
        ),
        compile_config=dict(enabled=False),
        experiment_opts=[
            "data_setting.action.actions_per_block=32",
            "data_setting.action.video_stride=4",
            "model.config.vlm_config.tokenizer.repository=null",
            "model.config.vlm_config.tokenizer.revision=null",
            f"+model.config.vlm_config.tokenizer.tokenizer_type={args.processor}",
            f"model.config.tokenizer.vae_path={args.vae}",
            *getattr(args, "override", []),
        ],
        keys_to_skip_loading=["net_ema."],
        seed=args.seed,
    )
    return ServingModelPolicy(
        model,
        statistics=statistics,
        template=template,
        adapter=adapter,
        history_blocks=args.history,
        execute_steps=args.execute_steps,
        controller_hz=args.fps,
        resolution=args.resolution,
        num_steps=args.steps,
        guidance=args.guidance,
        cache_mode=getattr(args, "cache_mode", "full"),
        enable_rtc=args.enable_rtc,
        seed=args.seed,
    )
