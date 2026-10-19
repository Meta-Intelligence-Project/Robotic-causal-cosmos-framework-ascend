# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run causal policy rollouts in an existing RoboTwin simulator installation.

RoboTwin/SAPIEN imports are deferred until main(), so --help and fake-environment
tests work without a simulator. Run this entry point in the RoboTwin Python env.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from cosmos_framework.inference.causal_action.client import CausalPolicyClient


def camera_images(observation):
    cameras = observation["observation"]
    # Detach each frame from renderer buffers before taking the next observation.
    return {
        target: np.asarray(cameras[source]["rgb"]).copy()
        for source, target in (
            ("head_camera", "cam_high"),
            ("left_camera", "cam_left_wrist"),
            ("right_camera", "cam_right_wrist"),
        )
    }


def native_state(observation, env, *, state_source="command"):
    joint = dict(observation["joint_action"])
    if state_source == "measured":
        left = list(env.robot.get_left_arm_real_jointState())
        right = list(env.robot.get_right_arm_real_jointState())
        joint.update(
            left_arm=left[:-1],
            left_gripper=left[-1],
            right_arm=right[:-1],
            right_gripper=right[-1],
            vector=np.asarray(left + right),
        )
    elif state_source != "command":
        raise ValueError(f"Unknown state_source: {state_source}")
    return dict(joint_action=joint, endpose=observation["endpose"])


def episode_finished(env):
    return env.eval_success or env.take_action_cnt >= env.step_lim


class VideoRecorder:
    """Stream copied RGB observations to an MP4 without retaining an episode in RAM."""

    def __init__(self, output, *, fps=30):
        self.path = Path(output) / "video.mp4" if output is not None else None
        self.fps = fps
        self.frames = 0
        self.writer = None
        self.shape = None

    def __enter__(self):
        return self

    def append(self, images):
        if self.path is None:
            return
        import cv2

        head, left, right = (images[name] for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"))
        for image in (head, left, right):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Video recording requires uint8 RGB camera images")
        height = head.shape[0] + max(left.shape[0], right.shape[0])
        width = max(head.shape[1], left.shape[1] + right.shape[1])
        # Preserve camera pixels and pad to even dimensions for the video codec.
        frame = np.zeros((height + height % 2, width + width % 2, 3), dtype=np.uint8)
        head_x = (width - head.shape[1]) // 2
        frame[: head.shape[0], head_x : head_x + head.shape[1]] = head
        frame[head.shape[0] : head.shape[0] + left.shape[0], : left.shape[1]] = left
        frame[head.shape[0] : head.shape[0] + right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
        if self.writer is None:
            self.shape = frame.shape
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (frame.shape[1], frame.shape[0])
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open MP4 writer: {self.path}; check OpenCV video codec support")
        if frame.shape != self.shape:
            raise ValueError("Camera image dimensions changed during video recording")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def __exit__(self, *args):
        if self.writer is not None:
            self.writer.release()


def rollout(env, client, *, state_source="command", output=None, video_fps=30):
    """One episode: initial frame, then exactly one frame per executed target.

    Predicted rows after execute_steps are discarded. If the episode finishes
    during a chunk, no partial feedback or padded observations are submitted.
    """
    action_types = {"joint": "qpos", "eef": "ee"}
    action_type = action_types[client.control_mode]
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executed = 0
    requests = []
    client.reset(prompt=env.get_instruction())

    with VideoRecorder(output, fps=video_fps) as video:
        if not episode_finished(env):
            observation = env.get_obs()
            images = camera_images(observation)
            video.append(images)
            request = dict(
                obs=images,
                state=native_state(observation, env, state_source=state_source),
                observation_step=0,
            )
            while True:
                response = client.infer(request)
                actions = np.asarray(client.actions(response), dtype=np.float32)
                count = response["execute_steps"]
                if type(count) is not int or count < 1 or count != client.metadata["execute_steps"]:
                    raise ValueError("Server execute_steps differs from connection metadata")
                width = (
                    client.metadata["left_arm_dof"] + client.metadata["right_arm_dof"] + 2
                    if client.control_mode == "joint"
                    else 16
                )
                if (
                    actions.shape != (client.metadata["actions_per_block"], width)
                    or len(actions) < count
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError("Server returned an invalid native action chunk")
                requests.append(dict(observation_step=executed, **response.get("server_timing", {})))
                if output is not None:
                    np.save(output / f"step{executed:06d}_action_{client.control_mode}.npy", actions)

                observations = []
                for action in actions[:count]:
                    env.take_action(action, action_type=action_type)
                    observation = env.get_obs()
                    images = camera_images(observation)
                    video.append(images)
                    observations.append(images)
                    executed += 1
                    if episode_finished(env):
                        break
                if episode_finished(env):
                    break
                request = dict(
                    obs=observations,
                    state=native_state(observation, env, state_source=state_source),
                    executed_steps=count,
                    observation_step=executed,
                )

    return dict(
        success=bool(env.eval_success),
        executed_steps=executed,
        seconds=time.perf_counter() - started,
        requests=requests,
        video_path=str(video.path) if video.path is not None else None,
        video_frames=video.frames,
        video_fps=video_fps,
    )


def load_robotwin(root):
    """Reuse this RoboTwin checkout's task/config helpers, without its RPC loop."""
    root = Path(root).expanduser().resolve()
    helper = root / "scripts/eval_policy_xpolicylab.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Expected RoboTwin task/config helpers: {helper}")
    # RoboTwin asset and robot paths are resolved against its repository root.
    os.chdir(root)
    sys.path.insert(0, str(root))
    name = "_cosmos_robotwin_eval_helpers"
    spec = importlib.util.spec_from_file_location(name, helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robotwin-root", required=True, help="Path to the RoboTwin simulator repository")
    p.add_argument("--server-uri", default="ws://127.0.0.1:8000")
    p.add_argument("--task-name", default="adjust_bottle")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--instruction", help="Override RoboTwin-generated instruction; reused across episodes")
    p.add_argument("--instruction-type", choices=["seen", "unseen"], help="Default: task config eval_instruction")
    p.add_argument("--seed", type=int, default=100000, help="First simulator seed; incremented after every attempt")
    p.add_argument("--episodes", type=positive_int, default=1)
    p.add_argument("--control-mode", choices=["joint", "eef"], default="joint")
    p.add_argument("--state-source", choices=["command", "measured"], default="command")
    p.add_argument("--max-steps", type=positive_int, help="Optional per-episode limit, capped by RoboTwin's task limit")
    p.add_argument("--render-freq", type=int, default=0)
    p.add_argument("--video-fps", type=positive_int, default=30, help="Saved observation video playback FPS")
    p.add_argument("--output", default="outputs/causal_action/robotwin_sim")
    return p


def main():
    args = parser().parse_args()
    if args.render_freq < 0:
        raise ValueError("render-freq must be nonnegative")
    if args.instruction is not None and not args.instruction.strip():
        raise ValueError("instruction override must not be empty")
    # Resolve output before load_robotwin changes cwd for simulator asset loading.
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        task=args.task_name,
        task_config=args.task_config,
        instruction=args.instruction,
        server_uri=args.server_uri,
        control_mode=args.control_mode,
        state_source=args.state_source,
        video_fps=args.video_fps,
        episodes=[],
        skipped_seeds=[],
    )
    with CausalPolicyClient(args.server_uri, control_mode=args.control_mode) as client:
        helpers = load_robotwin(args.robotwin_root)
        task_args, _ = helpers.load_task_args(
            dict(
                task_name=args.task_name,
                task_config=args.task_config,
                policy_name="cosmos_causal",
                ckpt_setting="causal_action",
            )
        )
        task_args.update(eval_mode=True, eval_video_log=False, eval_video_save_dir=None, render_freq=args.render_freq)
        task_args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
        task_args["data_type"].update(rgb=True, qpos=True, endpose=True)
        instruction_type = (
            helpers.resolve_eval_instruction(dict(instruction_type=args.instruction_type), task_args)
            if args.instruction is None
            else None
        )
        summary["instruction_type"] = instruction_type
        episode, seed = 0, args.seed
        while episode < args.episodes:
            env = helpers.class_decorator(args.task_name)
            env.suc, env.test_num = 0, episode
            record = dict(episode=episode, seed=seed)
            try:
                setup_args = dict(now_ep_num=episode, seed=seed, is_test=True, **task_args)
                env.setup_demo(**(dict(setup_args, render_freq=0) if args.instruction is None else setup_args))
                for side in ("left", "right"):
                    actual = len(getattr(env.robot, f"{side}_arm_joints"))
                    if actual != client.metadata[f"{side}_arm_dof"]:
                        raise ValueError(f"{side} arm DOF differs between RoboTwin ({actual}) and server")
                instruction = args.instruction
                if instruction is None:
                    # RoboTwin tasks populate object/arm placeholders in play_once().
                    # Generate text before closing, then restore the initial scene;
                    # expert observations must never enter the policy's video cache.
                    episode_info = env.play_once()
                    instruction = helpers.build_instruction(
                        task_args, episode_info, instruction_type, args.episodes, seed
                    )
                    if not isinstance(instruction, str) or not instruction.strip() or instruction == args.task_name:
                        raise ValueError("RoboTwin instruction generation failed; supply --instruction to override")
                    record.update(
                        instruction=instruction, instruction_source="robotwin", instruction_type=instruction_type
                    )
                    helpers.safe_close_env(env)
                    env.setup_demo(**setup_args)
                else:
                    record.update(instruction=instruction, instruction_source="override", instruction_type=None)
                if args.max_steps is not None:
                    env.step_lim = min(env.step_lim, args.max_steps)
                env.set_instruction(instruction=instruction)
                print(f"Episode {episode}, seed {seed}, instruction: {instruction}", flush=True)
                record.update(
                    rollout(
                        env,
                        client,
                        state_source=args.state_source,
                        output=output / f"episode{episode:03d}" / f"seed{seed}",
                        video_fps=args.video_fps,
                    )
                )
                record["status"] = "finished"
                print(json.dumps(record), flush=True)
            except helpers.UnStableError as error:
                record.update(status="skipped_unstable", error=f"{type(error).__name__}: {error}")
                print(
                    f"Episode {episode}, skipping unstable seed {seed}: {error}; retrying seed {seed + 1}", flush=True
                )
                continue
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                helpers.safe_close_env(env)
                summary["skipped_seeds" if record.get("status") == "skipped_unstable" else "episodes"].append(record)
                (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                seed += 1
            episode += 1


if __name__ == "__main__":
    main()
