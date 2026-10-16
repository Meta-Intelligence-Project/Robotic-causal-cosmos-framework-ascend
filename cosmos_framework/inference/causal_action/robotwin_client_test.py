# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise simulator integration with fakes; never import or run RoboTwin."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.inference.causal_action import robotwin_client as runner


class FakeEnv:
    def __init__(self, *, limit=35, success_at=None):
        self.step_lim = limit
        self.success_at = success_at
        self.eval_success = False
        self.take_action_cnt = 0
        self.executed = []
        self.buffer = np.zeros((2, 3, 3), dtype=np.uint8)
        self.robot = SimpleNamespace(
            left_arm_joints=[None] * 6,
            right_arm_joints=[None] * 6,
            get_left_arm_real_jointState=lambda: [1.0] * 6 + [0.25],
            get_right_arm_real_jointState=lambda: [2.0] * 6 + [0.75],
        )

    def get_obs(self):
        # Real renderers may reuse buffers; later steps must not overwrite history.
        self.buffer.fill(self.take_action_cnt)
        return dict(
            observation={name: dict(rgb=self.buffer) for name in ("head_camera", "left_camera", "right_camera")},
            joint_action=dict(
                left_arm=[self.take_action_cnt] * 6,
                right_arm=[self.take_action_cnt] * 6,
                left_gripper=0.0,
                right_gripper=1.0,
            ),
            endpose=dict(left_endpose=[0, 0, 0, 1, 0, 0, 0], right_endpose=[0, 0, 0, 1, 0, 0, 0]),
        )

    def take_action(self, action, *, action_type):
        assert not runner.episode_finished(self)
        self.executed.append((action.copy(), action_type))
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at

    def get_instruction(self):
        return self.instruction

    def set_instruction(self, *, instruction):
        self.instruction = instruction

    def setup_demo(self, **kwargs):
        self.setup_args = kwargs


class FakeClient:
    def __init__(self, uri=None, *, control_mode="joint"):
        self.control_mode = control_mode
        self.metadata = dict(actions_per_block=32, execute_steps=16, left_arm_dof=6, right_arm_dof=6)
        self.requests = []
        self.resets = []
        self.closed = False

    def reset(self, *, prompt):
        self.resets.append(prompt)

    def infer(self, request):
        self.requests.append(request)
        width = 14 if self.control_mode == "joint" else 16
        actions = np.zeros((32, width), dtype=np.float32)
        actions[:16] = np.arange(16)[:, None]
        actions[16:] = 999  # Must be discarded, never executed.
        return dict(action={self.control_mode: actions}, execute_steps=16, server_timing=dict(infer_ms=1.0))

    def actions(self, response):
        return response["action"][self.control_mode]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def test_rollout_sends_post_action_history_and_discards_unexecuted_predictions(tmp_path):
    env, client = FakeEnv(), FakeClient()
    env.set_instruction(instruction="Adjust the bottle.")
    result = runner.rollout(env, client, output=tmp_path)
    assert client.resets == ["Adjust the bottle."]
    assert result["executed_steps"] == 35
    assert not result["success"]
    assert [r["observation_step"] for r in client.requests] == [0, 16, 32]
    assert [r["observation_step"] for r in result["requests"]] == [0, 16, 32]
    assert client.requests[0]["obs"]["cam_high"][0, 0, 0] == 0
    for request, start in zip(client.requests[1:], (1, 17)):
        assert request["executed_steps"] == 16
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert [int(frame[camera][0, 0, 0]) for frame in request["obs"]] == list(range(start, start + 16))
        assert request["state"]["joint_action"]["left_arm"] == [start + 15] * 6
        assert "endpose" in request["state"]
    assert [int(action[0]) for action, _ in env.executed] == list(range(16)) * 2 + [0, 1, 2]
    assert {kind for _, kind in env.executed} == {"qpos"}
    saved = sorted(tmp_path.glob("*.npy"))
    assert len(saved) == 3
    assert np.load(saved[0]).shape == (32, 14)


def test_eef_rollout_stops_on_success_mid_chunk():
    env, client = FakeEnv(success_at=3), FakeClient(control_mode="eef")
    env.set_instruction(instruction="Move.")
    result = runner.rollout(env, client)
    assert result["success"]
    assert result["executed_steps"] == 3
    assert len(client.requests) == 1
    assert {kind for _, kind in env.executed} == {"ee"}
    assert all(action.shape == (16,) for action, _ in env.executed)


def test_measured_state_replaces_joint_values_without_mutating_get_obs():
    env = FakeEnv()
    observation = env.get_obs()
    command = runner.native_state(observation, env)
    measured = runner.native_state(observation, env, state_source="measured")
    assert command["joint_action"]["left_arm"] == [0] * 6
    assert observation["joint_action"]["left_gripper"] == 0.0
    assert measured["joint_action"]["left_arm"] == [1.0] * 6
    np.testing.assert_array_equal(measured["joint_action"]["vector"], [1.0] * 6 + [0.25] + [2.0] * 6 + [0.75])
    assert measured["endpose"] == observation["endpose"]


def test_invalid_action_chunk_is_rejected_before_simulator_execution():
    env, client = FakeEnv(), FakeClient()
    env.set_instruction(instruction="Move.")
    client.actions = lambda response: np.full((32, 14), np.nan)
    with pytest.raises(ValueError, match="invalid native action"):
        runner.rollout(env, client)
    assert not env.executed


def test_robotwin_root_is_required_even_with_environment_variable(monkeypatch, capsys):
    monkeypatch.setenv("ROBOTWIN_ROOT", "/path/to/RoboTwin")
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args([])
    assert error.value.code == 2
    assert "--robotwin-root" in capsys.readouterr().err


@pytest.mark.parametrize("fail_setup", [False, True])
def test_main_loads_config_resets_each_episode_and_always_closes_env(monkeypatch, tmp_path, fail_setup):
    client, envs, closed = FakeClient(), [], []
    config = dict(camera={}, data_type={})

    def make_env(task):
        assert task == "adjust_bottle"
        env = FakeEnv()
        if fail_setup:

            def fail(**kwargs):
                raise RuntimeError("setup failed")

            env.setup_demo = fail
        envs.append(env)
        return env

    helpers = SimpleNamespace(
        load_task_args=lambda args: (config, "aloha-agilex"),
        class_decorator=make_env,
        safe_close_env=lambda env: closed.append(env),
    )
    monkeypatch.setattr(runner, "load_robotwin", lambda root: helpers)
    monkeypatch.setattr(runner, "CausalPolicyClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robotwin_client.py",
            "--robotwin-root",
            "/path/to/RoboTwin",
            "--instruction",
            "Adjust.",
            "--seed",
            "42",
            "--episodes",
            "2",
            "--max-steps",
            "2",
            "--output",
            str(tmp_path),
        ],
    )
    if fail_setup:
        with pytest.raises(RuntimeError, match="setup failed"):
            runner.main()
    else:
        runner.main()
        assert [env.setup_args["seed"] for env in envs] == [42, 43]
        assert client.resets == ["Adjust."] * 2
        assert all(env.step_lim == 2 for env in envs)
        assert all(env.setup_args["is_test"] and env.setup_args["eval_mode"] for env in envs)
        assert config["camera"] == dict(collect_head_camera=True, collect_wrist_camera=True)
        assert config["data_type"] == dict(rgb=True, qpos=True, endpose=True)
        assert not config["eval_video_log"] and config["eval_video_save_dir"] is None
    assert closed == envs
    assert client.closed
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["episodes"]) == (1 if fail_setup else 2)
    assert metrics["episodes"][0]["status"] == ("error" if fail_setup else "finished")
    if not fail_setup:
        assert metrics["episodes"][0]["instruction"] == "Adjust."
        assert metrics["episodes"][0]["instruction_source"] == "override"


@pytest.fixture
def automatic_main(monkeypatch, tmp_path):
    client, envs, events = FakeClient(), [], []
    config = dict(task_name="adjust_bottle", eval_instruction="seen", camera={}, data_type={})

    class AutoEnv(FakeEnv):
        def setup_demo(self, **kwargs):
            super().setup_demo(**kwargs)
            self.take_action_cnt, self.eval_success = 0, False
            events.append(("setup", kwargs["seed"], kwargs["render_freq"]))

        def play_once(self):
            events.append(("expert", self.setup_args["seed"]))
            assert self.step_lim == 35  # The policy-only max-steps must not truncate the expert.
            self.take_action_cnt = 35
            return dict(info={"{A}": "bottle", "{a}": "left"})

    def make_env(task):
        env = AutoEnv()
        envs.append(env)
        return env

    def build_instruction(args, episode_info, kind, count, seed):
        assert args["task_name"] == "adjust_bottle"
        assert episode_info["info"] == {"{A}": "bottle", "{a}": "left"}
        assert count == 2
        events.append(("instruction", seed, kind))
        return f"Lift the bottle with the left arm. Seed {seed}."

    helpers = SimpleNamespace(
        load_task_args=lambda args: (config, "aloha-agilex"),
        resolve_eval_instruction=lambda args, task: args["instruction_type"] or task["eval_instruction"],
        build_instruction=build_instruction,
        class_decorator=make_env,
        safe_close_env=lambda env: events.append(("close", env.setup_args["seed"])),
    )
    monkeypatch.setattr(runner, "load_robotwin", lambda root: helpers)
    monkeypatch.setattr(runner, "CausalPolicyClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robotwin_client.py",
            "--robotwin-root",
            "/path/to/RoboTwin",
            "--seed",
            "42",
            "--episodes",
            "2",
            "--max-steps",
            "2",
            "--render-freq",
            "10",
            "--output",
            str(tmp_path),
        ],
    )
    return client, envs, events, helpers


@pytest.mark.parametrize("instruction_type", [None, "unseen"])
def test_automatic_instruction_uses_episode_metadata_and_recreates_scene(automatic_main, tmp_path, instruction_type):
    client, envs, events, _ = automatic_main
    if instruction_type is not None:
        sys.argv.extend(["--instruction-type", instruction_type])
    runner.main()
    kind = instruction_type or "seen"
    assert events == [
        (event, *values)
        for seed in (42, 43)
        for event, values in (
            ("setup", (seed, 0)),
            ("expert", (seed,)),
            ("instruction", (seed, kind)),
            ("close", (seed,)),
            ("setup", (seed, 10)),
            ("close", (seed,)),
        )
    ]
    assert client.resets == [f"Lift the bottle with the left arm. Seed {seed}." for seed in (42, 43)]
    assert len(client.requests) == 2
    assert all(request["obs"]["cam_high"][0, 0, 0] == 0 for request in client.requests)
    assert all(len(env.executed) == 2 for env in envs)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert [record["instruction"] for record in metrics["episodes"]] == client.resets
    assert all(record["instruction_source"] == "robotwin" for record in metrics["episodes"])
    assert metrics["instruction_type"] == kind


@pytest.mark.parametrize("instruction", ["adjust_bottle", "", None])
def test_failed_automatic_instruction_is_recorded_and_never_sent(automatic_main, tmp_path, instruction):
    client, _, events, helpers = automatic_main
    helpers.build_instruction = lambda *args: instruction
    with pytest.raises(ValueError, match="instruction generation failed"):
        runner.main()
    assert not client.resets and not client.requests
    assert events[-1] == ("close", 42)
    assert client.closed
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["episodes"][0]["status"] == "error"
