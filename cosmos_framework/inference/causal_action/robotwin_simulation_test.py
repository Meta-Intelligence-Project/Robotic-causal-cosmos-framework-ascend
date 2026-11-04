# SPDX-License-Identifier: OpenMDW-1.1
"""Virtual RTC uses completed RPC timings and observations of executed actions."""

import copy

import numpy as np
import pytest
import torch

from cosmos_framework.inference.causal_action import simulation_session
from cosmos_framework.inference.causal_action.client import CausalPolicyClient
from cosmos_framework.inference.causal_action.robotwin_client import rollout
from cosmos_framework.inference.causal_action.robotwin_client_test import (
    FakeClock,
    FakeEnv,
    adapter,
    command,
    prepare_image,
)
from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
from cosmos_framework.inference.causal_action.rtc_session import RTCDeadlineMiss


class ObservedActionEnv(FakeEnv):
    """Pixels encode the actual measured joint after actuator interpolation."""

    def get_obs(self):
        observation = super().get_obs()
        self.buffer[0, :4, 0] = np.frombuffer(np.float32(self.joints[0]).tobytes(), dtype=np.uint8)
        return observation


class SimulationClient(CausalPolicyClient):
    """Production client/session with an explicit server-owned old/new plan."""

    def __init__(self, env, *, prediction_seconds):
        robot = adapter()
        self.env, self.prediction_seconds = env, prediction_seconds
        self.control_mode = "joint"
        self.metadata = dict(
            simulation_sync=True,
            returned_action_steps=16,
            actions_per_block=32,
            execute_steps=16,
            history_blocks=32,
            video_stride=4,
            controller_hz=30.0,
            resolution="480",
            rtc_enabled=True,
            state_mask=robot.state_mask.numpy(),
            action_mask=robot.action_mask.numpy(),
        )
        self._session = self._full = None
        self.request_timings, self.requests, self.returned = [], [], []

    def infer(self, observation):
        self.requests.append(copy.deepcopy(observation))
        count = self.env.take_action_cnt
        seconds = self.prediction_seconds(observation) if callable(self.prediction_seconds) else self.prediction_seconds
        self.env.clock.now += seconds
        assert self.env.take_action_cnt == count
        tick = observation.get("tick", 0)
        operation = observation.get("operation", "predict")
        if operation == "reset":
            self._full = torch.stack([command(i) for i in range(32)])
            actions, info = self._full[:16].clone(), None
        elif operation == "calibrate":
            assert tick == 0 and not self.env.executed
            actions, info = self._full[:16].clone(), None
        else:
            delay = observation["simulation_rtc"]["delay_steps"]
            new = torch.stack([command(tick + i) for i in range(32)])
            new[:, 0] += tick / 100
            new[:delay] = self._full[16 : 16 + delay]
            self._full = new
            actions = new[:16].clone()
            info = dict(delay_steps=delay, old_action_steps=delay, new_action_steps=16 - delay, request_tick=tick)
        self.returned.append(actions.clone())
        return dict(actions=actions.numpy(), simulation_info=info)


def run_simulation(env, client, monkeypatch, *, max_waypoints=33, **options):
    monkeypatch.setattr(simulation_session, "perf_counter", env.clock)

    def no_wall_sleep(seconds):
        raise AssertionError(f"Synchronous simulation must not sleep for {seconds} wall seconds")

    return rollout(
        env,
        client,
        adapter=adapter(),
        prepare_image=prepare_image,
        clock=env.clock,
        sleep=no_wall_sleep,
        execution_hz=options.pop("execution_hz", 30),
        servo_hz=30,
        max_waypoints=max_waypoints,
        **options,
    )


@pytest.mark.parametrize("prediction_seconds", [0.1, 30.0])
def test_simulator_returns_observations_of_the_actual_mixed_actions(prediction_seconds, monkeypatch):
    env = ObservedActionEnv(FakeClock(), action_seconds=2.0, limit=10000)
    client = SimulationClient(env, prediction_seconds=prediction_seconds)
    result = run_simulation(env, client, monkeypatch)
    assert result["simulation_sync"] and result["simulated_rtc"] and result["virtual_rtc_feasible"]
    assert result["continuous_rtc"] is False
    assert result["executed_waypoints"] == 33
    assert result["skipped_servo_ticks"] == 0
    assert result["max_servo_lateness_seconds"] == result["max_waypoint_lateness_seconds"] == 0
    assert [request.get("operation", "predict") for request in client.requests] == [
        "reset",
        "calibrate",
        "calibrate",
        "calibrate",
        "predict",
        "predict",
    ]
    assert [request["tick"] for request in client.requests] == [0, 0, 0, 0, 16, 32]
    status = result["final_status"]
    assert status["prediction_horizon"] == 32
    assert status["execution_mode"] == "simulation_rtc"
    assert status["last_request"]["accepted"]
    assert not status["last_request"]["deadline_miss"]
    assert status["last_request"]["actual_delay_steps"] <= status["last_request"]["delay_steps"] <= 16
    targets = {
        trace["tick"]: action[1][0]
        for action, trace in zip(env.executed, result["servo_trace"], strict=True)
        if trace["alpha"] == 1
    }
    for request in client.requests[4:]:
        tick = request["tick"]
        assert "history" not in request and request["rtc"] is None
        observations = request["executed_observations"]
        assert [record["tick"] for record in observations] == list(range(tick - 15, tick + 1))
        for record in observations:
            encoded_joint = np.frombuffer(record["image"][0, 0, :4].tobytes(), dtype=np.float32)[0]
            assert encoded_joint == targets[record["tick"] - 1]
        np.testing.assert_array_equal(request["image"], observations[-1]["image"])
        delay = request["simulation_rtc"]["delay_steps"]
        for offset, trace in enumerate(result["action_trace"][tick : tick + 16]):
            provenance = trace["action_provenance"]
            old = offset < delay
            assert provenance["source"] == ("old_prefix" if old else "new_prediction")
            assert provenance["prediction_tick"] == (tick - 16 if old else tick)
            assert provenance["prediction_row"] == (16 + offset if old else offset)
    assert all(trace["action_provenance"]["source"] == "bootstrap" for trace in result["action_trace"][:16])
    expected = torch.cat([client.returned[0], client.returned[4], client.returned[5][:1]])[:, 0].numpy()
    np.testing.assert_array_equal(list(targets.values()), expected)
    if prediction_seconds == 0.1:
        assert 0 < status["last_request"]["old_action_steps"] < 16
    else:
        assert result["rate_changes"]
        assert status["last_request"]["old_action_steps"] == 16


def test_latency_fit_uses_completed_history_samples_and_reduces_the_rate(monkeypatch):
    env = ObservedActionEnv(FakeClock(), limit=10000)

    def latency(request):
        history = (request.get("tick", 0) // 32) * 32 / 30
        return 0.1 + 0.05 * history

    client = SimulationClient(env, prediction_seconds=latency)
    result = run_simulation(
        env, client, monkeypatch, max_waypoints=257, execution_policy=RTCExecutionPolicy(extra_latency_seconds=0.3)
    )
    status = result["final_status"]
    assert len(status["latency_samples"]) == 8
    assert status["execution_advice"]["fit_mode"] == "linear"
    assert status["execution_advice"]["slope_seconds_per_history_second"] == pytest.approx(0.05)
    assert result["final_execution_hz"] < result["initial_execution_hz"]
    assert result["virtual_rtc_feasible"]
    assert all(change["execution_hz"] <= change["previous_hz"] for change in result["rate_changes"])


def test_unexpected_latency_spike_stops_with_a_visible_deadline_miss(monkeypatch, caplog):
    env = ObservedActionEnv(FakeClock())
    client = SimulationClient(
        env, prediction_seconds=lambda request: 2.0 if request.get("operation", "predict") == "predict" else 0.1
    )
    with pytest.raises(RTCDeadlineMiss, match="reserved"):
        run_simulation(env, client, monkeypatch)
    status = client._session.status()
    assert status["last_request"]["actual_delay_steps"] > status["last_request"]["delay_steps"]
    assert status["last_request"]["deadline_miss"] and not status["last_request"]["accepted"]
    assert not status["virtual_rtc_feasible"]
    assert len(env.executed) == 16
    assert "stopping simulation" in caplog.text


def test_servo_quantization_respects_minimum_execution_rate(monkeypatch):
    env = ObservedActionEnv(FakeClock())
    client = SimulationClient(env, prediction_seconds=0.5)
    with pytest.raises(RTCDeadlineMiss, match="minimum"):
        run_simulation(env, client, monkeypatch, execution_policy=RTCExecutionPolicy(minimum_execution_hz=20))
    assert not env.executed
    assert client._session.status()["tick"] == 0


@pytest.mark.parametrize("servo_steps", [11, 13])
def test_fractional_rate_finishes_on_the_exact_integer_servo_step(servo_steps, monkeypatch):
    env = ObservedActionEnv(FakeClock())
    client = SimulationClient(env, prediction_seconds=0.1)
    result = run_simulation(env, client, monkeypatch, max_waypoints=3, execution_hz=30 / servo_steps)
    assert result["servo_commands"] == 3 * servo_steps
    assert [trace["alpha"] for trace in result["servo_trace"]][servo_steps - 1 :: servo_steps] == [1.0] * 3
