# SPDX-License-Identifier: OpenMDW-1.1
"""Simulated RTC sends executed old/new mixtures with a fixed 16-step wire contract."""

import copy
import time

import numpy as np
import pytest
import torch

from cosmos_framework.inference.causal_action.client import CausalPolicyClient
from cosmos_framework.inference.causal_action.network_test import live_server
from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
from cosmos_framework.inference.causal_action.server import PolicyConnection, parser
from cosmos_framework.inference.causal_action.serving_model import load_policy
from cosmos_framework.inference.causal_action.serving_model_test import Model, image, policy, reset_request
from cosmos_framework.inference.causal_action.wire import unpackb


class MarkerModel(Model):
    """Give all 32 predicted actions distinguishable, deterministic row values."""

    def __init__(self, delay=0.0):
        super().__init__()
        self.delay = delay
        self.generated_counts = []

    def generate_samples_from_batch(self, batch, **options):
        time.sleep(self.delay)
        result = super().generate_samples_from_batch(batch, **options)
        indexes = torch.where(result["generated_action_mask"][0])[0]
        self.generated_counts.append(len(indexes))
        result["action"][0][indexes, 0] = (
            torch.arange(len(indexes), dtype=torch.float32) / 1000 + (len(self.calls) - 1) / 10
        )
        # Deliberately omit the sampler's committed export: the server must still
        # preserve the exact physical commands committed during the virtual delay.
        result.pop("rtc_committed_actions", None)
        return result


def service(*, simulation_sync=True, history_blocks=2, delay=0.0):
    result = policy(enable_rtc=False, simulation_sync=simulation_sync, history_blocks=history_blocks)
    result.model = MarkerModel(delay=delay)
    result.gate = None  # The shared localhost server fixture can release gated fake policies.
    return result


def measurement(service, tick=0, *, include_size=False):
    state = torch.zeros(55)
    state[0] = tick / 100
    result = dict(
        state=state,
        state_mask=service.adapter.state_mask.clone(),
        action_mask=service.adapter.action_mask.clone(),
        image=torch.from_numpy(image(tick)),
    )
    if include_size:
        result["image_size"] = torch.tensor([16, 16, 16, 16], dtype=torch.float32)
    return result


def request(tick=16, *, episode="first", request_id=1, delay_steps=4):
    state = np.zeros(55, dtype=np.float32)
    state[0] = tick / 100
    return dict(
        operation="predict",
        episode_id=episode,
        request_id=request_id,
        tick=tick,
        revision=0,
        state=state,
        image=image(tick),
        rtc=None,
        simulation_rtc=dict(delay_steps=delay_steps, beta=5.0),
        executed_observations=[dict(tick=index, image=image(index)) for index in range(tick - 15, tick + 1)],
    )


def reset_connection(lifecycle, *, episode="first", request_id=0):
    return lifecycle.infer(reset_request(episode_id=episode, request_id=request_id))


@pytest.mark.parametrize("history_blocks", [1, 2])
def test_real_wire_predicts_32_sends_16_and_uploads_every_executed_observation(monkeypatch, history_blocks):
    model_service = service(history_blocks=history_blocks, delay=0.01)
    sent, received = [], []
    with live_server(model_service) as uri, CausalPolicyClient(uri) as client:
        send, receive = client.connection.send, client.connection.recv

        def record_send(payload):
            sent.append(unpackb(payload))
            return send(payload)

        def record_receive(*args, **kwargs):
            payload = receive(*args, **kwargs)
            received.append(unpackb(payload))
            return payload

        monkeypatch.setattr(client.connection, "send", record_send)
        monkeypatch.setattr(client.connection, "recv", record_receive)
        session = client.create_session(
            measurement(model_service, include_size=True),
            prompt="Move the cup",
            execution_rate_callback=lambda rate: True,
            execution_policy=RTCExecutionPolicy(latency_factor=10),
        )
        assert client.metadata["simulation_sync"] is True
        assert client.metadata["actions_per_block"] == 32
        assert client.metadata["returned_action_steps"] == client.metadata["execute_steps"] == 16
        assert client.metadata["rtc_enabled"] is True
        assert session.prepare(**measurement(model_service))["tick"] == 0
        commands = []
        for tick in range(81):
            commands.append(session.get_action(**measurement(model_service, tick)))
            assert session.status()["tick"] == tick + 1
            assert not session.status()["inference_pending"]
        assert 0 < session.execution_hz <= 30
        assert session.status()["adaptive_execution"]
        assert [row["operation"] for row in sent[:4]] == ["reset", "calibrate", "calibrate", "calibrate"]
        assert [row["tick"] for row in sent] == [0, 0, 0, 0, 16, 32, 48, 64, 80]
        assert all(row["actions"].shape == (16, 55) for row in received)
        assert model_service.model.generated_counts == [32] * 9
        assert all(row["server_timing"]["infer_seconds"] >= 0.01 for row in received)
        for row in sent[4:]:
            tick = row["tick"]
            records = row["executed_observations"]
            assert len(records) == 16
            assert [record["tick"] for record in records] == list(range(tick - 15, tick + 1))
            for record in records:
                np.testing.assert_array_equal(record["image"], image(record["tick"]))
            np.testing.assert_array_equal(row["image"], image(tick))
            assert row["state"][0] == np.float32(tick / 100)
            assert row.get("rtc") is None
            assert 1 <= row["simulation_rtc"]["delay_steps"] <= 16
        published = [
            reply for sent_row, reply in zip(sent, received, strict=True) if sent_row["operation"] != "calibrate"
        ]
        expected = np.concatenate([row["actions"] for row in published], axis=0)[: len(commands)]
        np.testing.assert_array_equal(torch.stack(commands).numpy(), expected)
        for row, (batch, _) in zip(sent, model_service.model.calls, strict=True):
            tick = row["tick"]
            length = min(tick // 32, history_blocks) * 32
            sampled = list(range(tick - length, tick + 1, 4))
            assert batch["rtc_observation_ticks"].tolist() == sampled
            video = batch["video"][0]
            np.testing.assert_array_equal(video[0, : len(sampled), 0, 0].numpy(), np.asarray(sampled, np.uint8))
            assert not video[:, len(sampled) :].any()
            metadata = batch["sequence_plan"][0].causal_action_metadata
            assert metadata.anchors[-1, 0] == pytest.approx(tick / 100, abs=1e-6)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "too_short",
        "too_long",
        "duplicate",
        "reordered",
        "before_execution",
        "float_image",
        "geometry",
        "current_mismatch",
        "missing_guidance",
        "delay_zero",
        "delay_too_long",
        "delay_fractional",
        "delay_boolean",
        "wire_prefix",
    ],
)
def test_malformed_executed_batches_are_rejected_before_inference_and_require_reset(fault):
    model_service = service()
    lifecycle = PolicyConnection(model_service)
    reset_connection(lifecycle)
    initial_cache = copy.deepcopy(lifecycle.history_images)
    malformed = request()
    records = malformed["executed_observations"]
    if fault == "missing":
        malformed.pop("executed_observations")
    elif fault == "too_short":
        records.pop(0)
    elif fault == "too_long":
        records.insert(0, dict(tick=0, image=image(0)))
    elif fault == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif fault == "reordered":
        records[0], records[1] = records[1], records[0]
    elif fault == "before_execution":
        malformed["executed_observations"] = [dict(tick=t, image=image(t)) for t in range(16)]
    elif fault == "float_image":
        records[0]["image"] = records[0]["image"].astype(np.float32)
    elif fault == "geometry":
        records[0]["image"] = np.zeros((3, 8, 16), dtype=np.uint8)
    elif fault == "current_mismatch":
        records[-1]["image"] = image(17)
    elif fault == "missing_guidance":
        malformed.pop("simulation_rtc")
    elif fault == "delay_zero":
        malformed["simulation_rtc"]["delay_steps"] = 0
    elif fault == "delay_too_long":
        malformed["simulation_rtc"]["delay_steps"] = 17
    elif fault == "delay_fractional":
        malformed["simulation_rtc"]["delay_steps"] = 4.5
    elif fault == "delay_boolean":
        malformed["simulation_rtc"]["delay_steps"] = True
    elif fault == "wire_prefix":
        malformed["rtc"] = dict(previous_actions=np.zeros((16, 55), np.float32), delay_steps=4)
    with pytest.raises(ValueError):
        lifecycle.infer(malformed)
    assert lifecycle.needs_reset
    assert len(model_service.model.calls) == 1
    assert set(lifecycle.history_images) == set(initial_cache)
    for tick in initial_cache:
        np.testing.assert_array_equal(lifecycle.history_images[tick], initial_cache[tick])
    with pytest.raises(ValueError, match="Reset"):
        lifecycle.infer(request(request_id=2))
    reset_connection(lifecycle, episode="second", request_id=3)
    assert lifecycle.infer(request(episode="second", request_id=4))["actions"].shape == (16, 55)
    assert not lifecycle.needs_reset


def test_server_retains_only_required_sampled_history_and_reset_clears_old_episode():
    model_service = service(history_blocks=1)
    lifecycle = PolicyConnection(model_service)
    reset_connection(lifecycle)
    for count, tick in enumerate(range(16, 97, 16), start=1):
        response = lifecycle.infer(request(tick, request_id=count))
        assert response["tick"] == tick
        keys = set(lifecycle.history_images)
        assert len(keys) <= 9
        assert all(index % 4 == 0 and tick - 32 <= index <= tick for index in keys)
        assert all(index < tick for index in response["cached_history_ticks"])
        batch, _ = model_service.model.calls[-1]
        observed = [tick] if tick == 16 else list(range(tick - 32, tick + 1, 4))
        assert batch["rtc_observation_ticks"].tolist() == observed
    reset_connection(lifecycle, episode="new", request_id=7)
    assert set(lifecycle.history_images) <= {0}
    assert lifecycle.last_tick == 0
    lifecycle.infer(request(episode="new", request_id=8))
    assert model_service.model.calls[-1][0]["rtc_observation_ticks"].tolist() == [16]


@pytest.mark.parametrize("execute_steps", [8, 32])
def test_simulation_config_rejects_incompatible_execution_before_model_loading(execute_steps):
    with pytest.raises(ValueError, match="simulation|Simulation"):
        policy(simulation_sync=True, enable_rtc=False, execute_steps=execute_steps)
    required = ["--checkpoint", "missing", "--processor", "missing", "--vae", "missing", "--statistics", "missing"]
    args = parser().parse_args(required + ["--simulation-sync", "--execute-steps", str(execute_steps)])
    with pytest.raises(ValueError, match="simulation|Simulation"):
        load_policy(args)


def test_simulation_enables_guidance_implicitly_and_allows_explicit_rtc_opt_in():
    assert service().metadata["rtc_enabled"] is True
    assert policy(simulation_sync=True, enable_rtc=True).metadata["rtc_enabled"] is True


@pytest.mark.parametrize("flag", ["--simulation-sync", "--simulation_sync"])
def test_simulation_flag_is_server_opt_in_with_both_spellings(flag):
    required = ["--checkpoint", "w", "--processor", "p", "--vae", "v", "--statistics", "s"]
    assert parser().parse_args(required).simulation_sync is False
    assert parser().parse_args(required + [flag]).simulation_sync is True


class ActionResponseEnvironment:
    """A deterministic actuator whose measured state and pixels depend on actual commands."""

    def __init__(self):
        self.state = np.zeros(55, dtype=np.float32)
        self.tick = 0
        self.observations = {0: self.frame()}
        self.executed = []

    def frame(self):
        color = int(round(float(self.state[0]) * 1000)) % 255
        return np.full((3, 16, 16), color, dtype=np.uint8)

    def take_action(self, action):
        self.executed.append(np.asarray(action).copy())
        self.state = np.asarray(action, dtype=np.float32).copy()
        self.tick += 1
        self.observations[self.tick] = self.frame()


@pytest.mark.parametrize("delay_steps", [4, 16])
def test_wire_mixes_old_and_new_actions_and_observes_the_commands_actually_executed(delay_steps):
    model_service = service()
    env = ActionResponseEnvironment()
    with live_server(model_service) as uri, CausalPolicyClient(uri) as client:
        initial = measurement(model_service, include_size=True)
        bootstrap = client.reset(prompt="Move the cup", measurement=initial)
        # Independent expected full bootstrap; only the first 16 rows crossed the wire.
        expected_previous = np.arange(32, dtype=np.float32) / 1000
        for _ in range(3):
            calibration = client.infer(
                dict(
                    operation="calibrate",
                    tick=0,
                    revision=0,
                    state=initial["state"].numpy(),
                    image=initial["image"].numpy(),
                    simulation_rtc=dict(delay_steps=1, beta=5.0),
                    rtc=None,
                )
            )
            assert calibration["simulation_info"]["calibration"] is True
            assert env.tick == 0 and not env.executed
        chunk = bootstrap["actions"]
        for _ in range(3):
            for action in chunk:
                env.take_action(action)
            tick = env.tick
            measured_state = env.state.copy()
            observed = [dict(tick=t, image=env.observations[t]) for t in range(tick - 15, tick + 1)]
            current = dict(
                operation="predict",
                tick=tick,
                revision=0,
                state=measured_state,
                image=env.frame(),
                executed_observations=observed,
                simulation_rtc=dict(delay_steps=delay_steps, beta=5.0),
                rtc=None,
            )
            response = client.infer(current)
            info = response["simulation_info"]
            assert info == dict(
                delay_steps=delay_steps,
                old_action_steps=delay_steps,
                new_action_steps=16 - delay_steps,
                request_tick=tick,
                calibration=False,
            )
            batch, _ = model_service.model.calls[-1]
            rtc = batch["causal_action_rtc"]
            assert rtc.previous_actions.shape == (16, 55)
            np.testing.assert_allclose(rtc.previous_actions[:, 0].numpy(), expected_previous[16:], atol=1e-6, rtol=0)
            assert rtc.delay_steps == delay_steps
            chunk = response["actions"]
            np.testing.assert_array_equal(chunk[:delay_steps], rtc.previous_actions[:delay_steps].numpy())
            fresh = (
                measured_state[0] + (len(model_service.model.calls) - 1) / 10 + np.arange(32, dtype=np.float32) / 1000
            )
            np.testing.assert_allclose(chunk[delay_steps:, 0], fresh[delay_steps:16], atol=1e-6, rtol=0)
            expected_previous = fresh
            metadata = batch["sequence_plan"][0].causal_action_metadata
            np.testing.assert_array_equal(metadata.anchors[-1].numpy(), measured_state)
            sampled = batch["rtc_observation_ticks"].tolist()
            video = batch["video"][0]
            for index, observation_tick in enumerate(sampled):
                np.testing.assert_array_equal(video[:, index].numpy(), env.observations[observation_tick])
            assert env.tick == tick  # Model/RPC waiting never executes an extra actuator command.
        assert len(env.executed) == 48
        if delay_steps == 4:
            # The second executed batch includes both bootstrap-tail commands and
            # newly generated commands. Its RGB bytes follow that exact mixture.
            assert env.executed[16][0] == pytest.approx(0.016, abs=1e-6)
            assert env.executed[20][0] > 0.4
            assert int(env.observations[17][0, 0, 0]) == 16
            assert int(env.observations[21][0, 0, 0]) == int(round(float(env.executed[20][0]) * 1000)) % 255


def test_ordinary_server_rejects_calibration_but_retains_native_rtc_prefix_protocol():
    model_service = policy(enable_rtc=True)
    lifecycle = PolicyConnection(model_service)
    bootstrap = reset_connection(lifecycle)
    assert bootstrap["actions"].shape == (32, 55)
    with pytest.raises(ValueError, match="simulation_sync"):
        lifecycle.infer(dict(operation="calibrate", episode_id="first", request_id=1, tick=0, revision=0))
    prefix = bootstrap["actions"][16:].copy()
    prefix[:, 0] += np.arange(16, dtype=np.float32) / 100
    guided = request(request_id=2)
    guided.pop("executed_observations")
    guided.pop("simulation_rtc")
    guided["history"] = []
    guided["rtc"] = dict(previous_actions=prefix, delay_steps=4, beta=5.0)
    result = lifecycle.infer(guided)
    assert result["actions"].shape == (32, 55)
    np.testing.assert_array_equal(result["actions"][:4], prefix[:4])
    assert "simulation_info" not in result


def test_default_wire_retains_32_actions_and_original_sparse_observation_payload(monkeypatch):
    model_service = service(simulation_sync=False)
    sent, received = [], []
    with live_server(model_service) as uri, CausalPolicyClient(uri) as client:
        send, receive = client.connection.send, client.connection.recv

        def record_send(payload):
            sent.append(unpackb(payload))
            return send(payload)

        def record_receive(*args, **kwargs):
            payload = receive(*args, **kwargs)
            received.append(unpackb(payload))
            return payload

        monkeypatch.setattr(client.connection, "send", record_send)
        monkeypatch.setattr(client.connection, "recv", record_receive)
        session = client.create_session(measurement(model_service, include_size=True), prompt="Move the cup")
        for tick in range(49):
            session.get_action(**measurement(model_service, tick))
        assert not client.metadata.get("simulation_sync", False)
        assert [row["tick"] for row in sent] == [0, 16, 32, 48]
        assert all(row["actions"].shape == (32, 55) for row in received)
        assert all("executed_observations" not in row for row in sent)
        assert [record["tick"] for record in sent[-1]["history"]] == list(range(16, 48, 4))
        assert model_service.model.generated_counts == [32] * 4
