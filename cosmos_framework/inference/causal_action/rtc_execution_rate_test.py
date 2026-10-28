# SPDX-License-Identifier: OpenMDW-1.1
"""Client waypoint speed changes wall-time budgets, not the model's action clock."""

from contextlib import nullcontext

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.inference.causal_action import rtc_session
from cosmos_framework.inference.causal_action.rtc_calibration import measure_rtc_latency
from cosmos_framework.inference.causal_action.rtc_session import RTCFallbackWarning, RTCSession
from cosmos_framework.inference.causal_action.rtc_session_test import (
    GatePlanner,
    _frequency_adapter,
    actions,
    consume,
    inputs,
)


def _clocked_measurement(monkeypatch, clock):
    def measure(operation, **options):
        return measure_rtc_latency(operation, clock=lambda: clock[0], **options)

    monkeypatch.setattr(rtc_session, "measure_rtc_latency", measure)


@pytest.mark.parametrize(
    "execution_hz,expected_steps,enabled",
    [(None, 61, False), (3.0, 7, True), (7.5, 16, True), (8.0, 17, False)],
)
def test_same_two_second_request_fits_only_when_client_consumes_waypoints_slowly(
    monkeypatch, execution_hz, expected_steps, enabled
):
    clock, calls = [0.0], []
    _clocked_measurement(monkeypatch, clock)

    def planner(**request):
        calls.append(request)
        clock[0] += 2.0
        return actions(32, start=100)

    with RTCSession(
        planner,
        actions(32),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=execution_hz,
        min_execution_steps=16,
        initial_delay_steps=None,
        ordinary_planner=lambda **request: actions(32, start=200),
        calibration_warmups=0,
        calibration_samples=1,
    ) as controller:
        warning = nullcontext() if enabled else pytest.warns(RTCFallbackWarning)
        with warning:
            controller.prepare(**inputs())
        status = controller.status()
        actual_hz = 30.0 if execution_hz is None else execution_hz
        assert status["controller_hz"] == 30.0
        assert status["execution_hz"] == actual_hz
        assert status["playback_speed"] == actual_hz / 30
        assert status["estimated_delay_steps"] == expected_steps
        assert status["rtc_enabled"] is enabled
        assert status["calibration"]["max_seconds"] == 2.0
        assert status["calibration"]["controller_hz"] == 30.0
        assert status["calibration"]["execution_hz"] == actual_hz
        assert status["calibration"]["playback_speed"] == actual_hz / 30
        assert status["tick"] == status["published_requests"] == 0
        assert len(calls) == 1


def test_unspecified_execution_rate_preserves_existing_model_rate():
    with RTCSession(
        lambda **request: actions(),
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=25.0,
        min_execution_steps=3,
        initial_delay_steps=2,
    ) as controller:
        status = controller.status()
        assert status["controller_hz"] == status["execution_hz"] == 25.0
        assert status["playback_speed"] == 1.0
        assert status["estimated_delay_steps"] == 2


@pytest.mark.parametrize("rate", [0, -1, True, False, float("nan"), float("inf"), [], torch.ones(2)])
def test_invalid_execution_rate_is_rejected_before_worker_creation(rate):
    with pytest.raises(ValueError, match="execution_hz"):
        RTCSession(
            lambda **request: actions(),
            actions(),
            template=ActionStateTemplate55(),
            execution_hz=rate,
        )


def test_model_adapter_keeps_thirty_hz_inputs_and_sampling_defaults_at_three_hz_execution(monkeypatch):
    clock = [0.0]
    _clocked_measurement(monkeypatch, clock)
    model, builder, calls = _frequency_adapter(30.0)
    builder.controller_hz = 30.0
    original_generate = model.generate_samples_from_batch

    def generate(batch, **options):
        clock[0] += 0.2
        return original_generate(batch, **options)

    model.generate_samples_from_batch = generate
    with RTCSession.for_model(
        model,
        builder,
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=3.0,
        min_execution_steps=3,
        calibration_warmups=0,
        calibration_samples=1,
    ) as controller:
        controller.prepare(**inputs())
        status = controller.status()
        assert status["rtc_enabled"]
        assert status["estimated_delay_steps"] == 2
        assert status["execution_hz"] == 3.0 and status["controller_hz"] == 30.0
        assert len(calls) == 1
        batch, options = calls[0]
        assert batch["conditioning_fps_action"] == 30.0
        assert batch["conditioning_fps"] == 7.5
        assert options["num_steps"] == 10 and options["guidance"] == 1.0
        assert "execution_hz" not in options


def test_slow_execution_does_not_permit_relabelling_model_fps(monkeypatch):
    _clocked_measurement(monkeypatch, [0.0])
    model, builder, calls = _frequency_adapter(3.0)
    with RTCSession.for_model(
        model,
        builder,
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=3.0,
        min_execution_steps=3,
        calibration_warmups=0,
        calibration_samples=1,
    ) as controller:
        with pytest.raises(ValueError, match="conditioning_fps_action must match"):
            controller.prepare(**inputs())
        assert not calls


def test_runtime_delay_still_counts_consumed_commands_at_slow_execution_rate():
    planner = GatePlanner()
    with RTCSession(
        planner,
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=3.0,
        min_execution_steps=3,
        initial_delay_steps=2,
    ) as controller:
        try:
            assert consume(controller, 0, 4) == [0, 1, 2, 3]
            assert planner.started[0].wait(2)
            planner.release[0].set()
            status = controller.join(2)
            assert status["last_request"]["actual_delay_steps"] == 1
            assert status["chunk_start_tick"] == 3
            assert planner.calls[0]["rtc"].delay_steps == 2
        finally:
            planner.unblock()
