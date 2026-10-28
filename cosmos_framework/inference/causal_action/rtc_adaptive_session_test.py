# SPDX-License-Identifier: OpenMDW-1.1
"""Online paired timing buffers and client-acknowledged RTC execution rates."""

import math

import pytest

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.inference.causal_action import rtc_session
from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
from cosmos_framework.inference.causal_action.rtc_session import RTCDeadlineMiss, RTCFallbackWarning, RTCSession
from cosmos_framework.inference.causal_action.rtc_session_test import (
    GatePlanner,
    _frequency_adapter,
    actions,
    consume,
    inputs,
)


def calibration(monkeypatch, records, *, warmups=(), run=False):
    """Supply measured wall times without waiting for a gated runtime planner."""
    remaining, calls = iter(records), []

    def measure(operation, **options):
        calls.append(options)
        if run:
            operation()
        samples = list(next(remaining))
        maximum = max(samples)
        return dict(
            controller_hz=options["controller_hz"],
            warmup_seconds=list(warmups),
            sample_seconds=samples,
            max_seconds=maximum,
            margin_steps=options["margin_steps"],
            delay_steps=max(1, math.ceil(maximum * options["controller_hz"]) + options["margin_steps"]),
        )

    monkeypatch.setattr(rtc_session, "measure_rtc_latency", measure)
    return calls


def policy(**options):
    return RTCExecutionPolicy(latency_factor=1.0, **options)


def adaptive_session(planner, callback, **options):
    return RTCSession(
        planner,
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=8.0,
        execution_hz=3.0,
        min_execution_steps=options.pop("min_execution_steps", 4),
        initial_delay_steps=options.pop("initial_delay_steps", 2),
        execution_policy=options.pop("execution_policy", policy()),
        execution_rate_callback=callback,
        **options,
    )


def complete(controller, planner, index, clock, seconds):
    assert planner.started[index].wait(2)
    clock[0] += seconds
    planner.release[index].set()
    return controller.join(2)


def test_adaptive_execution_requires_an_explicit_client_acknowledgement_handler():
    with pytest.raises(ValueError, match="execution_rate_callback"):
        RTCSession(lambda **request: actions(), actions(), template=ActionStateTemplate55(), execution_policy=policy())


def test_startup_records_individual_measurements_and_excludes_warmup(monkeypatch):
    calibration(monkeypatch, [(0.1, 2.0)], warmups=(1000.0,))
    acknowledged = []

    def pace(rate):
        acknowledged.append(rate)
        return True

    with RTCSession(
        lambda **request: actions(32, start=100),
        actions(32),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=30.0,
        min_execution_steps=16,
        execution_policy=policy(),
        execution_rate_callback=pace,
    ) as controller:
        controller.prepare(**inputs())
        status = controller.status()
        assert status["rtc_enabled"] and status["prepared"]
        assert status["tick"] == status["published_requests"] == 0
        assert acknowledged == [7.5]
        assert status["controller_hz"] == 30.0
        assert status["delay_history_seconds"] == [0.1, 2.0]
        assert status["latency_samples"] == [
            dict(history_seconds=0.0, elapsed_seconds=0.1),
            dict(history_seconds=0.0, elapsed_seconds=2.0),
        ]
        assert status["execution_advice"]["sample_count"] == 2
        assert status["execution_advice"]["distinct_history_count"] == 1
        assert status["execution_advice"]["slope_seconds_per_history_second"] == 0
        assert status["estimated_delay_steps"] == 16


def test_manual_prefix_budget_cannot_skip_wall_time_samples_for_online_fitting(monkeypatch):
    measured = calibration(monkeypatch, [(0.5,)])
    with adaptive_session(lambda **request: actions(start=100), lambda rate: True, initial_delay_steps=2) as controller:
        assert not controller.status()["prepared"]
        controller.prepare(**inputs())
        assert len(measured) == 1
        assert controller.status()["latency_samples"] == [dict(history_seconds=0.0, elapsed_seconds=0.5)]


@pytest.mark.parametrize("acknowledgement", [False, None, 1])
def test_declining_or_omitting_ack_does_not_install_server_only_slow_rate(monkeypatch, acknowledgement):
    calibration(monkeypatch, [(2.0,)])
    with RTCSession(
        lambda **request: actions(32, start=100),
        actions(32),
        template=ActionStateTemplate55(),
        min_execution_steps=16,
        execution_policy=policy(),
        execution_rate_callback=lambda rate: acknowledgement,
        ordinary_planner=lambda **request: actions(32, start=200),
    ) as controller:
        with pytest.warns(RTCFallbackWarning, match="did not acknowledge"):
            controller.prepare(**inputs())
        status = controller.status()
        assert not status["rtc_enabled"] and status["execution_hz"] == 30.0
        assert status["execution_rate_changes"] == 0


def test_growing_history_uses_new_online_samples_to_predict_next_request(monkeypatch):
    calibration(monkeypatch, [(0.5,)])
    clock = [10.0]
    monkeypatch.setattr(rtc_session, "perf_counter", lambda: clock[0])
    planner, acknowledged = GatePlanner(requests=3), []

    def pace(rate):
        acknowledged.append(rate)
        return True

    controller = adaptive_session(planner, pace)
    try:
        consume(controller, 0, 5)
        assert planner.started[0].wait(2)
        first = planner.calls[0]
        assert first["rtc"].delay_steps == 3
        complete(controller, planner, 0, clock, 0.5)
        consume(controller, 5, 9)
        complete(controller, planner, 1, clock, 1.5)
        controller.get_action(**inputs(9))
        assert acknowledged == [pytest.approx(2.0)]
        consume(controller, 10, 13)
        complete(controller, planner, 2, clock, 1.5)
        controller.get_action(**inputs(13))
        status = controller.status()
        assert status["execution_advice"]["history_seconds"] == 2.0
        assert status["execution_advice"]["predicted_seconds"] == pytest.approx(2.5)
        assert status["execution_advice"]["slope_seconds_per_history_second"] == pytest.approx(1.0)
        assert status["execution_hz"] == pytest.approx(1.2)
        assert len(acknowledged) == 2 and first["rtc"].delay_steps == 3
        assert status["controller_hz"] == 8.0
    finally:
        planner.unblock()
        controller.close()


def test_runtime_history_tag_belongs_to_submission_not_publication(monkeypatch):
    calibration(monkeypatch, [(1.0,)])
    clock = [10.0]
    monkeypatch.setattr(rtc_session, "perf_counter", lambda: clock[0])
    planner = GatePlanner()
    controller = adaptive_session(planner, lambda rate: True)
    try:
        consume(controller, 0, 8)
        status = complete(controller, planner, 0, clock, 1.0)
        assert status["tick"] == 8 and status["last_request"]["observation_tick"] == 4
        assert status["last_request"]["history_seconds"] == 0.0
        assert status["latency_samples"][-1] == dict(history_seconds=0.0, elapsed_seconds=1.0)
    finally:
        planner.unblock()
        controller.close()


def test_paired_history_and_latency_buffers_evict_together(monkeypatch):
    calibration(monkeypatch, [(0.5, 0.6)])
    clock = [10.0]
    monkeypatch.setattr(rtc_session, "perf_counter", lambda: clock[0])
    planner = GatePlanner(requests=2)
    controller = adaptive_session(planner, lambda rate: True, delay_history_size=3)
    try:
        consume(controller, 0, 5)
        complete(controller, planner, 0, clock, 0.7)
        consume(controller, 5, 9)
        status = complete(controller, planner, 1, clock, 0.8)
        assert len(status["latency_samples"]) == len(status["delay_history_seconds"]) == 3
        assert [row["history_seconds"] for row in status["latency_samples"]] == [0.0, 0.0, 1.0]
        assert status["delay_history_seconds"] == pytest.approx([0.6, 0.7, 0.8])
        assert [row["elapsed_seconds"] for row in status["latency_samples"]] == status["delay_history_seconds"]
    finally:
        planner.unblock()
        controller.close()


def test_ack_timeout_falls_back_without_consuming_tick_or_leaving_active_startup(monkeypatch):
    calibration(monkeypatch, [(2.0,)])

    def timeout(rate):
        raise TimeoutError("client pacing acknowledgement timed out")

    with adaptive_session(
        lambda **request: actions(start=100),
        timeout,
        ordinary_planner=lambda **request: actions(start=200),
    ) as controller:
        with pytest.warns(RTCFallbackWarning, match="acknowledgement failed"):
            controller.prepare(**inputs())
        status = controller.status()
        assert status["prepared"] and not status["preparing"]
        assert not status["inference_pending"] and not status["rtc_enabled"]
        assert status["tick"] == status["execution_rate_changes"] == 0


def test_unacknowledged_rate_without_ordinary_fallback_latches_until_reset(monkeypatch):
    calibration(monkeypatch, [(2.0,), (2.0,)])
    acknowledged = [False]
    with adaptive_session(lambda **request: actions(start=100), lambda rate: acknowledged[-1]) as controller:
        with pytest.raises(RTCDeadlineMiss, match="did not acknowledge"):
            controller.get_action(**inputs())
        acknowledged.append(True)
        with pytest.raises(RTCDeadlineMiss, match="did not acknowledge"):
            controller.get_action(**inputs())
        controller.reset(actions(start=50))
        assert controller.get_action(**inputs())[0].item() == 50
        assert controller.status()["error"] is None


def test_no_rate_or_prefix_changes_while_request_is_in_flight(monkeypatch):
    calibration(monkeypatch, [(0.5,)])
    planner, acknowledged = GatePlanner(), []

    def pace(rate):
        acknowledged.append(rate)
        return True

    controller = adaptive_session(planner, pace)
    try:
        consume(controller, 0, 7)
        assert planner.started[0].wait(2)
        assert controller.status()["inference_pending"]
        assert controller.execution_hz == 3.0 and not acknowledged
        assert planner.calls[0]["rtc"].delay_steps == 3
    finally:
        planner.unblock()
        controller.close()


def test_client_that_acknowledges_but_consumes_too_fast_still_misses_deadline(monkeypatch):
    calibration(monkeypatch, [(1.0,)])
    planner, ordinary = GatePlanner(), []

    def fallback(**request):
        ordinary.append(request)
        return actions(start=200)

    controller = adaptive_session(planner, lambda rate: True, min_execution_steps=3, ordinary_planner=fallback)
    try:
        consume(controller, 0, 7)
        assert planner.started[0].wait(2)
        assert controller.execution_hz == pytest.approx(2.0)
        assert planner.calls[0]["rtc"].delay_steps == 3
        with pytest.warns(RTCFallbackWarning, match="consumed 4 actions"):
            planner.release[0].set()
            status = controller.join(2)
        assert not status["rtc_enabled"] and status["published_requests"] == 0
        assert controller.get_action(**inputs(7))[0].item() == 200 and ordinary[0]["rtc"] is None
    finally:
        planner.unblock()
        controller.close()


def test_user_minimum_rate_causes_fallback_instead_of_slowing_below_limit(monkeypatch):
    calibration(monkeypatch, [(2.0,)])
    with adaptive_session(
        lambda **request: actions(start=100),
        lambda rate: True,
        execution_policy=policy(minimum_execution_hz=2.0),
        ordinary_planner=lambda **request: actions(start=200),
    ) as controller:
        with pytest.warns(RTCFallbackWarning, match="minimum_execution_hz"):
            controller.prepare(**inputs())
        assert not controller.status()["rtc_enabled"] and controller.execution_hz == 3.0


def test_runtime_wall_seconds_drive_budget_instead_of_consumed_tick_counts(monkeypatch):
    calibration(monkeypatch, [(0.5,)])
    clock = [10.0]
    monkeypatch.setattr(rtc_session, "perf_counter", lambda: clock[0])
    planner = GatePlanner()
    controller = adaptive_session(planner, lambda rate: True)
    try:
        consume(controller, 0, 5)
        status = complete(controller, planner, 0, clock, 2.0)
        assert status["last_request"]["actual_delay_steps"] == 1
        controller.get_action(**inputs(5))
        status = controller.status()
        assert status["execution_hz"] == pytest.approx(1.5)
        assert status["execution_advice"]["budget_seconds"] == 2.0
        assert status["estimated_delay_steps"] == 4
    finally:
        planner.unblock()
        controller.close()


def test_reset_clears_online_samples_and_recalibrates_without_silent_acceleration(monkeypatch):
    measured = calibration(monkeypatch, [(2.0,), (0.5,)])
    with adaptive_session(lambda **request: actions(start=100), lambda rate: True) as controller:
        controller.get_action(**inputs())
        slow_rate = controller.execution_hz
        assert slow_rate < 3.0
        controller.reset(actions(start=50))
        status = controller.status()
        assert status["execution_hz"] == slow_rate
        assert status["delay_history_seconds"] == status["latency_samples"] == []
        assert not status["prepared"] and status["execution_advice"] is None
        controller.get_action(**inputs())
        assert len(measured) == 2 and controller.execution_hz == slow_rate
        assert controller.status()["latency_samples"] == [dict(history_seconds=0.0, elapsed_seconds=0.5)]


def test_model_adapter_keeps_online_pacing_out_of_model_and_sampling_arguments(monkeypatch):
    calibration(monkeypatch, [(0.1,)], run=True)
    model, builder, calls = _frequency_adapter(30.0)
    builder.controller_hz = 30.0
    with RTCSession.for_model(
        model,
        builder,
        actions(),
        template=ActionStateTemplate55(),
        controller_hz=30.0,
        execution_hz=3.0,
        min_execution_steps=3,
        execution_policy=policy(),
        execution_rate_callback=lambda rate: True,
    ) as controller:
        controller.prepare(**inputs())
        assert controller.status()["rtc_enabled"]
        batch, sampling = calls[0]
        assert batch["conditioning_fps_action"] == 30.0 and batch["conditioning_fps"] == 7.5
        assert sampling["num_steps"] == 10 and sampling["guidance"] == 1.0
        assert "execution_policy" not in sampling and "execution_rate_callback" not in sampling
