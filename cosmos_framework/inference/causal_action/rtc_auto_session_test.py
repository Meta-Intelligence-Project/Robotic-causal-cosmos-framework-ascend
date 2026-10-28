# SPDX-License-Identifier: OpenMDW-1.1
"""Startup calibration and an actual ordinary-inference fallback, without hardware."""

import warnings
from threading import Event, Thread

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.inference.causal_action import rtc_session
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.rtc_session import RTCSession
from cosmos_framework.inference.causal_action.rtc_session_test import _frequency_adapter, actions, consume, inputs


def _calibration(monkeypatch, delays, *, run=True):
    """Keep the wall-clock measurement independent of paused controller ticks."""
    records = []
    values = iter(delays)

    def measure(operation, **options):
        records.append(options)
        if run:
            operation()
        delay = next(values)
        seconds = max(0, delay - options["margin_steps"]) / options["controller_hz"]
        return {
            "controller_hz": options["controller_hz"],
            "warmup_seconds": [],
            "sample_seconds": [seconds],
            "max_seconds": seconds,
            "margin_steps": options["margin_steps"],
            "delay_steps": delay,
        }

    monkeypatch.setattr(rtc_session, "measure_rtc_latency", measure)
    return records


def _session(planner, ordinary, **options):
    return RTCSession(
        planner,
        actions(),
        template=ActionStateTemplate55(),
        ordinary_planner=ordinary,
        initial_delay_steps=None,
        min_execution_steps=3,
        **options,
    )


def test_prepare_calibrates_without_consuming_or_publishing_bootstrap(monkeypatch):
    measurements = _calibration(monkeypatch, [2])
    calls, ordinary_calls = [], []

    def planner(**request):
        calls.append(request)
        return actions(start=100)

    def ordinary(**request):
        ordinary_calls.append(request)
        return actions(start=200)

    with _session(planner, ordinary) as controller:
        controller.prepare(**inputs())
        status = controller.status()
        assert status["rtc_enabled"] is True
        assert status["execution_mode"] == "rtc"
        assert status["tick"] == status["chunk_start_tick"] == 0
        assert status["published_requests"] == 0
        assert status["estimated_delay_steps"] == 2
        assert status["remaining_actions"] == 8
        assert len(measurements) == len(calls) == 1 and not ordinary_calls
        request = calls[0]
        assert request["tick"] == 0 and not request["history"]
        prefix = request["rtc"]
        assert isinstance(prefix, RTCPrefix)
        torch.testing.assert_close(prefix.previous_actions, actions()[: len(prefix.previous_actions)])
        assert consume(controller, 0, 2) == [0, 1]
        controller.prepare(**inputs(2))
        assert len(measurements) == 1


def test_first_get_action_lazily_calibrates_once(monkeypatch):
    measured = _calibration(monkeypatch, [2])
    with _session(lambda **request: actions(start=100), lambda **request: actions(start=200)) as controller:
        assert controller.status()["tick"] == 0
        assert consume(controller, 0, 3) == [0, 1, 2]
        assert len(measured) == 1
        assert controller.status()["tick"] == 3


def test_explicit_delay_keeps_manual_path_without_calibration(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An explicit delay must not unexpectedly run startup calibration")

    monkeypatch.setattr(rtc_session, "measure_rtc_latency", unexpected)
    with RTCSession(
        lambda **request: actions(start=100),
        actions(),
        template=ActionStateTemplate55(),
        ordinary_planner=lambda **request: actions(start=200),
        initial_delay_steps=2,
        min_execution_steps=3,
    ) as controller:
        controller.prepare(**inputs())
        assert consume(controller, 0, 3) == [0, 1, 2]
        assert controller.status()["estimated_delay_steps"] == 2


def test_infeasible_measured_delay_warns_and_replans_ordinary_at_fresh_chunk_boundary(monkeypatch):
    measured = _calibration(monkeypatch, [4])  # H=8,s=3 only permits d<=3; no controller ticks elapsed.
    guided, ordinary_calls = [], []

    def planner(**request):
        guided.append(request)
        return actions(start=900)

    def ordinary(**request):
        ordinary_calls.append(request)
        return actions(start=100 * len(ordinary_calls))

    with _session(planner, ordinary) as controller:
        with pytest.warns(rtc_session.RTCFallbackWarning):
            controller.prepare(**inputs())
        status = controller.status()
        assert status["rtc_enabled"] is False
        assert status["execution_mode"] == "blocking_non_rtc"
        assert status["tick"] == status["published_requests"] == 0
        assert len(measured) == len(guided) == 1 and not ordinary_calls
        assert consume(controller, 0, 8) == list(range(8))
        assert not ordinary_calls
        assert controller.get_action(**inputs(8))[0].item() == 100
        request = ordinary_calls[0]
        assert request["rtc"] is None
        assert request["tick"] == 8
        assert request["state"][0].item() == 8
        assert request["image"].eq(8).all()
        assert [record.tick for record in request["history"]] == list(range(8))
        assert consume(controller, 9, 16) == list(range(101, 108))
        assert len(guided) == 1
        assert controller.get_action(**inputs(16))[0].item() == 200
        assert ordinary_calls[1]["tick"] == 16


class _ActiveGate:
    def __init__(self):
        self.started, self.release, self.active = Event(), Event(), Event()

    def __call__(self, **request):
        self.active.set()
        self.started.set()
        try:
            if not self.release.wait(10):
                raise TimeoutError("Test did not release the guided planner")
            return actions(start=900)
        finally:
            self.active.clear()


def test_overdue_guided_result_is_discarded_before_ordinary_fresh_replan():
    guided, ordinary_calls = _ActiveGate(), []

    def ordinary(**request):
        assert not guided.active.is_set(), "Both planners used the same model concurrently"
        ordinary_calls.append(request)
        return actions(start=100)

    controller = RTCSession(
        guided,
        actions(),
        template=ActionStateTemplate55(),
        ordinary_planner=ordinary,
        min_execution_steps=3,
        initial_delay_steps=2,
    )
    try:
        assert consume(controller, 0, 6) == list(range(6))
        assert guided.started.wait(2)
        with pytest.warns(rtc_session.RTCFallbackWarning):
            guided.release.set()
            status = controller.join(2)
        assert status["rtc_enabled"] is False
        assert status["execution_mode"] == "blocking_non_rtc"
        assert status["published_requests"] == 0
        assert status["last_request"]["accepted"] is False
        assert status["error"] is None
        # Do not install 900-valued guided output or replay its elapsed prefix.
        assert controller.get_action(**inputs(6))[0].item() == 100
        assert ordinary_calls[0]["rtc"] is None
        assert ordinary_calls[0]["state"][0].item() == ordinary_calls[0]["tick"] == 6
    finally:
        guided.release.set()
        controller.close()


def test_exhaustion_fallback_joins_guided_worker_before_blocking_ordinary_replan():
    guided, ordinary_calls = _ActiveGate(), []
    boundary_entered, ordinary_started = Event(), Event()
    results, failures = [], []

    def ordinary(**request):
        assert not guided.active.is_set(), "Ordinary inference raced the still-running guided request"
        ordinary_started.set()
        ordinary_calls.append(request)
        return actions(start=100)

    controller = RTCSession(
        guided,
        actions(),
        template=ActionStateTemplate55(),
        ordinary_planner=ordinary,
        min_execution_steps=3,
        initial_delay_steps=2,
    )

    def request_boundary():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", rtc_session.RTCFallbackWarning)
                boundary_entered.set()
                results.append(controller.get_action(**inputs(8)))
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=request_boundary)
    try:
        assert consume(controller, 0, 8) == list(range(8))
        assert guided.started.wait(2)
        thread.start()
        assert boundary_entered.wait(2)
        assert not ordinary_started.wait(0.1)
        guided.release.set()
        thread.join(3)
        assert not thread.is_alive() and not failures
        assert len(results) == 1 and results[0][0].item() == 100
        assert ordinary_calls[0]["tick"] == 8 and ordinary_calls[0]["rtc"] is None
        assert controller.status()["rtc_enabled"] is False
    finally:
        guided.release.set()
        if thread.is_alive():
            thread.join(3)
        controller.close()


def test_reset_recalibrates_a_new_episode_without_reusing_old_fallback(monkeypatch):
    measured = _calibration(monkeypatch, [4, 2])
    with _session(lambda **request: actions(start=900), lambda **request: actions(start=100)) as controller:
        with pytest.warns(rtc_session.RTCFallbackWarning):
            controller.prepare(**inputs())
        assert consume(controller, 0, 2) == [0, 1]
        controller.reset(actions(start=50))
        controller.prepare(**inputs())
        assert len(measured) == 2
        status = controller.status()
        assert status["tick"] == status["chunk_start_tick"] == status["published_requests"] == 0
        assert status["rtc_enabled"] is True and status["estimated_delay_steps"] == 2
        assert consume(controller, 0, 2) == [50, 51]


def test_reset_recovers_a_failed_startup_future_and_remeasures_fresh_bootstrap(monkeypatch):
    measured = _calibration(monkeypatch, [2])
    requests = []

    def planner(**request):
        requests.append(request)
        if len(requests) == 1:
            raise ValueError("startup encoding failed")
        return actions(start=900)

    with _session(planner, lambda **request: actions(start=100)) as controller:
        with pytest.raises(ValueError, match="startup encoding failed"):
            controller.prepare(**inputs())
        controller.reset(actions(start=50))
        controller.prepare(**inputs())
        assert len(measured) == len(requests) == 2
        status = controller.status()
        assert status["revision"] == 1
        assert status["tick"] == status["chunk_start_tick"] == 0
        assert status["error"] is None
        assert status["rtc_enabled"] is True and status["estimated_delay_steps"] == 2
        assert requests[-1]["revision"] == 1 and requests[-1]["tick"] == 0
        assert requests[-1]["rtc"].previous_actions[0, 0].item() == 50
        assert consume(controller, 0, 2) == [50, 51]


def test_reset_recovers_failed_ordinary_future_without_replaying_the_old_episode(monkeypatch):
    measured = _calibration(monkeypatch, [4, 4])
    ordinary_calls = []

    def ordinary(**request):
        ordinary_calls.append(request)
        assert request["rtc"] is None
        if len(ordinary_calls) == 1:
            raise RuntimeError("ordinary episode failed")
        return actions(start=100)

    with _session(lambda **request: actions(start=900), ordinary) as controller:
        with pytest.warns(rtc_session.RTCFallbackWarning):
            controller.prepare(**inputs())
        assert consume(controller, 0, 8) == list(range(8))
        with pytest.raises(RuntimeError, match="ordinary episode failed"):
            controller.get_action(**inputs(8))
        controller.reset(actions(start=50))
        with pytest.warns(rtc_session.RTCFallbackWarning):
            controller.prepare(**inputs())
        assert len(measured) == 2
        status = controller.status()
        assert status["revision"] == 1 and status["error"] is None
        assert status["tick"] == status["chunk_start_tick"] == 0
        assert consume(controller, 0, 8) == list(range(50, 58))
        assert controller.get_action(**inputs(8))[0].item() == 100
        assert len(ordinary_calls) == 2
        assert ordinary_calls[-1]["revision"] == 1
        assert ordinary_calls[-1]["tick"] == ordinary_calls[-1]["state"][0].item() == 8


def test_invalid_observation_is_not_hidden_by_calibration_fallback(monkeypatch):
    measured = _calibration(monkeypatch, [4])
    with _session(lambda **request: actions(start=900), lambda **request: actions(start=100)) as controller:
        observation = inputs()
        observation["state"] = torch.zeros(2)
        with pytest.raises(ValueError, match="state"):
            controller.prepare(**observation)
        assert not measured


def test_model_adapter_defaults_to_auto_and_clears_prefix_for_real_ordinary_call(monkeypatch):
    measured = _calibration(monkeypatch, [4])
    model, build, calls = _frequency_adapter(30.0)

    def builder(**request):
        batch = build(**request)
        # A reused/custom builder must not smuggle a stale prefix into ordinary B.
        batch["causal_action_rtc"] = RTCPrefix(actions(4), 1)
        return batch

    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3
    ) as controller:
        with pytest.warns(rtc_session.RTCFallbackWarning):
            controller.prepare(**inputs())
        assert len(measured) == len(calls) == 1
        assert isinstance(calls[0][0]["causal_action_rtc"], RTCPrefix)
        assert consume(controller, 0, 8) == list(range(8))
        command = controller.get_action(**inputs(8))
        assert len(calls) == 2
        batch, options = calls[-1]
        assert "causal_action_rtc" not in batch
        assert options == {
            "guidance": 1.0,
            "num_steps": 10,
            "causal_block_size": 2,
            "causal_history_blocks": 8,
        }
        assert batch["sequence_plan"][0].causal_action_metadata.anchors[-1, 0].item() == 8
        assert command[0].item() == 8


def test_model_frequency_error_during_calibration_does_not_fall_back(monkeypatch):
    _calibration(monkeypatch, [4])
    model, builder, calls = _frequency_adapter(25.0)
    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3
    ) as controller:
        with warnings.catch_warnings(record=True) as recorded:
            with pytest.raises((ValueError, rtc_session.RTCPlannerError), match="conditioning_fps_action"):
                controller.prepare(**inputs())
        assert not calls
        assert not any(issubclass(row.category, rtc_session.RTCFallbackWarning) for row in recorded)
