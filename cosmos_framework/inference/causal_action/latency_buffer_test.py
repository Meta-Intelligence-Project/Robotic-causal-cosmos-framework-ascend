# SPDX-License-Identifier: OpenMDW-1.1
"""Completed RPC timings drive the same bounded buffer with or without RTC."""

from threading import Event

import pytest

from cosmos_framework.inference.causal_action.async_session import AsyncRemoteSession
from cosmos_framework.inference.causal_action.async_session_test import _actions, _Client, _measurement
from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
from cosmos_framework.inference.causal_action.unified_async_test import _CalibratedClient, _Clock


def _session(*, enable_rtc=False, latency_buffer_size=32, client=None, clock=None):
    client = _Client() if client is None else client
    client.metadata["rtc_enabled"] = True
    clock = _Clock() if clock is None else clock
    session = AsyncRemoteSession(
        client,
        _actions(),
        enable_rtc=enable_rtc,
        latency_buffer_size=latency_buffer_size,
        clock=clock,
    )
    return session, client, clock


def _submit_next(session, client, index):
    anchor = (index + 1) * 16
    for tick in range(session.status()["tick"], anchor + 1):
        session.get_action(**_measurement(tick))
    assert client.started[index].wait(2)
    assert client.calls[index]["tick"] == anchor


def _complete(session, client, clock, index, seconds):
    clock.advance(seconds)
    client.release[index].set()
    return session.join(2)


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_completed_requests_are_measured_without_an_execution_policy(enable_rtc):
    session, client, clock = _session(enable_rtc=enable_rtc)
    try:
        initial = session.status()
        assert initial["latency_buffer_capacity"] == 32
        assert initial["latency_history_basis"] == "encoded_window_model_seconds"
        assert initial["latency_samples"] == []
        assert initial["latency_fit"]["sample_count"] == 0
        assert initial["latency_fit"]["fit_mode"] == "unavailable"
        _submit_next(session, client, 0)
        assert session.status()["latency_samples"] == []
        assert (client.calls[0]["rtc"] is not None) is enable_rtc
        status = _complete(session, client, clock, 0, 2.0)
        assert not status["adaptive_execution"] and status["execution_advice"] is None
        assert status["latency_samples"] == [dict(history_seconds=1024 / 30, elapsed_seconds=2.0)]
        fit = status["latency_fit"]
        assert fit["sample_count"] == fit["distinct_history_count"] == 1
        assert fit["fit_mode"] == "constant" and fit["predicted_seconds"] == 2.0
        assert fit["slope_seconds_per_history_second"] == 0
        assert status["controller_hz"] == status["execution_hz"] == 30
        assert status["handoff_steps"] == 24 and not status["published_requests"]
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_only_the_recent_completed_requests_remain_in_a_bounded_buffer(enable_rtc):
    session, client, clock = _session(enable_rtc=enable_rtc, latency_buffer_size=2)
    try:
        for index, seconds in enumerate((4.0, 2.0, 3.0)):
            _submit_next(session, client, index)
            status = _complete(session, client, clock, index, seconds)
            assert len(status["latency_samples"]) <= 2
        assert status["latency_buffer_capacity"] == 2
        assert [sample["elapsed_seconds"] for sample in status["latency_samples"]] == [2.0, 3.0]
        assert all(sample["history_seconds"] == 1024 / 30 for sample in status["latency_samples"])
        assert status["latency_fit"]["sample_count"] == 2
        assert status["latency_fit"]["distinct_history_count"] == 1
        assert status["latency_fit"]["predicted_seconds"] == 3.0
        assert [call["tick"] for call in client.calls] == [16, 32, 48]
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("capacity", [1, 2, 32])
def test_positive_integer_capacities_are_accepted(capacity):
    session, client, _ = _session(latency_buffer_size=capacity)
    try:
        assert session.status()["latency_buffer_capacity"] == capacity
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("capacity", [True, False, 0, -1, 2.0, float("nan"), "2", None])
def test_invalid_capacities_are_rejected_before_execution(capacity):
    with pytest.raises(ValueError, match="latency_buffer_size"):
        _session(latency_buffer_size=capacity)


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_reset_clears_samples_and_fit_without_changing_buffer_configuration(enable_rtc):
    session, client, clock = _session(enable_rtc=enable_rtc, latency_buffer_size=2)
    try:
        _submit_next(session, client, 0)
        assert _complete(session, client, clock, 0, 2.0)["latency_fit"]["sample_count"] == 1
        session.reset(_actions(start=500))
        status = session.status()
        assert status["latency_samples"] == []
        assert status["latency_buffer_capacity"] == 2
        assert status["latency_history_basis"] == "encoded_window_model_seconds"
        assert status["latency_fit"]["sample_count"] == 0
        assert status["latency_fit"]["fit_mode"] == "unavailable"
        assert status["tick"] == 0 and status["rtc_enabled"] is enable_rtc
    finally:
        client.unblock()
        session.close()


def test_completed_sample_uses_history_frozen_at_submission(monkeypatch):
    session, client, clock = _session()
    try:
        _submit_next(session, client, 0)
        # Simulate a future history-selection change after submission. The
        # finished sample must retain the actual submitted window's label.
        monkeypatch.setattr(session, "_history_seconds", lambda: 2.4)
        status = _complete(session, client, clock, 0, 2.0)
        assert status["latency_samples"] == [dict(history_seconds=1024 / 30, elapsed_seconds=2.0)]
    finally:
        client.unblock()
        session.close()


class _HintClient(_Client):
    def __init__(self):
        super().__init__()
        self.hint_started, self.hint_release = Event(), Event()

    def prepare_window(self, *, anchor_tick, revision):
        super().prepare_window(anchor_tick=anchor_tick, revision=revision)
        if anchor_tick == 32:
            self.hint_started.set()
            if not self.hint_release.wait(5):
                raise TimeoutError("Test did not release the next-window hint")

    def unblock(self):
        super().unblock()
        self.hint_release.set()


def test_completed_latency_includes_time_spent_enqueuing_the_next_window_hint():
    client = _HintClient()
    session, _, clock = _session(client=client)
    try:
        _submit_next(session, client, 0)
        clock.advance(2.0)
        client.release[0].set()
        assert client.hint_started.wait(2)
        status = session.status()
        assert status["latency_samples"] == [] and not status["inference_ready"]
        clock.advance(3.0)
        client.hint_release.set()
        status = session.join(2)
        assert status["latency_samples"] == [dict(history_seconds=1024 / 30, elapsed_seconds=5.0)]
        assert status["latency_fit"]["predicted_seconds"] == 5.0
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_pacing_changes_wall_speed_while_samples_keep_the_encoded_history_basis(enable_rtc):
    clock, accepted = _Clock(), []
    client = _CalibratedClient(clock)

    def acknowledge(rate):
        accepted.append(rate)
        return True

    session = AsyncRemoteSession(
        client,
        _actions(),
        enable_rtc=enable_rtc,
        execution_policy=RTCExecutionPolicy(),
        execution_rate_callback=acknowledge,
        calibration_warmups=0,
        calibration_samples=1,
        clock=clock,
    )
    try:
        prepared = session.prepare(**_measurement(0))
        assert prepared["execution_hz"] == accepted[-1] < 30
        _submit_next(session, client, 0)
        first_rate = session.execution_hz
        prefix = client.calls[0]["rtc"]
        clock.advance(10.0)
        session.get_action(**_measurement(17))
        assert session.execution_hz == accepted[-1] < first_rate
        client.release[0].set()
        status = session.join(2)
        assert [sample["history_seconds"] for sample in status["latency_samples"]] == [1024 / 30, 1024 / 30]
        assert status["latency_fit"]["fit_mode"] == "constant"
        assert status["latency_fit"]["predicted_seconds"] == 10.0
        assert status["execution_advice"]["sample_count"] == 1
        assert status["controller_hz"] == 30 and status["minimum_execution_steps"] == 16
        assert status["handoff_steps"] == 24 and status["estimated_delay_steps"] == 8
        assert client.calls[0]["rtc"] is prefix
        if enable_rtc:
            assert prefix.delay_steps == 8
    finally:
        client.unblock()
        session.close()
