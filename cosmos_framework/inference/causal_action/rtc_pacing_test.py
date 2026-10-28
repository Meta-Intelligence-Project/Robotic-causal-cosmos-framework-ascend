# SPDX-License-Identifier: OpenMDW-1.1

import math
from collections import deque

import pytest

from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy


def recommend(policy, *, history=3.0, rate=3.0, samples=((0, 2), (1, 3), (2, 4)), **options):
    return policy.recommend(history, horizon=32, execution_steps=16, current_hz=rate, samples=samples, **options)


def test_no_latency_is_fabricated_before_startup_measurement():
    advice = recommend(RTCExecutionPolicy(), samples=())
    assert not advice.feasible
    assert advice.reason == "no_latency_samples"
    assert advice.budget_seconds is None
    assert advice.sample_count == 0


def test_cpu_line_is_fitted_from_the_supplied_completed_requests():
    advice = recommend(RTCExecutionPolicy())
    assert advice.feasible and advice.fit_mode == "linear"
    assert advice.sample_count == advice.distinct_history_count == 3
    assert (advice.history_min_seconds, advice.history_max_seconds) == (0, 2)
    assert advice.intercept_seconds == pytest.approx(2)
    assert advice.slope_seconds_per_history_second == pytest.approx(1)
    assert advice.predicted_seconds == pytest.approx(5)
    assert advice.residual_seconds == pytest.approx(0)
    assert advice.budget_seconds == pytest.approx(6)
    assert advice.recommended_hz == pytest.approx(2.5)


def test_an_updated_buffer_changes_the_prediction_without_changing_policy_parameters():
    policy = RTCExecutionPolicy()
    before = recommend(policy, samples=((0, 2), (1, 3)))
    after = recommend(policy, samples=((0, 2), (1, 4)))
    assert before.slope_seconds_per_history_second == 1
    assert after.slope_seconds_per_history_second == 2
    assert after.predicted_seconds > before.predicted_seconds
    assert after.recommended_hz < before.recommended_hz


def test_a_single_history_uses_the_largest_measured_latency_until_a_slope_is_observed():
    advice = recommend(RTCExecutionPolicy(), samples=((0, 2), (0, 3)), history=1)
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.distinct_history_count == 1
    assert advice.slope_seconds_per_history_second == 0
    assert advice.predicted_seconds == 3
    assert advice.budget_seconds == pytest.approx(3.6)


def test_a_full_window_with_only_one_history_length_is_not_singular():
    advice = recommend(RTCExecutionPolicy(), history=1024 / 30, samples=((1024 / 30, 12), (1024 / 30, 13)))
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.predicted_seconds == 13


def test_negative_observed_slope_cannot_predict_that_more_history_becomes_faster():
    advice = recommend(RTCExecutionPolicy(), samples=((0, 4), (1, 3), (2, 2)))
    assert advice.slope_seconds_per_history_second == 0
    assert advice.budget_seconds >= 4 * 1.2


def test_positive_residual_is_added_to_the_next_prediction():
    advice = recommend(RTCExecutionPolicy(latency_factor=1), samples=((0, 2), (1, 4), (2, 4)))
    assert advice.predicted_seconds == pytest.approx(16 / 3)
    assert advice.residual_seconds == pytest.approx(2 / 3)
    assert advice.budget_seconds == pytest.approx(6)


def test_a_local_line_cannot_produce_a_negative_latency_budget():
    advice = recommend(RTCExecutionPolicy(), history=0, samples=((10, 1), (11, 2)))
    assert advice.intercept_seconds == pytest.approx(-9)
    assert advice.predicted_seconds == 0
    assert advice.budget_seconds == pytest.approx(2.4)


def test_no_offline_maximum_history_is_required_for_online_prediction():
    advice = recommend(RTCExecutionPolicy(), history=36, samples=((34, 12), (35, 13)))
    assert advice.feasible
    assert advice.predicted_seconds == pytest.approx(14)


def test_only_the_current_bounded_buffer_is_used_without_retained_fit_state():
    samples = deque([(0, 100), (1, 100)], maxlen=2)
    policy = RTCExecutionPolicy()
    first = recommend(policy, samples=samples)
    samples.extend([(2, 3), (3, 4)])
    second = recommend(policy, history=4, samples=samples)
    assert first.predicted_seconds == 100
    assert second.sample_count == 2
    assert second.history_min_seconds == 2
    assert second.predicted_seconds == pytest.approx(5)


def test_large_absolute_histories_preserve_small_differences_when_fitting():
    origin = 1e12
    advice = recommend(
        RTCExecutionPolicy(), history=origin + 3, samples=((origin, 2), (origin + 1, 3), (origin + 2, 4))
    )
    assert advice.slope_seconds_per_history_second == 1
    assert advice.predicted_seconds == 5


def test_recent_wall_latency_and_transport_headroom_override_the_fit():
    advice = recommend(RTCExecutionPolicy(extra_latency_seconds=0.5), observed_seconds=10)
    assert advice.budget_seconds == 12.5
    assert advice.recommended_hz == pytest.approx(15 / 12.5)


def test_observed_seconds_can_supply_a_constant_startup_estimate():
    advice = recommend(RTCExecutionPolicy(), samples=(), observed_seconds=2)
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.sample_count == 0
    assert advice.budget_seconds == pytest.approx(2.4)


def test_policy_does_not_speed_up_after_a_lower_latency_observation():
    policy = RTCExecutionPolicy()
    first = recommend(policy, observed_seconds=20)
    second = recommend(policy, rate=first.recommended_hz, samples=((0, 1), (1, 1)))
    assert first.feasible and second.feasible
    assert second.recommended_hz == first.recommended_hz


def test_history_growth_reduces_rate_before_the_prefix_budget_is_exhausted():
    policy = RTCExecutionPolicy()
    samples = deque(maxlen=8)
    rate = 5.0
    for block in range(33):
        history = block * 32 / 30
        samples.append((history, 2 + 0.17 * history + 0.003 * history * history))
        advice = recommend(policy, history=history + 32 / 30, rate=rate, samples=samples)
        assert advice.feasible and 0 < advice.recommended_hz <= rate
        assert math.ceil(advice.budget_seconds * advice.recommended_hz) + policy.margin_steps <= 16
        rate = advice.recommended_hz
    assert rate < 1.2


def test_minimum_rate_can_make_slowdown_infeasible():
    advice = recommend(RTCExecutionPolicy(minimum_execution_hz=2), samples=((0, 10),))
    assert not advice.feasible and advice.reason == "minimum_execution_hz"
    assert advice.recommended_hz == 3


def test_exactly_feasible_minimum_rate_is_accepted():
    advice = recommend(RTCExecutionPolicy(latency_factor=1, minimum_execution_hz=2), samples=((0, 7.5),))
    assert advice.feasible and advice.recommended_hz == 2
    assert advice.delay_steps == 16


def test_asymmetric_replanning_uses_the_smaller_available_prefix():
    policy = RTCExecutionPolicy(margin_steps=2)
    advice = policy.recommend(0, horizon=32, execution_steps=24, current_hz=3, samples=((0, 10),))
    assert advice.feasible and advice.delay_steps == 8
    assert advice.recommended_hz == pytest.approx(6 / 12)
    assert not policy.recommend(0, horizon=32, execution_steps=1, current_hz=3, samples=((0, 10),)).feasible


@pytest.mark.parametrize("budget", [0.1, 0.3, 1, 3.14, 11.975, 15.484, 1000])
def test_float_boundaries_preserve_the_integer_deadline_inequality(budget):
    advice = recommend(RTCExecutionPolicy(latency_factor=1), rate=10000, samples=((0, budget),))
    assert advice.feasible
    assert math.ceil(budget * advice.recommended_hz) + 1 <= 16


@pytest.mark.parametrize(
    "options",
    [
        dict(latency_factor=0.9),
        dict(latency_factor=True),
        dict(extra_latency_seconds=float("inf")),
        dict(extra_latency_seconds=-1),
        dict(margin_steps=True),
        dict(margin_steps=0),
        dict(minimum_execution_hz=-1),
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        RTCExecutionPolicy(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(history_seconds=-1),
        dict(horizon=True),
        dict(execution_steps=32),
        dict(current_hz=0),
        dict(observed_seconds=float("nan")),
        dict(samples=None),
        dict(samples=((0,),)),
        dict(samples=((-1, 2),)),
        dict(samples=((1, -2),)),
        dict(samples=((True, 2),)),
        dict(samples=((1, float("nan")),)),
    ],
)
def test_invalid_requests_are_rejected(options):
    arguments = dict(history_seconds=0, horizon=32, execution_steps=16, current_hz=3, samples=((0, 2),))
    with pytest.raises(ValueError):
        RTCExecutionPolicy().recommend(**(arguments | options))
