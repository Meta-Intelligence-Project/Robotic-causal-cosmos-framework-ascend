# SPDX-License-Identifier: OpenMDW-1.1
"""Client-owned RTC delay prediction for a step-driven simulator.

The simulator pauses during model RPCs. Their measured latency still determines
the reserved old-action prefix and the logical execution rate for the next
16-action interval; this is a virtual RTC experiment, not wall-clock execution.
"""

import logging
import math
from collections import deque
from dataclasses import asdict
from time import perf_counter

import torch

from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
from cosmos_framework.inference.causal_action.rtc_session import RTCDeadlineMiss, RTCObservation

logger = logging.getLogger(__name__)


class SimulatedRTCSession:
    """Execute fixed 16-step old/new batches using only previously timed RPCs."""

    horizon = 32
    steps = 16

    def __init__(
        self,
        client,
        initial_actions,
        *,
        execution_hz,
        execution_rate_callback,
        execution_policy=None,
        servo_hz=None,
        beta=5.0,
        clock=None,
    ):
        if (
            isinstance(execution_hz, bool)
            or not math.isfinite(execution_hz)
            or execution_hz <= 0
            or not callable(execution_rate_callback)
        ):
            raise ValueError("Simulated RTC needs a positive execution rate and its acknowledgement callback")
        if servo_hz is not None and (isinstance(servo_hz, bool) or not math.isfinite(servo_hz) or servo_hz <= 0):
            raise ValueError("servo_hz must be positive and finite")
        if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        if client.metadata["actions_per_block"] != 32 or client.metadata["execute_steps"] != 16:
            raise ValueError("Simulated RTC requires prediction H=32 and execution s=16")
        self.client = client
        self.controller_hz = float(client.metadata["controller_hz"])
        self.execution_hz = float(execution_hz)
        self.execution_policy = RTCExecutionPolicy() if execution_policy is None else execution_policy
        if not isinstance(self.execution_policy, RTCExecutionPolicy):
            raise ValueError("execution_policy must be an RTCExecutionPolicy")
        self._rate_callback, self._servo_hz = execution_rate_callback, servo_hz
        self._clock = perf_counter if clock is None else clock
        self.beta = float(beta)
        self._actions = self._validate_actions(initial_actions)
        self._samples = deque(maxlen=8)
        self._observations = []
        self._tick = self._origin = self._requests = 0
        self._prepared = self._closed = False
        self._error = self._last_request = self._advice = self._calibration = None
        self._last_action = None

    @staticmethod
    def _validate_actions(actions):
        actions = torch.as_tensor(actions).detach().float().cpu()
        if actions.shape != (16, 55) or not torch.isfinite(actions).all():
            raise ValueError("Simulated RTC requires 16 finite unified55 actions per response")
        return actions.clone()

    def _check_state(self):
        if self._closed:
            raise RuntimeError("Simulated RTC session is closed")
        if self._error is not None:
            raise self._error

    def _history_seconds(self, tick):
        blocks = min(tick // self.horizon, self.client.metadata["history_blocks"])
        return blocks * self.horizon / self.controller_hz

    def _recommend(self, tick):
        advice = self.execution_policy.recommend(
            self._history_seconds(tick),
            horizon=self.horizon,
            execution_steps=self.steps,
            current_hz=self.execution_hz,
            samples=tuple(self._samples),
        )
        if not advice.feasible:
            raise RTCDeadlineMiss(f"Cannot satisfy simulated RTC budget: {advice.reason}")
        rate = advice.recommended_hz
        if self._servo_hz is not None:
            # Recover an already quantized rate before testing the inequality;
            # e.g. 30 / (30 / 13) can round slightly above thirteen.
            divisor = max(1, round(self._servo_hz / rate))
            rate = self._servo_hz / divisor
            while rate > advice.recommended_hz:
                divisor += 1
                rate = self._servo_hz / divisor
        if rate < self.execution_policy.minimum_execution_hz:
            raise RTCDeadlineMiss("Servo-rate quantization exceeds the minimum simulated execution rate")
        delay = math.ceil(advice.budget_seconds * rate) + self.execution_policy.margin_steps
        if not 1 <= delay <= self.steps:
            raise RTCDeadlineMiss(f"Simulated RTC reserved prefix {delay} exceeds the 16-step interval")
        if rate < self.execution_hz:
            if self._rate_callback(rate) is not True:
                raise RTCDeadlineMiss("The client did not acknowledge the simulated execution rate")
            self.execution_hz = rate
        self._advice = {**asdict(advice), "recommended_hz": rate, "delay_steps": delay}
        return delay

    def _measure(self, run):
        started = self._clock()
        result = run()
        seconds = self._clock() - started
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("The RPC clock must measure finite nonnegative durations")
        return result, seconds

    def prepare(self, **measurement):
        self._check_state()
        if self._prepared:
            return self.status()
        try:

            def run():
                return self.client.calibrate_simulation(measurement, delay_steps=1, beta=self.beta)

            _, warmup = self._measure(run)
            samples = []
            for _ in range(2):
                _, seconds = self._measure(run)
                samples.append(seconds)
                self._samples.append((self._history_seconds(0), seconds))
            self._calibration = dict(warmup_seconds=[warmup], sample_seconds=samples)
            self._recommend(self.steps)
            self._prepared = True
        except Exception as error:
            self._error = error
            logger.warning("Simulated RTC startup failed: %s", error)
            raise
        return self.status()

    def get_action(self, *, state, state_mask, image, action_mask):
        self._check_state()
        measurement = dict(state=state, state_mask=state_mask, image=image, action_mask=action_mask)
        self.prepare(**measurement)
        if self._tick > self._origin:
            self._observations.append(RTCObservation(self._tick, image.detach().cpu().clone()))
        if self._tick - self._origin == self.steps:
            try:
                delay = self._recommend(self._tick)
                rate = self.execution_hz
                actions, seconds = self._measure(
                    lambda: self.client.plan(
                        **measurement,
                        tick=self._tick,
                        revision=0,
                        rtc=None,
                        executed_observations=tuple(self._observations),
                        simulation_rtc=dict(delay_steps=delay, beta=self.beta),
                    )
                )
                info = self.client.last_simulation_info
                expected = dict(
                    delay_steps=delay,
                    old_action_steps=delay,
                    new_action_steps=self.steps - delay,
                    request_tick=self._tick,
                )
                if not isinstance(info, dict) or any(
                    type(info.get(key)) is not int or info[key] != value for key, value in expected.items()
                ):
                    raise ValueError("Server returned mismatched simulated RTC handoff metadata")
                actual_delay = math.ceil(seconds * rate)
                self._last_request = dict(
                    observation_tick=self._tick,
                    history_seconds=self._history_seconds(self._tick),
                    elapsed_seconds=seconds,
                    execution_hz=rate,
                    estimated_delay_steps=delay,
                    actual_delay_steps=actual_delay,
                    deadline_miss=actual_delay > delay,
                    accepted=False,
                    **expected,
                )
                self._samples.append((self._history_seconds(self._tick), seconds))
                if actual_delay > delay:
                    raise RTCDeadlineMiss(
                        f"Simulated RTC needed {actual_delay} old actions but reserved {delay}; stopping simulation"
                    )
                self._actions = self._validate_actions(actions)
                self._origin = self._tick
                self._requests += 1
                self._observations.clear()
                self._last_request["accepted"] = True
            except Exception as error:
                self._error = error
                logger.warning("Simulated RTC handoff failed: %s", error)
                raise
        offset = self._tick - self._origin
        source = "bootstrap" if self._origin == 0 else "new_prediction"
        prediction_tick, prediction_row = self._origin, offset
        if self._origin and offset < self._last_request["delay_steps"]:
            source = "old_prefix"
            prediction_tick, prediction_row = self._origin - self.steps, self.steps + offset
        self._last_action = dict(
            tick=self._tick, prediction_tick=prediction_tick, prediction_row=prediction_row, source=source
        )
        action = self._actions[offset].clone()
        self._tick += 1
        return action

    def status(self):
        return dict(
            tick=self._tick,
            chunk_start_tick=self._origin,
            remaining_actions=max(0, self.steps - (self._tick - self._origin)),
            prediction_horizon=self.horizon,
            minimum_execution_steps=self.steps,
            controller_hz=self.controller_hz,
            execution_hz=self.execution_hz,
            rtc_enabled=self._prepared and self._error is None,
            simulated_rtc=True,
            virtual_rtc_feasible=self._prepared and self._error is None,
            execution_mode="simulation_rtc",
            prepared=self._prepared,
            closed=self._closed,
            error=None if self._error is None else str(self._error),
            ordinary_requests=1,
            published_requests=self._requests,
            inference_pending=False,
            adaptive_execution=True,
            estimated_delay_steps=None if self._advice is None else self._advice["delay_steps"],
            latency_samples=[dict(history_seconds=h, elapsed_seconds=t) for h, t in self._samples],
            execution_advice=self._advice,
            startup_calibration=self._calibration,
            last_request=self._last_request,
            last_action=self._last_action,
        )

    def close(self, *, wait=True):
        self._closed = True
