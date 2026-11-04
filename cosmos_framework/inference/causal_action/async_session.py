# SPDX-License-Identifier: OpenMDW-1.1
"""Ordinary prediction on a worker, with fixed tick-aligned buffer handoffs."""

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock, get_ident

import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55


class AsyncSessionError(RuntimeError):
    """Command delivery must stop until the session is reset or closed."""


class AsyncPlannerError(AsyncSessionError):
    """Ordinary background planning or observation transport failed."""


def resolve_async_schedule(H, S, R=None):
    """Return trigger S and handoff R for one H-step prediction and one candidate."""
    if type(H) is not int or type(S) is not int or H < 1 or S < 1:
        raise ValueError("Async H and S must be positive integers")
    if R is None:
        R = min(H, S + max(1, S // 2))
    # H是预测长度，S是提交新推理的时刻，R是交接时刻；三者不可混成执行长度。
    # R<=2S保证交接时不会错过下一次提交（R=2S需要同tick先交接再提交）。
    if type(R) is not int or not 0 < S < R <= min(H, 2 * S):
        raise ValueError("Async schedule requires 0 < S < R <= min(H, 2*S)")
    return S, R


def _frequency(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class _Observation:
    tick: int
    image: torch.Tensor


@dataclass
class _Candidate:
    revision: int
    anchor: int
    handoff_tick: int
    inputs: dict
    future: object = None
    actions: torch.Tensor | None = None
    late: bool = False


class AsyncRemoteSession:
    """Execute old commands until R, then use the new plan at offset R-S.

    Planning uses a frozen measurement taken at S and never supplies RTC guidance.
    A result arriving early waits for R. A late result pauses command delivery at
    R without advancing the control tick or replaying an expired command.
    """

    def __init__(self, client, initial_actions, *, execution_hz=None):
        self.client, self.template = client, ActionStateTemplate55()
        self._actions = self._validate_actions(initial_actions)
        self.horizon = len(self._actions)
        advertised = client.metadata.get("action_horizon", client.metadata.get("actions_per_block", self.horizon))
        if advertised != self.horizon:
            raise ValueError("Initial actions must preserve the server prediction horizon")
        self.steps, self.handoff_steps = resolve_async_schedule(
            self.horizon, client.metadata["execute_steps"], client.metadata.get("handoff_steps")
        )
        self.controller_hz = _frequency(client.metadata["controller_hz"], "controller_hz")
        self.execution_hz = _frequency(self.controller_hz if execution_hz is None else execution_hz, "execution_hz")
        history_steps = client.metadata.get("history_steps")
        if history_steps is None:
            history_steps = client.metadata.get("history_blocks", 32) * 32
        if type(history_steps) is not int or history_steps < 0:
            raise ValueError("history_steps must be a nonnegative integer")
        for method in ("plan", "observe", "prepare_window"):
            if not callable(getattr(client, method, None)):
                raise ValueError(f"Async client requires {method}")
        self._history = deque(maxlen=max(1, history_steps))
        self._lock, self._callback_lock = RLock(), RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-async-policy")
        self._worker_thread = None
        self._pending = self._future = self._consumer = None
        self._tick = self._origin = self._revision = 0
        self._requests = self._published = self._late_handoffs = 0
        self._closed = self._resetting = self._prepared = self._waiting = False
        self._error = self._action_mask = self._state_mask = self._last_hint = None
        self._last_request = None

    def _validate_actions(self, actions):
        if (
            not isinstance(actions, torch.Tensor)
            or not actions.is_floating_point()
            or actions.ndim != 2
            or actions.shape[1] != self.template.width
            or not len(actions)
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite floating-point [H,55] targets")
        return actions.detach().float().cpu().clone()

    def _check(self):
        if self._closed:
            raise AsyncSessionError("Async session is closed")
        if self._resetting:
            raise AsyncSessionError("Async session is resetting")
        if self._error is not None:
            raise self._error

    def _live(self, revision):
        return not self._closed and not self._resetting and revision == self._revision

    def _fail(self, exception, revision):
        error = (
            exception
            if isinstance(exception, AsyncPlannerError)
            else AsyncPlannerError(f"Async planning failed: {exception}")
        )
        if error is not exception:
            error.__cause__ = exception
        with self._lock:
            if self._live(revision) and self._error is None:
                self._error = error
        return error

    def _notify(self, method, revision, **kwargs):
        # 客户端方法只入发送队列；不持controller锁调用网络/用户回调。
        # 独立回调锁让reset/close能隔离旧episode的迟到hint。
        with self._callback_lock:
            with self._lock:
                if not self._live(revision):
                    return False
                if method == "prepare_window" and self._last_hint == (revision, kwargs["anchor_tick"]):
                    return True
            try:
                getattr(self.client, method)(revision=revision, **kwargs)
            except Exception as exc:
                error = self._fail(exc, revision)
                if error is exc:
                    raise
                raise error from exc
            with self._lock:
                if method == "prepare_window" and self._live(revision):
                    self._last_hint = revision, kwargs["anchor_tick"]
            return True

    def _measurement(self, *, state, state_mask, image, action_mask):
        if not isinstance(state, torch.Tensor) or state.shape != (self.template.width,):
            raise ValueError("state must be a current [55] measurement")
        sm = self.template.validate_valid_mask(state_mask).detach().cpu().clone()
        am = self.template.validate_valid_mask(action_mask).detach().cpu().clone()
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Current actions require valid measured anchor channels")
        state = self.template.sanitize(state.detach().float().cpu(), sm).clone()
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or min(image.shape) < 1
            or not torch.isfinite(image).all()
        ):
            raise ValueError("image must be a finite real [3,H,W] observation")
        return state, sm, image.detach().cpu().clone(), am

    def _accept_masks(self, sm, am):
        if self._action_mask is None:
            self.template.sanitize(self._actions, am)
            self._actions = self._actions.masked_fill(~am, 0)
            self._action_mask, self._state_mask = am.clone(), sm.clone()
        elif not torch.equal(am, self._action_mask) or not torch.equal(sm, self._state_mask):
            raise ValueError("Robot validity masks cannot change within an async episode; reset first")

    def prepare(self, **measurement):
        values = self._measurement(**measurement) if measurement else None
        with self._lock:
            self._check()
            if get_ident() == self._worker_thread:
                raise AsyncSessionError("Cannot prepare from the planning worker")
            if values is not None:
                self._accept_masks(values[1], values[3])
            if self._prepared:
                return self.status()
            revision, anchor = self._revision, self._origin + self.steps
        self._notify("prepare_window", revision, anchor_tick=anchor)
        with self._lock:
            self._check()
            if revision != self._revision:
                raise AsyncSessionError("Episode changed while preparing async execution")
            self._prepared = True
            return self.status()

    def _submit(self, state, sm, image, am):
        inputs = dict(
            state=state,
            state_mask=sm,
            action_mask=am,
            image=image,
            history=tuple(self._history),
            tick=self._tick,
            revision=self._revision,
            rtc=None,
        )
        candidate = _Candidate(self._revision, self._tick, self._origin + self.handoff_steps, inputs)
        self._pending = candidate
        self._requests += 1
        self._last_request = dict(tick=candidate.anchor, handoff_tick=candidate.handoff_tick, accepted=False)
        candidate.future = self._future = self._executor.submit(self._run_candidate, candidate)

    def _run_candidate(self, candidate):
        with self._lock:
            self._worker_thread = get_ident()
            if not self._live(candidate.revision):
                return
        try:
            actions = self._validate_actions(self.client.plan(**candidate.inputs))
            if len(actions) != self.horizon:
                raise ValueError("Planner must preserve the prediction horizon")
            mask = candidate.inputs["action_mask"]
            self.template.sanitize(actions, mask)
            actions = actions.masked_fill(~mask, 0)
            # 推理一结束就提示下个anchor；提前量来自旧动作仍在执行，不必等R交接。
            if not self._notify("prepare_window", candidate.revision, anchor_tick=candidate.anchor + self.steps):
                return
            with self._lock:
                if self._live(candidate.revision) and self._pending is candidate:
                    candidate.actions = actions
        except Exception as exc:
            error = self._fail(exc, candidate.revision)
            if error is exc:
                raise
            raise error from exc

    def get_action(self, *, state, state_mask, image, action_mask):
        self.prepare()
        values = self._measurement(state=state, state_mask=state_mask, image=image, action_mask=action_mask)
        state, sm, image, am = values
        consumer = object()
        with self._lock:
            self._check()
            if self._consumer is not None:
                raise AsyncSessionError("Concurrent command requests are not supported")
            self._accept_masks(sm, am)
            self._consumer = consumer
            revision, tick = self._revision, self._tick
        try:
            self._notify("observe", revision, tick=tick, image=image)
            while True:
                with self._lock:
                    self._check()
                    if revision != self._revision or tick != self._tick:
                        raise AsyncSessionError("Episode or command tick changed while waiting for handoff")
                    offset = tick - self._origin
                    if offset >= self.handoff_steps:
                        candidate = self._pending
                        if candidate is None or candidate.handoff_tick != tick:
                            raise self._fail(AsyncSessionError("No matching candidate for the fixed handoff"), revision)
                        if candidate.actions is not None:
                            self._actions, self._origin = candidate.actions, candidate.anchor
                            self._pending = None
                            self._published += 1
                            self._last_request["accepted"] = True
                            self._waiting = False
                            # R=2S时，新buffer的offset正好为S；必须继续循环再提交。
                            continue
                        if not candidate.late:
                            candidate.late = True
                            self._late_handoffs += 1
                        self._waiting = True
                        future = candidate.future
                    else:
                        if offset >= self.steps and self._pending is None:
                            self._submit(state, sm, image, am)
                        command = self._actions[offset].clone()
                        self._history.append(_Observation(tick, image))
                        self._tick += 1
                        return command
                # 迟到只能在R等待，不能重放旧命令；等待期间不持controller锁。
                future.result()
        finally:
            with self._lock:
                if self._consumer is consumer:
                    self._consumer = None
                    self._waiting = False

    def status(self):
        with self._lock:
            candidate = self._pending
            ready = candidate is not None and candidate.actions is not None and not self._closed
            return dict(
                tick=self._tick,
                chunk_start_tick=self._origin,
                remaining_actions=max(0, self.handoff_steps - (self._tick - self._origin)),
                prediction_horizon=self.horizon,
                minimum_execution_steps=self.steps,
                handoff_steps=self.handoff_steps,
                controller_hz=self.controller_hz,
                execution_hz=self.execution_hz,
                playback_speed=self.execution_hz / self.controller_hz,
                rtc_enabled=False,
                execution_mode="async_non_rtc",
                prepared=self._prepared,
                closed=self._closed,
                resetting=self._resetting,
                error=None if self._error is None else str(self._error),
                ordinary_requests=self._requests,
                published_requests=self._published,
                inference_ready=ready,
                inference_pending=candidate is not None and not ready and not self._closed,
                waiting_for_handoff=self._waiting,
                late_handoffs=self._late_handoffs,
                revision=self._revision,
                adaptive_execution=False,
                last_request=None if self._last_request is None else dict(self._last_request),
            )

    def join(self, timeout=None):
        """Wait for the current worker without publishing before its fixed R tick."""
        with self._lock:
            if get_ident() == self._worker_thread:
                raise AsyncSessionError("Cannot join from the planning worker")
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        with self._lock:
            self._check()
            return self.status()

    wait_for_inference = join

    def reset(self, initial_actions):
        actions = self._validate_actions(initial_actions)
        if len(actions) != self.horizon:
            raise ValueError("Reset must preserve the prediction horizon")
        with self._callback_lock:
            with self._lock:
                if self._closed or self._resetting or get_ident() == self._worker_thread:
                    raise AsyncSessionError("Cannot reset this async session now")
                self._revision += 1
                self._resetting = True
                future = self._future
        try:
            if future is not None:
                future.exception()  # The invalidated episode cannot publish or send another hint.
            with self._lock:
                if self._closed:
                    raise AsyncSessionError("Async session closed during reset")
                self._actions = actions
                self._pending = self._future = self._consumer = None
                self._tick = self._origin = self._requests = self._published = self._late_handoffs = 0
                self._error = self._action_mask = self._state_mask = self._last_hint = self._last_request = None
                self._prepared = self._waiting = False
                self._history.clear()
        finally:
            with self._lock:
                self._resetting = False

    def close(self, *, wait=True):
        with self._callback_lock:
            with self._lock:
                if get_ident() == self._worker_thread:
                    raise AsyncSessionError("Cannot close from the planning worker")
                self._closed = True
                self._revision += 1
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        with self._lock:
            self._check()
        return self

    def __exit__(self, *args):
        self.close()
