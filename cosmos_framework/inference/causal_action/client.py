# SPDX-License-Identifier: OpenMDW-1.1
"""WebSocket planner with a local RTC clock and measured request round trips.

The wire client imports neither Torch nor model code until a local execution
session is requested. RTC execution uses CPU Torch; checkpoints stay on server.
"""

import math
import threading
import time
import uuid
from collections import deque

import numpy as np

from cosmos_framework.inference.causal_action.wire import HISTORY_TRANSPORT, MAX_MESSAGE_BYTES, PROTOCOL, packb, unpackb


class RemotePolicyError(RuntimeError):
    """A matched server response reports an inference or protocol error."""


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class CausalPolicyClient:
    def __init__(self, uri="ws://localhost:8000", *, control_mode="joint", request_timeout=120.0):
        from websockets.sync.client import connect

        if isinstance(request_timeout, bool) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive and finite")
        if control_mode not in ("joint", "eef"):
            raise ValueError("control_mode must be joint or eef")
        self.request_timeout = float(request_timeout)
        self.control_mode = control_mode
        self._lock = threading.Lock()
        self._closed = False
        self._session = None
        self._episode_id = None
        self._request_id = 0
        self._image_size = None
        self._cached_history_ticks = set()
        self.last_timing = None
        self.request_timings = []
        self.connection = connect(
            uri, compression=None, max_size=MAX_MESSAGE_BYTES, ping_interval=None, close_timeout=1, proxy=None
        )
        try:
            self.metadata = unpackb(self.connection.recv(timeout=self.request_timeout))
            if not isinstance(self.metadata, dict) or self.metadata.get("protocol") != PROTOCOL:
                raise ValueError(f"Expected a {PROTOCOL} server")
            modes = self.metadata.get("available_control_modes")
            if modes is not None and control_mode not in modes:
                raise ValueError(f"Server does not support control_mode={control_mode!r}")
        except BaseException:
            self.connection.close()
            self._closed = True
            raise

    def infer(self, observation):
        """One serialized RPC, including encoding, transport and decoding timing."""
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy connection is closed")
            operation = observation.get("operation", "predict")
            episode = str(uuid.uuid4()) if operation == "reset" else self._episode_id
            if episode is None:
                raise ValueError("Reset with a first measurement before requesting predictions")
            request = dict(observation)
            request.update(operation=operation, episode_id=episode, request_id=self._request_id)
            request.setdefault("tick", 0)
            request.setdefault("revision", 0)
            self._request_id += 1
            started = time.perf_counter()
            try:
                encoded = packb(request)
                self.connection.send(encoded)
                received = self.connection.recv(timeout=self.request_timeout)
                response = unpackb(received)
                if not isinstance(response, dict) or any(
                    response.get(key) != request[key] for key in ("request_id", "episode_id", "tick", "revision")
                ):
                    raise ValueError("Response identity does not match the submitted plan")
            except BaseException:
                # An unread late reply must never become the next RPC's result.
                self._closed = True
                self.connection.close()
                raise
            self.last_timing = {
                "round_trip_seconds": time.perf_counter() - started,
                "request_bytes": len(encoded),
                "response_bytes": len(received),
                "history_frames_sent": sum("image" in record for record in request.get("history", [])),
                "history_frames_reused": sum("image" not in record for record in request.get("history", [])),
                **response.get("server_timing", {}),
            }
            self.request_timings.append(
                dict(
                    operation=operation,
                    request_id=request["request_id"],
                    episode_id=episode,
                    tick=request["tick"],
                    revision=request["revision"],
                    **self.last_timing,
                )
            )
            if "error" in response:
                raise RemotePolicyError(response["error"])
            if self.metadata.get("history_transport") == HISTORY_TRANSPORT:
                ticks = response.get("cached_history_ticks")
                if (
                    not isinstance(ticks, list)
                    or any(type(index) is not int or not 0 <= index < request["tick"] for index in ticks)
                    or len(set(ticks)) != len(ticks)
                ):
                    self._closed = True
                    self.connection.close()
                    raise ValueError("Invalid historical image cache acknowledgement")
                self._cached_history_ticks = set(ticks)
            if operation == "reset":
                self._episode_id = episode
            return response

    def _close_session(self):
        if self._session is not None:
            session, self._session = self._session, None
            session.close()

    def reset(self, *, prompt, measurement):
        """Drain the old episode and obtain an ordinary, unexecuted bootstrap."""
        self._close_session()
        self._image_size = _numpy(measurement["image_size"]).astype(np.float32, copy=True)
        return self.infer(
            dict(
                operation="reset",
                prompt=prompt,
                state=_numpy(measurement["state"]).astype(np.float32, copy=True),
                image=_numpy(measurement["image"]).copy(),
                image_size=self._image_size,
                tick=0,
                revision=0,
                rtc=None,
            )
        )

    def _actions(self, response):
        import torch

        actions = np.asarray(response.get("actions"))
        if (
            actions.dtype.kind != "f"
            or actions.shape != (self.metadata["actions_per_block"], 55)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Server returned an invalid physical unified55 action chunk")
        return torch.from_numpy(actions.astype(np.float32, copy=True))

    def plan(self, *, state, state_mask, image, action_mask, history=(), tick, revision, rtc):
        """RPC callback for RTCSession; upload only genuinely required history."""
        for name, mask in (("state_mask", state_mask), ("action_mask", action_mask)):
            if not np.array_equal(_numpy(mask), np.asarray(self.metadata[name], dtype=bool)):
                raise ValueError(f"{name} differs from the server's configured robot")
        horizon, stride = self.metadata["actions_per_block"], self.metadata["video_stride"]
        blocks = min(tick // horizon, self.metadata["history_blocks"])
        selected = range(tick - blocks * horizon, tick, stride)
        observed = {record.tick: record.image for record in history}
        if any(index not in observed for index in selected):
            raise ValueError("Missing real historical image required by the remote batch builder")
        prefix = None
        if rtc is not None:
            prefix = dict(
                previous_actions=_numpy(rtc.previous_actions).astype(np.float32, copy=True),
                delay_steps=rtc.delay_steps,
                beta=rtc.beta,
            )
        response = self.infer(
            dict(
                state=_numpy(state).astype(np.float32, copy=True),
                image=_numpy(image).copy(),
                history=[
                    dict(tick=index)
                    if self.metadata.get("history_transport") == HISTORY_TRANSPORT
                    and index in self._cached_history_ticks
                    else dict(tick=index, image=_numpy(observed[index]).copy())
                    for index in selected
                ],
                tick=tick,
                revision=revision,
                rtc=prefix,
            )
        )
        return self._actions(response)

    def create_session(
        self,
        first_measurement,
        *,
        prompt,
        enable_rtc=False,
        execution_hz=None,
        execution_policy=None,
        execution_rate_callback=None,
        adaptive_execution=True,
        **session_options,
    ):
        """Return a bootstrapped local session; prepare before enabling motion."""
        from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
        from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy
        from cosmos_framework.inference.causal_action.rtc_session import RTCSession

        if enable_rtc and not self.metadata.get("rtc_enabled", False):
            raise ValueError("Enable RTC on the server before creating an RTC client session")
        if enable_rtc and not adaptive_execution and execution_policy is not None:
            raise ValueError("execution_policy conflicts with adaptive_execution=False")
        if enable_rtc and adaptive_execution and execution_policy is None:
            execution_policy = RTCExecutionPolicy()
        if enable_rtc and execution_policy is not None and not callable(execution_rate_callback):
            raise ValueError("Adaptive RTC needs a client execution_rate_callback")
        initial = self._actions(self.reset(prompt=prompt, measurement=first_measurement))
        frequency = self.metadata["controller_hz"]
        rate = frequency if execution_hz is None else execution_hz
        if not enable_rtc:
            if session_options:
                raise ValueError("RTC session options require enable_rtc")
            session = BlockingRemoteSession(self, initial, execution_hz=rate)
        else:
            session = RTCSession(
                self.plan,
                initial,
                template=ActionStateTemplate55(),
                controller_hz=frequency,
                execution_hz=rate,
                min_execution_steps=self.metadata["execute_steps"],
                initial_delay_steps=None,
                history_steps=self.metadata["history_blocks"] * self.metadata["actions_per_block"],
                ordinary_planner=self.plan,
                execution_policy=execution_policy,
                execution_rate_callback=execution_rate_callback,
                **session_options,
            )
        self._session = session
        return session

    def actions(self, response):
        from cosmos_framework.inference.causal_action.serving_model import adapter_from_metadata

        return adapter_from_metadata(self.metadata).split_actions(_numpy(self._actions(response)))[self.control_mode]

    def close(self):
        if not self._closed:
            self._closed = True
            self.connection.close()
        self._close_session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BlockingRemoteSession:
    """Ordinary baseline: predict H, execute s, then block for a new chunk."""

    def __init__(self, client, initial_actions, *, execution_hz):
        from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55

        if isinstance(execution_hz, bool) or not math.isfinite(execution_hz) or execution_hz <= 0:
            raise ValueError("execution_hz must be positive and finite")
        self.client, self._actions = client, initial_actions
        self.template = ActionStateTemplate55()
        self.controller_hz = client.metadata["controller_hz"]
        self.execution_hz = float(execution_hz)
        self.horizon, self.steps = len(initial_actions), client.metadata["execute_steps"]
        if type(self.steps) is not int or not 0 < self.steps <= self.horizon:
            raise ValueError("Invalid server execution horizon")
        self._history = deque(maxlen=client.metadata["history_blocks"] * self.horizon)
        self._tick = self._origin = self._requests = 0
        self._closed = False

    def prepare(self, **measurement):
        if self._closed:
            raise RuntimeError("Policy session is closed")
        return self.status()

    def get_action(self, *, state, state_mask, image, action_mask):
        from cosmos_framework.inference.causal_action.rtc_session import RTCObservation

        self.prepare()
        if self._tick - self._origin >= self.steps:
            self._actions = self.client.plan(
                state=state,
                state_mask=state_mask,
                image=image,
                action_mask=action_mask,
                history=tuple(self._history),
                tick=self._tick,
                revision=0,
                rtc=None,
            )
            self._origin = self._tick
            self._requests += 1
        action = self._actions[self._tick - self._origin].clone()
        self._history.append(RTCObservation(self._tick, image.detach().cpu().clone()))
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
            rtc_enabled=False,
            execution_mode="blocking_non_rtc",
            prepared=True,
            closed=self._closed,
            error=None,
            ordinary_requests=self._requests,
            published_requests=0,
            inference_pending=False,
            adaptive_execution=False,
        )

    def close(self, *, wait=True):
        self._closed = True
