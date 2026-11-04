# SPDX-License-Identifier: OpenMDW-1.1
"""Single-model WebSocket service for client-owned RTC execution sessions."""

import argparse
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cosmos_framework.inference.causal_action.wire import HISTORY_TRANSPORT, MAX_MESSAGE_BYTES, PROTOCOL, packb, unpackb

logger = logging.getLogger(__name__)


class PolicyConnection:
    """Validate request identity without owning the client's execution clock."""

    def __init__(self, policy):
        self.policy = policy
        self.episode_id = None
        self.last_request_id = -1
        self.last_tick = 0
        self.last_revision = 0
        self.needs_reset = False
        self.history_images = {}

    def _expand_history(self, request, tick, *, reset):
        """Stage immutable measured history only; current images are never cached."""
        images = {} if reset else dict(self.history_images)
        records = request.get("history", [])
        if not isinstance(records, list) or (reset and records):
            raise ValueError("history must be a list and reset must not contain history")
        expanded, previous = [], -1
        for record in records:
            if not isinstance(record, dict) or not set(record) <= {"tick", "image"}:
                raise ValueError("history records must contain tick and optionally image")
            index = record.get("tick")
            if type(index) is not int or not previous < index < tick:
                raise ValueError("Historical ticks must be unique, ordered and before the current observation")
            previous = index
            if "image" in record:
                image = np.asarray(record["image"])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError("Cached history requires real uint8 CHW images")
                if index in images and not np.array_equal(images[index], image):
                    raise ValueError("A historical image cannot be overwritten at an existing tick")
                if index not in images:
                    image = image.copy()
                    image.setflags(write=False)
                    images[index] = image
            elif index not in images:
                raise ValueError("Unknown historical image reference; reset is required")
            expanded.append(dict(tick=index, image=images[index]))
        capacity = self.policy.metadata["history_blocks"] * self.policy.metadata["actions_per_block"]
        images = {index: image for index, image in images.items() if index >= tick - capacity}
        return {**request, "history": expanded}, images

    def _simulation_history(self, request, tick, *, reset):
        """Validate one executed batch, then sample the existing model history grid."""
        if request.get("rtc") is not None or request.get("history", []) != []:
            raise ValueError(
                "simulation_sync accepts executed_observations without wire RTC prefixes or history references"
            )
        records = request.get("executed_observations", [])
        if not isinstance(records, list):
            raise ValueError("executed_observations must be a list")
        if reset or request["operation"] == "calibrate":
            if records:
                raise ValueError("Stationary simulation requests must not contain executed_observations")
            if request["operation"] == "calibrate":
                if tick != 0 or self.last_tick != 0:
                    raise ValueError("Simulation calibration must precede motion")
                return {**request, "history": []}, dict(self.history_images)
            images = {}
            frames = [dict(tick=0, image=request["image"])]
        else:
            if tick != self.last_tick + 16 or len(records) != 16:
                raise ValueError("simulation_sync requires exactly 16 executed_observations and a 16-step advance")
            images = dict(self.history_images)
            frames = records
        stride = self.policy.metadata["video_stride"]
        shape = next(iter(images.values())).shape if images else None
        for expected, record in zip(range(0 if reset else self.last_tick + 1, tick + 1), frames, strict=True):
            if not isinstance(record, dict) or set(record) != {"tick", "image"}:
                raise ValueError("Executed observations must contain tick and image")
            if type(record["tick"]) is not int or record["tick"] != expected:
                raise ValueError("Executed observation ticks must be consecutive post-action ticks")
            image = np.asarray(record["image"])
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[0] != 3
                or min(image.shape[1:]) < 1
                or (shape is not None and image.shape != shape)
            ):
                raise ValueError("Executed observations require matching real uint8 CHW images")
            shape = image.shape
            if expected == tick and not np.array_equal(image, request["image"]):
                raise ValueError("The final executed observation must match the current image")
            # Every frame is validated; only the training stride is retained.
            if expected % stride == 0:
                image = image.copy()
                image.setflags(write=False)
                images[expected] = image
        horizon = self.policy.metadata["actions_per_block"]
        capacity = self.policy.metadata["history_blocks"] * horizon
        blocks = min(tick // horizon, self.policy.metadata["history_blocks"])
        history = [dict(tick=t, image=images[t]) for t in range(tick - blocks * horizon, tick, stride)]
        images = {t: image for t, image in images.items() if t >= tick - capacity}
        return {**request, "history": history}, images

    def infer(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        request_id, episode_id = request.get("request_id"), request.get("episode_id")
        if type(request_id) is not int or request_id <= self.last_request_id:
            raise ValueError("request_id must increase on this connection")
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128:
            raise ValueError("episode_id must be a nonempty bounded string")
        self.last_request_id = request_id
        operation = request.get("operation")
        if operation not in ("reset", "predict", "calibrate"):
            raise ValueError("operation must be reset, predict or calibrate")
        if operation == "calibrate" and not self.policy.metadata.get("simulation_sync", False):
            raise ValueError("Calibration operations require simulation_sync")
        tick, revision = request.get("tick", 0), request.get("revision", 0)
        if type(tick) is not int or tick < 0 or type(revision) is not int or revision < 0:
            raise ValueError("tick and revision must be nonnegative integers")
        if operation == "reset":
            if tick != 0 or request.get("rtc") is not None:
                raise ValueError("Reset bootstraps ordinary inference at tick zero")
            if episode_id == self.episode_id:
                raise ValueError("Reset requires a new episode_id")
        else:
            if self.needs_reset or self.episode_id is None:
                raise ValueError("Reset is required before inference")
            if episode_id != self.episode_id:
                raise ValueError("Request belongs to a different episode")
            if tick < self.last_tick or revision < self.last_revision:
                raise ValueError("Observation tick/revision cannot go backwards")
            if request.get("rtc") is not None and not self.policy.metadata.get("rtc_enabled", False):
                raise ValueError("RTC is disabled on this server")
        started = time.perf_counter()
        try:
            expand = (
                self._simulation_history if self.policy.metadata.get("simulation_sync", False) else self._expand_history
            )
            expanded, history_images = expand(request, tick, reset=operation == "reset")
            result = self.policy.infer(expanded)
            if not isinstance(result, dict):
                raise ValueError("Policy response must be a dictionary")
        except Exception:
            self.needs_reset = True
            raise
        self.episode_id = episode_id
        self.last_tick, self.last_revision = tick, revision
        self.needs_reset = False
        self.history_images = history_images
        result = dict(result)
        result.update(request_id=request_id, episode_id=episode_id, tick=tick, revision=revision)
        result["server_timing"] = {"infer_seconds": time.perf_counter() - started}
        result["cached_history_ticks"] = sorted(t for t in history_images if t < tick)
        return result


async def serve(policy, *, host="0.0.0.0", port=8000, ready=None, stop=None):
    """Serve one connection; serialize model work outside the asyncio event loop.

    Optional ready(server)/stop hooks support local integration tests. Disconnects
    never permit another client to use a model while an old inference is running.
    """
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed

    connected = False
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-rpc-model")

    async def handler(connection):
        nonlocal connected
        if connected:
            await connection.close(code=1013, reason="A policy client is already connected")
            return
        connected = True
        lifecycle = PolicyConnection(policy)
        try:
            await connection.send(
                packb({**policy.metadata, "protocol": PROTOCOL, "history_transport": HISTORY_TRANSPORT})
            )
            async for payload in connection:
                request = None
                try:
                    request = unpackb(payload)
                    result = await asyncio.get_running_loop().run_in_executor(executor, lifecycle.infer, request)
                except Exception as error:
                    logger.exception("Causal policy request failed")
                    result = dict(error=str(error), reset_required=True)
                    if isinstance(request, dict):
                        result.update(
                            {key: request.get(key) for key in ("request_id", "episode_id", "tick", "revision")}
                        )
                    lifecycle.needs_reset = True
                await connection.send(packb(result))
        except ConnectionClosed:
            pass
        finally:
            # Await serialized cleanup before making the model available again.
            try:
                reset = getattr(policy, "clear", None) or getattr(policy, "close_episode", None)
                if reset is not None:
                    await asyncio.get_running_loop().run_in_executor(executor, reset)
            finally:
                connected = False

    try:
        async with websocket_serve(
            handler,
            host,
            port,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            logger.info("Causal policy listening on ws://%s:%s", host, server.sockets[0].getsockname()[1])
            if ready is not None:
                ready(server)
            if stop is None:
                await server.serve_forever()
            else:
                await stop.wait()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "processor", "vae", "statistics"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--robot-config")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--history", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--execute-steps", type=int, default=16)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--resolution", default="480")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-mode", choices=("off", "plans", "full"), default="full")
    p.add_argument("--enable_rtc", "--enable-rtc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--simulation-sync",
        "--simulation_sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emulate RTC delay and adaptive speed with paused inference; return 16 mixed old/new actions",
    )
    p.add_argument("--override", action="append", default=[])
    return p


def main():
    args = parser().parse_args()
    from cosmos_framework.inference.causal_action.serving_model import load_policy

    policy = load_policy(args)
    try:
        asyncio.run(serve(policy, host=args.host, port=args.port))
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
