# SPDX-License-Identifier: OpenMDW-1.1
"""Tick alignment, committed commands and asynchronous RTC lifecycle tests."""

from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.inference.causal_action.rtc import RTCPrefix
from cosmos_framework.inference.causal_action.rtc_session import (
    RTCBufferUnderrun,
    RTCDeadlineMiss,
    RTCError,
    RTCPlannerError,
    RTCSession,
)


def actions(horizon=8, start=0):
    result = torch.zeros(horizon, 55)
    result[:, 0] = start + torch.arange(horizon)
    result[:, 28] = 0.5
    return result


def inputs(tick=0):
    state = torch.zeros(55)
    state[0] = tick
    state[28] = float("nan")  # Invalid absolute-field state is allowed.
    state_mask = torch.zeros(55, dtype=torch.bool)
    state_mask[0] = True
    action_mask = state_mask.clone()
    action_mask[28] = True
    return dict(state=state, state_mask=state_mask, action_mask=action_mask, image=torch.full((3, 4, 4), tick))


class GatePlanner:
    def __init__(self, horizon=8, requests=1):
        self.horizon = horizon
        self.started = [Event() for _ in range(requests)]
        self.release = [Event() for _ in range(requests)]
        self.calls = []

    def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        self.started[index].set()
        if not self.release[index].wait(10):
            raise TimeoutError("Test did not release the planner")
        return actions(self.horizon, start=100 * (index + 1))

    def unblock(self):
        for event in self.release:
            event.set()


def session(planner, **kwargs):
    return RTCSession(
        planner,
        actions(),
        template=ActionStateTemplate55(),
        min_execution_steps=3,
        initial_delay_steps=2,
        **kwargs,
    )


def consume(controller, begin, end):
    return [controller.get_action(**inputs(i))[0].item() for i in range(begin, end)]


def test_partial_execution_replans_and_skips_already_consumed_prefix():
    planner = GatePlanner()
    controller = session(planner)
    try:
        assert consume(controller, 0, 4) == [0, 1, 2, 3]
        assert planner.started[0].wait(2)
        request = planner.calls[0]
        assert request["tick"] == 3
        assert request["state"][0].item() == 3
        assert torch.isfinite(request["state"]).all()
        assert request["image"].eq(3).all()
        assert [r.tick for r in request["history"]] == [0, 1, 2]
        assert [r.image[0, 0, 0].item() for r in request["history"]] == [0, 1, 2]
        assert not hasattr(request["history"][0], "state")
        assert isinstance(request["rtc"], RTCPrefix)
        torch.testing.assert_close(request["rtc"].previous_actions[:, 0], torch.arange(3, 8).float())
        assert request["rtc"].delay_steps == 2
        assert consume(controller, 4, 5) == [4]
        planner.release[0].set()
        status = controller.wait_for_inference(2)
        assert status["chunk_start_tick"] == 3
        assert status["tick"] == 5
        assert status["last_request"]["actual_delay_steps"] == 2
        assert status["published_requests"] == 1
        # The new plan began at tick3. Its outputs 0 and 1 are already consumed.
        assert consume(controller, 5, 6) == [102]
    finally:
        planner.unblock()
        controller.close()


def test_fast_result_keeps_unconsumed_committed_commands_exactly():
    planner = GatePlanner()
    controller = session(planner)
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        controller.join(2)
        # Only action3 was consumed while inference ran. Reserved action4 must
        # remain old action4, even if the planner returned an arbitrary value.
        assert consume(controller, 4, 6) == [4, 102]
        assert controller.status()["last_request"]["actual_delay_steps"] == 1
    finally:
        planner.unblock()
        controller.close()


def test_delay_estimate_updates_and_second_request_uses_new_buffer_origin():
    planner = GatePlanner(requests=2)
    controller = session(planner, delay_history_size=1)
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        controller.join(2)
        assert controller.status()["estimated_delay_steps"] == 1
        assert consume(controller, 4, 7) == [4, 102, 103]
        assert planner.started[1].wait(2)
        second = planner.calls[1]
        assert second["tick"] == 6
        assert second["rtc"].delay_steps == 1
        torch.testing.assert_close(second["rtc"].previous_actions[:, 0], torch.arange(103, 108).float())
        assert second["state"][0] == 6
        planner.release[1].set()
        status = controller.join(2)
        assert status["chunk_start_tick"] == 6
        assert status["published_requests"] == 2
        assert consume(controller, 7, 8) == [201]
    finally:
        planner.unblock()
        controller.close()


def test_balanced_deadline_publication_does_not_skip_next_planning_tick():
    planner = GatePlanner(requests=2)
    controller = RTCSession(
        planner, actions(), template=ActionStateTemplate55(), min_execution_steps=4, initial_delay_steps=4
    )
    published, release_publication = Event(), Event()

    class PublishBarrier:
        """Pause after the publish critical section, before worker cleanup."""

        def __init__(self, lock):
            self.lock = lock
            self.seen = False

        def __enter__(self):
            return self.lock.__enter__()

        def __exit__(self, *args):
            pause = not self.seen and get_ident() == controller._worker_thread and controller._published_requests == 1
            result = self.lock.__exit__(*args)
            if pause:
                self.seen = True
                published.set()
                if not release_publication.wait(10):
                    raise TimeoutError("Test did not release publication barrier")
            return result

    controller._lock = PublishBarrier(controller._lock)
    try:
        assert consume(controller, 0, 8) == list(range(8))
        assert planner.started[0].wait(2)
        planner.release[0].set()
        assert published.wait(2)
        # The new plan starts at tick4, and four actions have elapsed. Inference
        # for its successor must start at tick8, even while old cleanup is paused.
        assert consume(controller, 8, 9) == [104]
        assert controller._pending is not None and controller._pending.tick == 8
        release_publication.set()
        assert planner.started[1].wait(2)
        assert planner.calls[1]["tick"] == 8
        torch.testing.assert_close(planner.calls[1]["rtc"].previous_actions[:, 0], torch.arange(104, 108).float())
        planner.release[1].set()
        assert controller.join(2)["published_requests"] == 2
        assert consume(controller, 9, 10) == [105]
    finally:
        release_publication.set()
        planner.unblock()
        controller.close()


def test_late_plan_is_rejected_instead_of_installing_uncommitted_actions():
    planner = GatePlanner()
    controller = RTCSession(
        planner, actions(), template=ActionStateTemplate55(), min_execution_steps=3, initial_delay_steps=1
    )
    try:
        consume(controller, 0, 5)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        with pytest.raises(RTCDeadlineMiss, match="consumed 2.*reserved prefix 1"):
            controller.join(2)
        status = controller.status()
        assert status["published_requests"] == 0
        assert status["chunk_start_tick"] == 0
        assert not status["last_request"]["accepted"]
        with pytest.raises(RTCDeadlineMiss):
            controller.get_action(**inputs(5))
    finally:
        planner.unblock()
        controller.close()


def test_buffer_underrun_raises_without_repeating_the_last_action():
    planner = GatePlanner()
    controller = session(planner)
    try:
        assert consume(controller, 0, 8) == list(range(8))
        with pytest.raises(RTCBufferUnderrun, match="exhausted"):
            controller.get_action(**inputs(8))
        assert controller.status()["tick"] == 8
        assert controller.status()["remaining_actions"] == 0
        planner.release[0].set()
        with pytest.raises(RTCBufferUnderrun):
            controller.join(2)
    finally:
        planner.unblock()
        controller.close()


@pytest.mark.parametrize(
    "result", [RuntimeError("broken planner"), torch.ones(7, 55), torch.full((8, 55), float("nan"))]
)
def test_background_failures_are_reported_at_control_boundary(result):
    def planner(**kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    with session(planner) as controller:
        consume(controller, 0, 4)
        with pytest.raises(RTCPlannerError):
            controller.wait_for_inference(2)
        with pytest.raises(RTCPlannerError):
            controller.get_action(**inputs(4))


def test_reset_invalidates_running_old_episode_before_joining():
    planner = GatePlanner()
    controller = session(planner)
    finished = Event()
    thread = None
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)

        def reset():
            controller.reset(actions(start=500))
            finished.set()

        invalidated = Event()
        original_lock = controller._lock

        class ResetBarrier:
            def __enter__(self):
                return original_lock.__enter__()

            def __exit__(self, *args):
                resetting = controller._resetting
                result = original_lock.__exit__(*args)
                if resetting:
                    invalidated.set()
                return result

        # Observe episode invalidation, independently of how the old Future is
        # drained (including failed jobs whose exception must not block reset).
        controller._lock = ResetBarrier()
        thread = Thread(target=reset)
        thread.start()
        # Acquire/release is deterministic: reset changes the version while the
        # planner remains gated, so it cannot publish into the new episode.
        assert invalidated.wait(2)
        interim = controller.status()
        assert interim["resetting"]
        assert interim["revision"] == 1
        assert not finished.is_set()
        with pytest.raises(RTCError, match="resetting"):
            controller.get_action(**inputs(4))
        planner.release[0].set()
        thread.join(2)
        assert finished.is_set()
        status = controller.status()
        assert status["revision"] == 1
        assert status["published_requests"] == 0
        assert status["tick"] == 0 and status["chunk_start_tick"] == 0
        assert status["delay_history_steps"] == [2]
        assert consume(controller, 0, 1) == [500]
    finally:
        planner.unblock()
        if thread is not None:
            thread.join(2)
        controller.close()


def test_close_blocks_ticks_and_does_not_publish_old_inference():
    planner = GatePlanner()
    controller = session(planner)
    consume(controller, 0, 4)
    assert planner.started[0].wait(2)
    controller.close(wait=False)
    with pytest.raises(RTCError, match="closed"):
        controller.get_action(**inputs(4))
    planner.release[0].set()
    controller.close(wait=True)
    assert controller.status()["published_requests"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_delay_steps": 0},
        {"initial_delay_steps": True},
        {"initial_delay_steps": 5},
        {"min_execution_steps": 8},
        {"min_execution_steps": 0},
        {"delay_history_size": 0},
        {"history_steps": 0},
        {"controller_hz": 0},
        {"controller_hz": float("nan")},
        {"controller_hz": True},
        {"beta": 0},
    ],
)
def test_invalid_controller_geometry_is_rejected_before_starting_a_worker(kwargs):
    with pytest.raises(ValueError):
        RTCSession(lambda **_: actions(), actions(), template=ActionStateTemplate55(), **kwargs)


def test_action_mask_changes_require_reset_and_invalid_slots_are_zero():
    with session(lambda **_: actions()) as controller:
        first = controller.get_action(**inputs(0))
        assert first[28] == 0.5 and first[1:28].eq(0).all()
        changed = inputs(1)
        changed["action_mask"][1] = True
        changed["state_mask"][1] = True
        with pytest.raises(ValueError, match="validity cannot change"):
            controller.get_action(**changed)
        assert controller.status()["tick"] == 1


def test_planner_inputs_are_copied_before_control_caller_mutates_them():
    planner = GatePlanner()
    controller = session(planner)
    try:
        consume(controller, 0, 3)
        data = inputs(3)
        controller.get_action(**data)
        data["state"][0] = 999
        data["image"].fill_(999)
        data["action_mask"].fill_(False)
        assert planner.started[0].wait(2)
        assert planner.calls[0]["state"][0] == 3
        assert planner.calls[0]["image"].eq(3).all()
        assert planner.calls[0]["action_mask"].any()
    finally:
        planner.unblock()
        controller.close()


@pytest.mark.parametrize("with_quaternion", [False, True])
@pytest.mark.parametrize("stale_anchor", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float64])
def test_for_model_uses_fresh_anchor_and_passes_rtc_to_sampling(with_quaternion, stale_anchor, state_dtype):
    template = ActionStateTemplate55()
    calls = []
    planned_targets = []

    def batch_builder(**kwargs):
        state = kwargs["state"]
        assert state.dtype == torch.float32
        sm, am = kwargs["state_mask"], kwargs["action_mask"]
        targets = actions(32, start=state[0] + 1)
        if with_quaternion:
            targets[:, 17:21] = state[17:21]
        low, high = torch.full((55,), -100.0), torch.full((55,), 100.0)
        stats = BlockStatistics(Quantiles(low, high, sm), Quantiles(low, high, am))
        sample, metadata = build_block_sample(
            dict(
                state_trajectory=state.expand(33, -1),
                action_target=targets,
                state_mask=sm,
                action_mask=am,
                conditioning_fps=30.0,
            ),
            template=template,
            planner=SegmentPlanner(
                max_action_steps=32, overlap_action_steps=0, geometry=CausalBlockGeometry(temporal_compression_factor=4)
            ),
            history_blocks=1,
            statistics=stats,
        )
        if stale_anchor:
            metadata.anchors[0, 0] += 1e-3
        planned_targets.append(targets)
        return dict(
            sequence_plan=[SimpleNamespace(causal_action_metadata=metadata)],
            encoded=sample["action"],
            conditioning_fps_action=torch.tensor([30.0]),
            conditioning_fps=torch.tensor([7.5]),
        )

    batch_builder.controller_hz = 30.0
    batch_builder.video_stride = 4

    class Model:
        def generate_samples_from_batch(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return dict(action=[batch["encoded"]], generated_action_mask=[torch.ones(32, dtype=torch.bool)])

    def measurement(tick):
        data = inputs(tick)
        data["state"] = data["state"].to(state_dtype)
        if state_dtype == torch.float64:
            data["state"][28] = 35.123456789
            data["state_mask"][28] = True
        if with_quaternion:
            data["state"][17:21] = torch.tensor([0.2, 0.3, 0.4, 0.5])
            data["state_mask"][17:21] = True
            data["action_mask"][17:21] = True
        return data

    initial = actions(32)
    if with_quaternion:
        initial[:, 17:21] = template.sanitize(measurement(0)["state"], measurement(0)["state_mask"])[17:21]
    with RTCSession.for_model(
        Model(), batch_builder, initial, template=template, initial_delay_steps=2, history_blocks=1
    ) as controller:
        commands = []
        for i in range(17):
            data = measurement(i)
            original_state = data["state"].clone()
            commands.append(controller.get_action(**data)[0].item())
            torch.testing.assert_close(data["state"], original_state, equal_nan=True)
        assert commands == list(range(17))
        if stale_anchor:
            with pytest.raises(RTCPlannerError, match="fresh measured state"):
                controller.join(2)
            assert not calls
            return
        controller.join(2)
        assert len(calls) == 1
        batch, kwargs = calls[0]
        assert isinstance(batch["causal_action_rtc"], RTCPrefix)
        assert batch["causal_action_rtc"].delay_steps == 2
        torch.testing.assert_close(batch["causal_action_rtc"].previous_actions[:, 0], torch.arange(16, 32).float())
        assert batch["causal_action_preview"] is False
        assert kwargs["guidance"] == 1.0 and kwargs["num_steps"] == 10
        assert kwargs["causal_block_size"] == 2 and kwargs["causal_history_blocks"] == 1
        # Reserved command17 is retained; the following new action uses the
        # observation at tick16, not the bootstrap observation at tick0.
        assert controller.get_action(**measurement(17))[0].item() == 17
        assert controller.get_action(**measurement(18))[0].item() == pytest.approx(19, abs=1e-5)


def _frequency_adapter(action_fps, video_fps=7.5, stride_source="attribute"):
    template = ActionStateTemplate55()
    calls = []

    def builder(**kwargs):
        state, sm, am = kwargs["state"], kwargs["state_mask"], kwargs["action_mask"]
        low, high = torch.full((55,), -100.0), torch.full((55,), 100.0)
        provenance = {"geometry": {"video_stride": 4}} if stride_source == "statistics" else {}
        stats = BlockStatistics(Quantiles(low, high, sm), Quantiles(low, high, am), provenance)
        metadata = SimpleNamespace(
            states=state[None].clone(),
            anchors=state[None].clone(),
            state_mask=sm[None].clone(),
            action_mask=am.expand(8, -1).clone(),
            action_frame_ids=torch.arange(1, 3).repeat_interleave(4),
            block_size=2,
            template=template,
            statistics=stats,
        )
        batch = dict(
            sequence_plan=[SimpleNamespace(causal_action_metadata=metadata)],
            conditioning_fps=video_fps,
        )
        if action_fps is not None:
            batch["conditioning_fps_action"] = action_fps
        if stride_source == "batch":
            batch["video_stride"] = 4
        return batch

    if stride_source == "attribute":
        builder.video_stride = 4
    elif stride_source == "geometry":
        builder.geometry = CausalBlockGeometry(temporal_compression_factor=4)

    class Model:
        def generate_samples_from_batch(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return dict(action=[torch.zeros(8, 55)], generated_action_mask=[torch.ones(8, dtype=torch.bool)])

    return Model(), builder, calls


@pytest.mark.parametrize(
    "fps", [30.0, torch.tensor(30.0), torch.tensor([30.0]), torch.tensor([30.0], dtype=torch.float64)]
)
def test_for_model_accepts_scalar_and_single_tensor_control_frequencies(fps):
    model, builder, calls = _frequency_adapter(fps)
    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3, initial_delay_steps=2
    ) as controller:
        consume(controller, 0, 4)
        controller.join(2)
        assert len(calls) == 1


@pytest.mark.parametrize(
    "fps", [25.0, torch.tensor([25.0]), None, True, torch.tensor(True), torch.tensor([30.0, 30.0]), float("nan")]
)
def test_for_model_rejects_missing_invalid_or_mismatched_action_frequency(fps):
    model, builder, calls = _frequency_adapter(fps)
    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3, initial_delay_steps=2
    ) as controller:
        consume(controller, 0, 4)
        with pytest.raises(RTCPlannerError, match="conditioning_fps_action"):
            controller.join(2)
        assert not calls


@pytest.mark.parametrize("stride_source", ["attribute", "geometry", "statistics", "batch"])
def test_for_model_checks_visual_frequency_against_available_stride(stride_source):
    model, builder, calls = _frequency_adapter(30.0, video_fps=torch.tensor([6.25]), stride_source=stride_source)
    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3, initial_delay_steps=2
    ) as controller:
        consume(controller, 0, 4)
        with pytest.raises(RTCPlannerError, match="conditioning_fps times video_stride"):
            controller.join(2)
        assert not calls


def test_for_model_rejects_disagreeing_stride_declarations():
    model, builder, calls = _frequency_adapter(30.0)
    builder.geometry = SimpleNamespace(video_stride=2)
    with RTCSession.for_model(
        model, builder, actions(), template=ActionStateTemplate55(), min_execution_steps=3, initial_delay_steps=2
    ) as controller:
        consume(controller, 0, 4)
        with pytest.raises(RTCPlannerError, match="video_stride declarations disagree"):
            controller.join(2)
        assert not calls


def test_for_model_rejects_advertised_builder_frequency_before_starting_worker():
    model, builder, calls = _frequency_adapter(25.0)
    builder.controller_hz = 25.0
    with pytest.raises(ValueError, match="batch_builder.controller_hz"):
        RTCSession.for_model(model, builder, actions(), template=ActionStateTemplate55(), controller_hz=30.0)
    assert not calls
