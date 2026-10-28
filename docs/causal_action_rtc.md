# Training-free RTC for causal action inference

RTC is opt-in. Ordinary inference keeps CFG=1, ten denoising steps and the existing
request-local cache defaults. No checkpoint parameters or training losses change.
The implementation follows the soft action-prefix masking and input-VJP guidance
in [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339v1),
Equations 2–5. It integrates the guided velocity with the checkpoint's native
UniPC or fixed-step sampler, rather than replacing it with the paper's Euler loop.

## Network design principles

The supported RTC path preserves the causal attention visibility, measured-state
anchors, joint video/action network, checkpoint weights and CFG branch formula.
Previous predicted actions provide an external guidance target; they are not
inserted into historical attention or supplied as clean current-block actions.
The gradient-compatible prefix cache validates that fixed K/V dependencies do not
read current noisy inputs, and rebuilds the differentiable current-block graph
on every evaluation, including the first one.

RTC deliberately changes the sampling velocity and preserves commands already
committed to execution. Its output distribution therefore need not equal ordinary
inference. Numerical differences between cached and reference RTC execution, and
repeated-run variation, also remain possible. Preserving these architectural
principles does not establish lossless task accuracy or closed-loop success.

In particular, the first committed action rows are restored after sampling and
again at physical export. Video latents are not regenerated after this final
action projection, so the exported video/action pair is not guaranteed to be an
unchanged draw from the original joint distribution or to remain strictly
consistent after projection. This implements the execution constraint that
committed commands cannot be withdrawn. Unclipped prefix normalization also
applies only to the external guidance target, which can lie outside the training
clipping range; it does not replace the network's original input statistics.

## Components

- `cosmos_framework/inference/causal_action/rtc.py`: physical action prefix,
  new-anchor encoding, soft weights and joint video/action input guidance.
- `cosmos_framework/inference/causal_action/rtc_prefix_cache.py`: request-local
  fixed-prefix caching with a fresh differentiable current-block graph per step.
- `cosmos_framework/inference/causal_action/rtc_session.py`: one background
  inference worker and an action buffer aligned by absolute controller ticks.
- `cosmos_framework/inference/causal_action/rtc_calibration.py`: synchronized
  startup latency measurements and conversion to a controller-step budget.
- `cosmos_framework/inference/causal_action/rtc_batch.py`: constructs a model
  batch from actual observed images and a fresh measured state.
- `cosmos_framework/inference/causal_action/cli.py`: offline guided inference
  from an explicitly supplied remaining action prefix.

## Prefix contract and numerical conventions

`RTCPrefix.previous_actions` has shape `[P, template.width]`, in absolute physical
action units. It starts at the current observation time: remove already executed
commands from the preceding prediction before supplying it. `delay_steps=d` is
the number of commands committed while the next inference is running. The prefix
length is `P=H-s`, where `H` is the prediction horizon and `s` is the number of
commands executed before starting this inference.

The first `d` prefix weights equal one. The rest of the overlap receives the
paper's decaying soft weights; positions after the overlap have zero weight.
The valid action mask removes unused template channels.

Old absolute targets are encoded again against the **current measured anchor**,
then normalized with the same action statistics as the model. Prefix normalization
does not clip quantile tails. Constant statistics channels must match their one
representable value. Export restores the original physical first `d` commands,
including quaternion components, rather than relying on a normalization round-trip.

Cosmos integrates noise sigma from one to zero. Its clean estimate is
`x_clean = x - sigma * velocity`; RTC therefore subtracts the clipped VJP correction
from the Cosmos velocity. The error is action-only, but its Jacobian is taken
with respect to the complete current video/action input, so video latents may
also change. Only input gradients are requested; model parameter gradients are
not accumulated.

Use `torch.no_grad()`, not `torch.inference_mode()`. RTC locally enables input
autograd. Native samplers pass their actual integration sigma separately from
the potentially quantized network timestep. A custom sampler must call
`velocity_fn.set_integration_sigma(sigma)` immediately before every guided
velocity evaluation.

## Cache behavior

With `cache_mode="full"`, KV caching enabled and frozen eval-mode model weights,
the supported dense, default-RoPE, unsharded grouped-TND model uses optimization
①'s layout/index plans together with a separate, gradient-safe version of ②.
The first velocity evaluation builds the fixed text/visual/state prefix under
`no_grad`, then runs the current video/action tokens with input gradients. Every
subsequent evaluation builds a fresh current-block graph while reusing only fixed
prefix K/V. K/V assembly and output scattering are functional: current values and
autograd graphs are never written into the persistent prefix buffers.

This is request-local reuse. Fixed inputs and positions are checked, CFG branches
are isolated, and a new request gets a new prefix. Input VJPs are supported;
parameter training through detached fixed-prefix computations is not supported.
Freeze inference weights explicitly with `model.net.eval().requires_grad_(False)`.
Eval callers whose parameters still require gradients retain the uncached RTC
path, with the reason recorded in `result["rtc"]["cache_fallback_reason"]`.
Unsupported language models, RoPE variants and MoE configurations also retain the
reference path instead of enabling the new cache.

Use `--cache-mode plans` or `batch["causal_action_cache_mode"]="plans"` for the
reference RTC path: optimization ① remains enabled and every velocity evaluation
uses a complete differentiable forward. `--cache-mode off` also disables cached
index plans. `--no-kv-cache` disables the new numerical prefix cache regardless
of the requested cache mode. These reference options permit paired output/VJP
checks and do not change the RTC guidance formula.

`result["rtc"]` and `result["inference_cache"]` record whether the gradient-safe
prefix is active. The original local attention cache is not used for RTC input
VJPs. The original non-RTC implementation of ② remains unchanged.

An ordinary request without `batch["causal_action_rtc"]` continues to use the
existing B path. RTC changes the sampling algorithm; it does not add old action
tokens to the model's historical attention inputs. Cross-request visual caching
from the C experiment is not enabled by this implementation.

## Offline guided inference

Create `remaining_prefix.json` as an array of absolute `[P,D]` action values, or
as `{"previous_actions": [...]}`. A tensor `.pt` file with the same contents is
also accepted using `weights_only=True` loading.

Add these flags to the existing causal-action CLI command:

```bash
python -m cosmos_framework.inference.causal_action.cli \
  --checkpoint /path/to/checkpoint \
  --sources-file /path/to/sources.json \
  --processor /path/to/processor \
  --vae /path/to/vae.pth \
  --output /path/to/result \
  --mode policy --steps 10 --guidance 1 --enable_rtc \
  --rtc-prefix remaining_prefix.json --rtc-delay-steps 4 --rtc-beta 5
```

`--enable_rtc` (alias `--enable-rtc`) selects RTC guidance. With the existing
defaults of CFG=1, ten steps, `--cache-mode full` and KV caching enabled, the
supported model uses the `rtc_enabled` configuration: RTC plus compatible ②.
Explicit `plans`/`off`/`--no-kv-cache` options still permit reference comparisons;
the cache diagnostics report what actually ran.

Without a prefix or the flag, ordinary inference remains the default. For backward
compatibility, the existing paired `--rtc-prefix` / `--rtc-delay-steps` arguments
also enable RTC when the new flag is omitted. `--no-enable_rtc` (alias
`--no-enable-rtc`) explicitly disables it; combining that with a prefix is an
error. Explicit enabling without a prefix and delay is also an error. The flag
does not generate a previous prediction: obtain the bootstrap chunk through
ordinary inference, then pass its remaining absolute actions.

The example launcher `examples/action_pretrain/inference_causal_action.sh` forwards
these flags unchanged. Python callers retain the existing explicit `RTCPrefix`
and `RTCSession.for_model` interfaces.

`sample.pt` includes physical actions, video latents, the effective `rtc_enabled`
boolean, RTC diagnostics and cache diagnostics. `metrics.json` records the same
boolean and distinguishes requested and effective KV caching.
This command validates one guided prediction; it does not run a robot controller.

## Asynchronous execution adapter

The execution process owns the clock and hardware. `RTCSession` neither sleeps
to simulate control periods nor sends commands to a robot. It returns one CPU
absolute action for each call to `get_action`. Startup calibration and ordinary
fallback can block; only the active RTC mode plans asynchronously.

```python
from cosmos_framework.inference.causal_action.rtc_batch import RTCObservedBatchBuilder
from cosmos_framework.inference.causal_action.rtc_session import RTCSession

# prototype_batch supplies only the template, normalization statistics, prompt
# and image geometry. Its stored state/action/future pixels are not observations.
builder = RTCObservedBatchBuilder(prototype_batch, controller_hz=30.0, history_blocks=8)
model.net.eval().requires_grad_(False)

# Obtain initial_actions once from ordinary model inference at the current state.
with RTCSession.for_model(
    model,
    builder,
    initial_actions,
    template=template,
    controller_hz=30.0,
    min_execution_steps=16,
    initial_delay_steps=None,  # Default: measure actual full RTC requests.
    history_blocks=8,
    beta=5.0,
) as session:
    # Complete this before enabling the external command clock/actuators.
    session.prepare(
        state=initial_measured_state,
        state_mask=state_mask,
        image=initial_image,
        action_mask=action_mask,
    )
    print(session.status())  # Check rtc_enabled / execution_mode / calibration.

    # Exactly once per external control tick, before consuming that tick's command:
    command = session.get_action(
        state=measured_state,
        state_mask=state_mask,
        image=current_image,
        action_mask=action_mask,
    )
    # The caller's hardware adapter consumes command and supplies the next measurement.
```

`for_model` defaults to `initial_delay_steps=None`. `prepare` performs one warmup
and two timed full RTC requests using the actual supplied observation and a
time-aligned bootstrap prefix. The measured path includes batch construction,
VAE, guided sampling, physical export and transfer to CPU. Devices synchronize
around every measurement. The initial budget is
`max(1, ceil(max(sample_seconds) * execution_hz) + 1)`; `execution_hz` defaults to
`controller_hz`. The extra controller step
is a scheduling margin. Warmups, samples and margin are configurable through
`calibration_warmups`, `calibration_samples` and `calibration_margin_steps`.

Calibration outputs are discarded: no command is published or consumed, and the
control tick does not advance. This wall-clock measurement is not inferred from
a paused execution clock. If `prepare` is omitted, the first `get_action` performs
the same initialization and therefore blocks. Use explicit `prepare` before the
external control clock starts. An explicit integer `initial_delay_steps` skips
automatic measurement. The low-level `RTCSession(planner, ...)` keeps its manual
default of one step; use `None` to request measurement and supply an
`ordinary_planner` to permit non-RTC fallback.

The startup measurement covers the history actually available at startup. It
does not fabricate future observations to predict full-window latency. Runtime
deadline checks remain active as the history grows.

## Client execution speed and model time

The model action-time grid and the client's wall-clock execution pace can differ.
Keep `controller_hz=30` and the builder's action/video FPS unchanged for this
checkpoint. Set `execution_hz` to the number of model waypoints the client
actually consumes per wall-clock second. This changes latency budgeting and
diagnostics; the session itself neither paces the client nor changes a robot's
speed.

For example, `execution_hz=3` stretches 16 model waypoints across about 5.33 real
seconds, while they still represent 0.533 model seconds. A client can maintain
30 Hz servo updates and interpolate between these slower waypoints. With a full
RTC request latency `L`, the budget condition becomes
`ceil(L * execution_hz) + margin_steps <= min(s, H-s)`.
This can keep RTC active even when executing every model waypoint at 30 Hz would
force fallback. It lengthens physical execution; it does not reduce inference
latency or establish that the model's task behavior is invariant to slower motion.

Pass both rates explicitly when using a slower client:

```python
session = RTCSession.for_model(
    model, builder, initial_actions,
    template=template,
    controller_hz=30.0,  # Model action-time grid; batch FPS remains 30.
    execution_hz=3.0,   # Client consumes three model waypoints per real second.
    min_execution_steps=16,
    calibration_margin_steps=5,  # Example headroom for history growth/jitter.
)
```

`status()` and the calibration report expose `controller_hz`, `execution_hz` and
`playback_speed=execution_hz/controller_hz`. The actual runtime delay still counts
consumed waypoints, so merely declaring a slower rate without actually pacing
the client will not bypass deadline checks. A request's committed prefix and
delay remain fixed after submission. The client may stretch execution further;
it must not reinterpret those already submitted action indexes.

`rtc_client_simulation.replay_rtc_client` provides a paced offline client with
servo interpolation while the normal RTC background worker computes. Prepare
the session before entering the timed loop. The runner never joins a pending
request inside that loop and rejects non-RTC fallback as a successful continuous
RTC demonstration. It records waypoint/servo timestamps, lateness, publications
and outputs. `interpolate_action` handles scalar channels linearly and valid
quaternion groups with shortest-sign normalized interpolation.

Recorded image/state replay at this slower pace is a time-stretched observation
stream. It does not simulate the environment responding to predicted actions,
real robot dynamics or network transport. Successful replay demonstrates
continuous scheduling/buffering; it does not measure closed-loop task success.

Images must be real CPU uint8 observations, already concatenated/resized to the
prototype's image shape. The builder selects actual image ticks relative to the
latest observation, appends zero future placeholders and zeros action targets.
It reconstructs the current block anchor from the fresh state, including requests
started partway through a preceding prediction. Missing required observations
are errors; the prototype's stored pixels and future states are never substituted.
Sensor states are canonicalized to FP32 before model batch construction. The model
adapter also checks that action FPS matches the execution clock and, when the
video stride is known, that video FPS describes the same time grid.

For a custom `batch_builder`, its keyword inputs are `state`, `state_mask`, `image`,
`action_mask`, `history`, `tick`, `revision` and `rtc`. History contains real
`RTCObservation(tick, image)` records only. The batch must end its observed history
at `tick`, carry matching template/statistics, and preserve the fresh anchor and
state mask. Historical state/action slots may be zero placeholders. The model
adapter sets the non-preview current-block mode and supplies the prefix.

## Adaptive execution as history grows

`RTCExecutionPolicy` optionally reduces the client's waypoint rate before a new
request. The default remains fixed-rate execution. Model FPS, network weights,
the attention plan cache and the RTC-compatible prefix cache stay unchanged.

The session maintains a bounded buffer of pairs: observed history length in model
seconds and the measured request latency in wall seconds. `delay_history_size`
sets its capacity (default eight requests). Each completed request is tagged with
the history supplied at submission, including the window cap, rather than the
history available when inference finishes.

On the CPU, the policy fits a simple straight line to those recent pairs. Its
coefficients are recomputed from the buffer; no checkpoint-specific coefficients,
offline latency file or predetermined slowdown schedule are required. Let `T` be
the next request's model-history seconds, `r` the client's waypoint rate and `m`
an integer margin:

```text
L_predict(T) = a + b*T                  # a, b fitted from recent actual requests
L_budget = max(L_predict(T) + positive_fit_residual, recent_measured_max)
           * latency_factor + extra_latency_seconds
D = min(s, H-s)
r_next <= min(r_current, (D-m) / L_budget)
d = ceil(L_budget * r_next) + m <= D
```

At least two distinct history lengths are required to estimate a slope. Startup
and same-length-only buffers use the maximum measured latency as a constant
forecast. Negative fitted slopes are clamped to zero, and forecasts are guarded
by the recent maximum and positive fitting residuals. The implementation uses
Python scalar arithmetic; it does not require NumPy, SciPy or accelerator work.

With adaptive execution enabled, startup calibration supplies the first paired
samples before commands begin, including when a manual `initial_delay_steps` was
passed. Warmup measurements are excluded. Subsequent completed requests evict the
oldest samples and automatically update the fit. A missing measurement cannot be
replaced by a hardcoded nominal latency.

For finite latency and `D > m`, a sufficiently small positive rate satisfies this
inequality unless the client imposes a minimum rate. This is a conditional timing
budget, not a worst-case latency guarantee. Fit residuals, jitter and unmeasured
transport can exceed an empirical estimate; actual tick-based deadline checks
continue to reject late results. Slowing execution increases the time available
to finish inference; it does not reduce inference computation or establish task
success under different robot dynamics.

The session anticipates the next request's complete history blocks. It uses the
online forecast and startup/runtime wall-time measurements, then obtains client
acknowledgment before installing a slower rate. The callback must return the
literal `True` after the client commits the new waypoint schedule. Rejection or
an exception selects the existing warning/fallback path. With no ordinary planner,
adaptive infeasibility raises a latched `RTCDeadlineMiss` until reset.
The callback is synchronous: a network acknowledgment also takes wall time and
must fit the client's command-delivery schedule. The replay's lateness diagnostics
do not hide pauses caused by acknowledgment or observation transport.
The session never increases the rate or changes a submitted request's prefix or
rate while it is in flight. A missed request cannot be rescued retroactively.

Both the builder and session must permit 32 history blocks to cover the previously
discussed 34.133 model seconds. Their default of eight blocks covers 8.533 model
seconds. Enable the online policy and provide an adapter callback that actually
controls the client:

```python
from cosmos_framework.inference.causal_action.rtc_pacing import RTCExecutionPolicy

policy = RTCExecutionPolicy(latency_factor=1.2, margin_steps=1)
builder = RTCObservedBatchBuilder(
    prototype_batch, controller_hz=30.0, history_blocks=32,
)
session = RTCSession.for_model(
    model, builder, initial_actions,
    template=template,
    controller_hz=30.0,
    execution_hz=30.0,  # Desired starting rate; calibration may reduce it.
    min_execution_steps=16,
    history_blocks=32,
    delay_history_size=8,
    execution_policy=policy,
    execution_rate_callback=apply_client_rate_and_wait_for_ack,
)
session.prepare(**first_measurement)
```

`apply_client_rate_and_wait_for_ack` is the deployment adapter's responsibility;
it is not a bundled robot RPC client. For `replay_rtc_client` only, a callback
returning `True` can acknowledge the runner's own schedule: the runner reads each
new session rate after `get_action`, applies it to future intervals and retains
the independent servo clock. It records every interval's `execution_hz`, plus the
initial and final rates. A real client must implement the same pacing contract.

The policy parameters are `latency_factor` (default 1.2), `extra_latency_seconds`
(default zero), `margin_steps` (default one) and `minimum_execution_hz` (default
zero, no lower bound). Neither a fixed fit domain nor a minimum tenfold slowdown
is imposed: the next rate follows the measurements and available prefix budget.
Adaptive reservations use `policy.margin_steps`; `calibration_margin_steps` still
controls fixed-rate calibration but does not override the policy's margin.
Account for sensor processing and network transport in `extra_latency_seconds`
when they were excluded from the measurement. Reset clears latency history and
recalibrates adaptive sessions, but retains an already acknowledged slower rate;
starting a new episode does not silently accelerate the client.

When execution rates vary, wall duration is the sum of each executed interval
`1/r_tick`. A 34.133-model-second history therefore takes progressively longer
to accumulate; it is not a fixed 34.133 seconds on the robot's wall clock. The
latest measured image/state still refresh on each request, while historical image
selection remains aligned to the model action ticks.

`status()` exposes the paired latency buffer and current fit diagnostics alongside
the predicted budget and acknowledged execution rate. This makes the predicted
latency inspectable while the robot runs. A rolling linear forecast is intended
for the next request; it does not promise an accurate prediction far into the
future or protect against an unobserved latency jump.

### Offline timing evidence

On 2026-10-28, the local Ascend host was measured with the 0930 checkpoint at
iteration 20,000, CFG=1, ten steps, beta=5 and caches ①＋②. All 33 history lengths
from zero to 34.13 model seconds had a per-shape warmup and three timed d=16
requests each. Another 63 timed requests checked independent episodes, device
drift and d=4/8/12 sensitivity. These measurements are reference evidence, not
coefficients loaded by the online policy.

The timer includes observed-batch construction, VAE encoding, RTC forward/input
backward, sampling and physical CPU action export. It excludes file/sensor
decoding, history snapshot/IPC assembly, model loading and robot/network transport.
Three otherwise-idle NPU devices handled disjoint history ranges; a fourth
handled episode/device checks. The host CPU and filesystem were shared.

Measured median latency was about 2 seconds with no history, 3 seconds at 5 model
seconds, 6 seconds at 16 model seconds and 11 seconds at 34 model seconds. Online
forecasting learns the current deployment's own relationship from its buffer;
different hardware, host load or network costs can produce different rates.

A separate discrete-event replay drives the actual `RTCSession` worker using
those recorded durations as the environment's request times. The online policy
only receives startup measurements and already completed requests; it cannot
read the recorded future durations. All three repeat sequences and a per-length
maximum sequence completed 1,120 waypoints each with adaptive starting targets
of either 3 or 30 Hz. Each run accepted 69 requests, including six with a full
34-second model-history window. From a 30 Hz target, startup calibration reduced
execution to about 6 Hz, followed by online adjustments to about 1 Hz at the
longest history. Fixed-3-Hz comparisons retaining the maximum 16-step prefix
instead exhausted their buffers. These are scheduling replays with synthetic
state/image/actions, not new NPU inferences or physical closed-loop robot tests.

## Deadlines, resets and current limits

The controller uses actual commands consumed during inference to align the new
prediction. It discards the new chunk's already elapsed positions and publishes
the remaining commands atomically. A single worker serializes model use; do not
share that model concurrently with another inference process in the same session.

The initial delay estimate is updated from a bounded history of observed delays.
The execution geometry must satisfy `d <= s <= H-d`. With the model adapter,
infeasible startup timing, insufficient remaining actions or a runtime deadline
miss emit `RTCFallbackWarning` and switch to ordinary chunk inference.
`status()` records `rtc_enabled=False`, `execution_mode="blocking_non_rtc"`,
the calibration measurements and `fallback_reason`.

After a startup fallback, the existing bootstrap chunk is consumed normally.
At each chunk boundary, ordinary inference uses the latest supplied observation
and blocks until its new chunk is ready. A runtime-overdue RTC prediction is
discarded, and ordinary replanning starts on the next supplied observation.
The old worker is drained before ordinary inference so model patches are never
used concurrently. The ordinary batch has no `causal_action_rtc` field: it uses
the original B sampling/cache path, with no RTC VJP or committed-prefix restore.

Blocking fallback allows gaps between action chunks; it does not maintain a
continuous 30 Hz command stream and does not replay a stale command to conceal
the gap. The external hardware adapter owns the behavior while waiting. A custom
low-level session without `ordinary_planner` retains `RTCDeadlineMiss` and
`RTCBufferUnderrun` instead of inventing an ordinary policy. Input validation and
model failures still surface as errors, rather than being hidden by fallback.

`reset(initial_actions)` invalidates the old revision and drains pending inference
before a new episode. Automatically calibrated sessions measure again at the next
`prepare`/first command; `close()` releases the worker.

With the current `H=32` and 30 Hz action rate, the whole prediction covers about
1.067 seconds. At a balanced execution horizon of 16, the guided inference must
fit in roughly 0.533 seconds. Algorithm integration does not establish that timing
condition. Earlier A/B latency measurements exclude RTC VJP computation and cannot
be used as RTC latency guarantees. Do not lower the controller frequency without
an explicit adapter/control-policy decision.

For `K` guided evaluations, the new cache runs one full no-grad prefill plus `K`
compact forwards and `K` input backward passes. It saves repeated fixed-prefix
work; shorter histories are not guaranteed to become faster. Measure the complete
observed-input-to-physical-action latency when calibrating the controller.

CPU tests cover guidance derivatives, masks, physical-prefix preservation,
request-local cache isolation, exact sigma notifications, asynchronous alignment,
late results, exhaustion and resets. Real-checkpoint offline tests validate finite
full guided chunks and default-path restoration; they do not validate closed-loop
task success or certify worst-case latency.

## Recorded-observation validation

The 0930 checkpoint at iteration 20,000 was evaluated with CFG=1 and ten steps
on 20 task-376 episodes and three seeds. Static comparisons cover 240 windows
with 0, 5.333, 16 and 34.133 model seconds of history. Executed-action comparisons
cover 60 recorded-observation rollouts, each consuming 112 waypoints. The two
asynchronous policies use the same virtual response delay of 12 waypoints at
3 waypoints/s; this is a latency-matched comparison, not another live robot run.

After excluding the actual shared first 28 waypoints, the mean reference errors
over the remaining 84 waypoints are:

| Metric                    | Asynchronous without RTC | RTC      | Relative change |
| ------------------------- | ------------------------ | -------- | --------------- |
| Joint MAE (rad)           | 0.041384                 | 0.041415 | +0.075%         |
| EEF position L2 (mm)      | 18.6593                  | 18.6785  | +0.103%         |
| EEF rotation angle (deg)  | 3.26362                  | 3.24916  | -0.443%         |
| Gripper MAE (source unit) | 0.744677                 | 0.742567 | -0.283%         |

All four paired, episode-bootstrap 95% intervals include zero; these results do
not establish strict equivalence or lossless accuracy. At publication boundaries,
mean joint command jumps decrease by 4.16% (19 of 20 episodes improve), and EEF
rotation jumps decrease by 2.67%. These are command differences, not measured
robot dynamics. In the static 16-second-history comparison, uncommitted joint and
position errors increase by 1.81% and 1.07%; their paired intervals exclude zero.

References follow the reader's next-state target, `target[t]=state[t+1]`.
Grippers use the state source's native units. The selected episodes are not held
out from the fitted statistics population, and the exact checkpoint training
statistics remain unconfirmed. Holding the current state can outperform both
models on low-motion windows. Recorded observations do not respond to predictions,
so neither these errors nor the continuity changes establish task success.
