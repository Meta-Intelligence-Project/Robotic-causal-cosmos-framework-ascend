# Causal action client/server with RTC

The `rtc` branch provides a WebSocket model server and a RoboTwin client adapter.
The transport and native robot conversion are adapted from
`feat/robotwin-client-server`; inference uses the current RTC branch's shared
model planner, caches ①＋② and optional RTC guidance.

The protocol is `cosmos-causal-action-rtc-v1`. Run matching client/server code from
this branch. The older synchronous `cosmos-causal-action-v2` client is a different
protocol and fails the handshake rather than silently changing execution.

Choose one execution mode and use the matching client from this checkout:

| Mode                  | Server flag         | Client flag    | Returned action shape | Execution                                            |
| --------------------- | ------------------- | -------------- | --------------------- | ---------------------------------------------------- |
| Ordinary baseline     | None                | None           | `[32,55]`             | Execute 16, then wait for a fresh prediction         |
| Wall-clock RTC        | `--enable_rtc`      | `--enable_rtc` | `[32,55]`             | Continue old actions during a background RPC         |
| Simulated RTC         | `--simulation-sync` | None required  | `[16,55]`             | Execute a delay-dependent mixture of old/new actions |

## Compatibility with the client/server source branch

The RoboTwin environment setup, camera/state helpers and native robot adapter
were adapted from `feat/robotwin-client-server`. Its complete client is not a
drop-in replacement for this branch. The comparison below uses the locally
available source-branch commit `b4e90f0278d9797343ed79fce25bdbb0780970c0`.

| Interface           | Source branch client                                 | Current simulated RTC client                          |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| Protocol            | `cosmos-causal-action-v2`                            | `cosmos-causal-action-rtc-v1`                         |
| Observation request | `obs`: raw three-camera dictionaries, native state   | Prepared CHW image, unified55 state, explicit ticks   |
| Executed feedback   | 16 camera sets, `executed_steps`, `observation_step` | 16 `executed_observations`, `tick`, `simulation_rtc`  |
| Actions on the wire | 32 native joint/EEF rows; execute the first 16       | 16 unified55 rows; convert before robot execution     |
| State default       | `command`                                            | `measured`                                            |
| Execution schedule  | One `take_action` per received target                | Calibration, online fit and logical servo subdivision |

The source client's `take_action` followed by `get_obs` is the feedback rule
retained here: return observations of the actions actually executed. Its RPC
wrapper and execution loop still need adaptation; changing only a server URI
or the protocol string is insufficient. The current client also needs CPU Torch
for preprocessing and sessions. Source-branch prediction-video and
`finish_video` interfaces are not supplied by this server.

## Ownership and execution

The server loads the checkpoint, VAE, processor and training statistics. The
client owns the action buffer, observation ticks, servo schedule and online
latency fit. With RTC enabled, the existing `RTCSession` runs on the client CPU;
its background planner performs a blocking RPC while the foreground continues
executing the previous chunk.

In simulated RTC, the server additionally retains its full 32-action prediction
and supplies only 16 execution actions per round. The client owns the latency
samples, rate acknowledgement and real executed-observation batch. This mode
uses `SimulatedRTCSession`; the following background-RPC diagram describes
wall-clock `RTCSession` execution.

```text
Client                                      Server
fresh observation ------------------------> ordinary bootstrap inference
             <----------------------------- initial 32-action chunk
remain stationary; prepare() -------------> RTC warmup and timed requests
apply calibrated execution rate
execute old actions; background RPC ------> current RTC + compatible prefix cache
continue old actions <--------------------- new chunk, aligned to observation tick
```

The first chunk is generated without RTC. Startup calibration uses complete RTC
RPCs, discards their predictions and does not advance the execution tick. Only
after `prepare()` finishes should the execution adapter start moving the robot.
At `s=16`, the client requests the next `H=32` chunk in the background. Publication,
committed-prefix restoration, elapsed-action skipping, deadline checks and
fallback use the existing session logic.

Outside simulation mode, `--enable_rtc` (alias `--enable-rtc`) is opt-in on both
ends. Without it, the client
performs the ordinary baseline: predict 32 actions, execute 16, then block for a
fresh prediction. With RTC, adaptive execution is enabled by default in the client
adapter; `--no-adaptive-execution` permits fixed-rate comparisons. Adaptive
execution fits recent `(history length, completed RPC latency)` pairs on the CPU.
No offline latency coefficients or fixed tenfold slowdown are required.

## Start the model server

Use the existing Cosmos model environment with NPU support. The model configuration
defaults to CFG=1, ten denoise steps, full prefix caching, 32 history blocks and
the checkpoint's 30 Hz action-time grid. Thirty-two history blocks cover about
34 model seconds; `--history` changes that capacity.

The training statistics and robot coordinate conventions must match the checkpoint.
For robots whose DOFs, masks or EEF frame cannot be determined unambiguously from
the statistics, provide `--robot-config`. Seven-joint Agibot data must not be
treated as the six-joint RoboTwin default. EEF native poses use world coordinates
and `wxyz`; the adapter handles the configured model frame and `xyzw` template.

```bash
ASCEND_RT_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.inference.causal_action.server \
  --checkpoint /path/to/trained/checkpoint \
  --processor /path/to/Cosmos3-Edge \
  --vae /path/to/Wan2.2_VAE.pth \
  --statistics /path/to/training_statistics.json \
  --robot-config /path/to/robot_config.json \
  --host 0.0.0.0 --port 8000 \
  --enable_rtc
```

The model server does not require a training dataset or future ground-truth
observations. Reset constructs an input schema from the initial real image,
current measured state, task prompt and supplied statistics. All future image
and action placeholders are zero. Historical observations come exclusively from
the client. Camera composition and resizing reuse the training pipeline.

The service supports one model rank and one client connection. Model calls run
serially in a worker outside the WebSocket event loop. An old request must finish
before disconnect cleanup releases the model to another client. Model weights
are frozen; RTC input gradients remain enabled where needed. There is no enclosing
`torch.inference_mode()` around guided inference.

The configurable launcher is
`examples/action_pretrain/robotwin_server_template.sh`. Set its checkpoint,
statistics, processor, VAE, environment and device paths for the server machine.
`ENABLE_RTC=1` enables RTC; `PRINT_ONLY=1` prints the command without activating an
environment or launching the model.

## RTC simulation with 16-step observation batches

For RoboTwin evaluations that emulate inference delay without waiting for that
delay in simulation time, start the server with `--simulation-sync` (alias
`--simulation_sync`). This flag enables simulated RTC automatically. The matching
client reads the setting from the handshake; it needs no additional simulation
flag. The ordinary hardware RTC path remains available through `--enable_rtc`.
This mode requires `--execute-steps 16` and adaptive execution; passing
`--no-adaptive-execution` to the client is rejected. An explicit client
`--enable_rtc` is optional and does not select the wall-clock session when the
server advertises `simulation_sync`.

```bash
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.inference.causal_action.server \
  --checkpoint /path/to/trained/checkpoint \
  --processor /path/to/Cosmos3-Edge \
  --vae /path/to/Wan2.2_VAE.pth \
  --statistics /path/to/training_statistics.json \
  --robot-config /path/to/robot_config.json \
  --simulation-sync
```

Run the matching client in the RoboTwin environment:

```bash
python examples/action_pretrain/robotwin_client.py \
  --robotwin-root /path/to/compatible/RoboTwin \
  --server-uri ws://SERVER_IP:8000 \
  --task-name adjust_bottle --task-config demo_clean \
  --control-mode joint --state-source measured \
  --initial-execution-hz 30 --servo-hz 30 \
  --max-waypoints 48 \
  --output outputs/causal_action/robotwin_rtc_sim
```

For programmatic clients, `create_session` needs an `execution_rate_callback`
that actually applies the accepted rate and returns `True`. The CLI adapter
provides it and supplies `simulation_servo_hz` for integer servo subdivision.
The source-branch client does not implement this session contract.

The server predicts a full 32-action chunk using caches ①＋② and returns a
16-action execution batch. The first prediction is ordinary inference. Before
motion, the client performs one RTC warmup and two timed RTC requests, discarding
their predictions. Calibration does not advance observation ticks or replace
the bootstrap prediction retained by the server.

The client retains the last eight completed `(history model seconds, full RPC
seconds)` measurements and uses the existing CPU online latency fit to choose
the next execution rate and committed prefix `d`. Predictions use completed
measurements only. The rate is rounded down to an integer servo subdivision,
and the client acknowledges and applies it to the upcoming waypoint schedule.
The request freezes that rate and prefix before the server starts inference.

With a predicted latency budget `L`, execution rate `r` and margin `m`, the
reservation is `d = ceil(L*r) + m`. After measuring the completed RPC, the
client checks `q = ceil(actual_RPC_seconds*r)` against `q <= d <= 16`.
`d` is the committed prefix; `q` is the measured latency expressed in waypoint
steps. They are recorded separately. The completed request only becomes a
latency sample for subsequent decisions; it cannot determine its own predicted
reservation. History length remains `min(floor(tick/32), history_blocks)*32/fps`
model seconds, regardless of the slower execution rate.

For a request at tick `t`, the server uses the unexecuted 16-action tail of its
previous full prediction as the RTC prefix, then generates a new 32-action
prediction. Its committed first `d` actions are restored exactly. The returned
batch therefore contains:

```text
old prediction's remaining actions [0:d]
new prediction aligned to tick t    [d:16]
```

The server keeps the full new prediction, including its unreturned `[16:32]`
tail for the next round. `simulation_info` reports the old/new counts and the
request tick. `d` can vary; `d=16` is a legal boundary where a batch is entirely
old actions. A batch is not guaranteed to contain a fixed mix of old and new.

The client executes the returned sequence in the simulator. Before consuming
the next waypoint, it reads the actual observation produced by the preceding
completed waypoint. Each uploaded image is one preprocessed multi-camera
composition in uint8 CHW format, rather than three separate raw camera arrays.
It submits these exact 16 post-action observations and the final measured state
in the next request.
The first batch contains ticks 1 through 16, then 17 through 32. Observations
belong to the actions actually executed, including the old-action prefix;
they never come from generated video or an unexecuted new trajectory. Reset
supplies the real tick-zero image. The final batch frame also supplies the
request's current image.

The server validates every uploaded frame and retains the stride-four samples
needed by the existing history-window rule. Sixteen uploaded images contribute
four new sampled observations; they are not sixteen consecutive VAE input
frames. The original complete-32-step history-block schedule is preserved:
the tick-16 request uses only its current image, and tick 32 can use images
0, 4, ..., 32. The server retains earlier samples until their history horizon
expires, including samples not needed by the immediately preceding request.

While the server encodes and predicts, the client makes no simulator step.
The process still waits for the real RPC and spends its actual wall time.
That wait does not advance simulated physics. The modelled delay is represented
by the committed old-action prefix, which is executed after the RPC returns.
Servo interpolation advances by logical waypoint time and does not sleep or
skip steps because of
model or rendering wall time. Model history still uses the checkpoint's 30 Hz
grid; actual simulated motion uses the selected execution rate.

After each request, the client checks the measured full RPC duration against
the reservation chosen before inference. A request exceeding its reservation
stops with a deadline error; its new actions are not executed. Diagnostics
separate reserved `d` from the measured delay expressed in waypoint steps.
This is a simulation of the RTC execution sequence, not an actual asynchronous
hardware or network deadline test. `virtual_rtc_feasible` reports the emulated
condition; `continuous_rtc` is not asserted for this mode.

The client writes `metrics.json` and `waypoints_unified.npy` as usual.
`action_trace[].action_provenance` identifies each executed waypoint as
`bootstrap`, `old_prefix` or `new_prediction`, with its `prediction_tick` and
`prediction_row`. `final_status.last_request` records the old/new counts,
reserved and measured delays, `accepted` and `deadline_miss`.
`requested_execution_hz` is the requested starting rate; `initial_execution_hz`
is the calibrated rate when motion begins; `final_execution_hz` is the final
applied rate. Trace schedules are logical times in this mode, while RPC/call
durations and wall timestamps are still measured. Zero reported logical
lateness does not establish a real-time deadline bound.

The server launcher accepts `SIMULATION_SYNC=1`. Without the flag, the existing
ordinary and hardware RTC paths retain their behavior.

## Start a RoboTwin client

Use a Python environment that can already run the RoboTwin simulator. The client
needs NumPy, msgpack, websockets with the synchronous direct-connection client
API (validated with version 17), CPU-capable
PyTorch and Torchvision. It needs the repository's Python code, but no checkpoint,
processor weights, VAE weights or NPU. The control and image helpers import Torch
on the client; the wire codec and basic connection do not load model code.

The adapter expects the supplied RoboTwin checkout to provide
`scripts/eval_policy_xpolicylab.py` and its task/configuration helpers, as did the
source client/server branch. Point `--robotwin-root` at that checkout.
The required helpers include `load_task_args`, `resolve_eval_instruction`,
`class_decorator`, `build_instruction`, `safe_close_env` and `UnStableError`.
A checkout with only `script/eval_policy.py` is not directly compatible. The
local RoboTwin checkout used during development lacks the expected helper and
object assets; the flag does not install those dependencies or adapt that
checkout. Use a compatible RoboTwin installation before running task evaluation.

```bash
python examples/action_pretrain/robotwin_client.py \
  --robotwin-root /path/to/RoboTwin \
  --server-uri ws://SERVER_IP:8000 \
  --task-name adjust_bottle --task-config demo_clean \
  --control-mode joint --state-source measured \
  --enable_rtc --initial-execution-hz 30 --servo-hz 30 \
  --max-waypoints 48 \
  --output outputs/causal_action/robotwin_sim
```

`--initial-execution-hz` sets the requested starting rate of consuming model
waypoints; it defaults to the server's model rate when omitted. With adaptive RTC,
startup calibration can reduce it before motion starts, and the online fit can
reduce it further as history grows. The runtime `execution_hz` status field
reports the current execution rate. Without adaptive RTC, the requested rate
stays fixed. `--execution-hz` remains a compatibility alias.

In `metrics.json`, `requested_execution_hz` records this pre-calibration target;
`initial_execution_hz` records the actual rate when movement starts after
calibration, which can be lower.

`--servo-hz` is separate: the adapter interpolates targets and
issues native joint/EEF commands at that higher rate. The local rate callback
acknowledges a slowdown by changing the adapter's actual future schedule.
Moving to a target immediately and then sleeping would not implement this
interpolation contract.

`--state-source measured` uses the simulator's measured joint getters; EEF
observations come from `get_obs()`. `command` retains the older command-state
option. Deployment must preserve the training state and gripper units.
`--control-mode eef` requires the server statistics and robot configuration to
support both EEFs and grippers. Camera layout and padded image geometry must stay
consistent during an episode.

`--max-waypoints` limits model actions. `--max-steps` instead limits native
`env.take_action` calls; servo interpolation can use several such calls for a
single model waypoint. Episode metrics distinguish issued waypoints, fully
completed waypoints and native servo calls. In wall-clock execution, slow
simulator calls produce reported
lateness and skipped servo deadlines rather than a burst of old commands.

The client saves `waypoints_unified.npy` and `metrics.json`, including action and
servo timing, rate changes, session state and RPC timing. This entry point does
not import the source branch's prediction-video decoding engine.

`examples/action_pretrain/robotwin_client_template.sh` forwards these settings
through environment variables. Use `INITIAL_EXECUTION_HZ` for the requested
starting rate; the older `EXECUTION_HZ` variable is a fallback when the new one is
unset or empty. Activate the existing RoboTwin environment first. The script
also supports `PRINT_ONLY=1`.

## Transport, latency and resets

Each request carries a connection-local increasing request ID, an episode ID,
observation tick and revision. Responses must match all four before the client
accepts their actions. Reset drains the previous local session and creates a new
episode. A timeout closes the connection so an unread late result cannot become
the next request's response. Remote errors, mismatched IDs and disconnects surface
as errors rather than silently replaying old commands.

In ordinary and wall-clock RTC modes, the client uploads the genuine historical
frame indexes required by the current window. With `raw-history-refs-v1`, the
server acknowledges which historical images
it has received; later requests refer to those ticks and upload only new history
images and the fresh current image. The server expands references before calling
the same model planner. This retains raw observations only: VAE encoding and model
prefix computation follow the existing inference path on each request.

In those modes, only historical images enter this transport cache. A bootstrap or calibration
current image must not substitute for a later historical observation with the
same tick. Historical images are immutable within an episode; conflicting bytes
for a cached tick or a reference to a missing image require recovery. The cache is
bounded by the configured history horizon and cleared at episode boundaries.
Clients without this capability use complete history payloads.

At each continuing planning boundary, simulated RTC uploads its complete
16-step post-action batch, without history references. Reset retains the real
tick-zero image; later batches retain
the stride-four samples, including each batch's final current frame. Calibration
does not replace those retained images. The current frame is also serialized in
the top-level `image` field, so there are 16 distinct post-action images and a
duplicate serialization of the final one. Feedback is sent when another
prediction is needed. Episode termination does not trigger an extra request;
an incomplete final batch is not padded or uploaded. The generic
`history_frames_sent`/`history_frames_reused` counters describe history-reference
payloads, not `executed_observations`, and can both be zero in simulation mode.

Both endpoints use a bounded 512 MiB message limit. A cold full window of
257 prepared 640×640 RGB images is about 316 MB before small metadata overhead;
this fits the limit. After streaming has populated the history cache, requests
normally upload only newly required images. Larger resolutions/windows still
need their message sizes checked against this bound.

The session's runtime timer covers local history packing, RPC encoding, upload,
server queueing/inference, download, decoding and conversion to CPU actions.
Startup calibration times the RPC planner as well. `request_timings` additionally
records the wire round-trip and server inference portions for diagnosis. The
online latency buffer uses completed request durations on the client, so a
slow server or network increases the predicted delay and the required slowdown.
Image acquisition, client camera preparation and actuator delivery lie outside
the planner timer; use `--extra-latency-seconds` for deployment headroom and inspect
actual command/servo lateness.

`--latency-factor`, `--margin-steps` and `--minimum-execution-hz` configure the
adaptive budget. In wall-clock RTC, an unattainable minimum rate, missing
acknowledgment or missed
deadline preserves the current warning/fallback behavior. Ordinary fallback can
pause command delivery. The client records whether RTC remained active throughout
the episode; successful inference does not alone imply continuous RTC.
Simulated RTC logs a warning and raises on an infeasible budget, failed rate
acknowledgement or exceeded reservation. It stops before returning a new batch's
action and does not automatically enter the ordinary fallback path. Reset the
episode before attempting another simulated run after such an error.

## Validation scope

CPU tests exercise the real localhost WebSocket protocol, ordinary execution,
background RPC progress, round-trip timing, prefix preservation, adaptive pacing,
request/reset isolation, timeout closure and historical-image references. Model
adapter tests check that bootstrap uses ordinary inference and later requests use
the shared RTC planner with genuine observations and zero future placeholders.

The simulated-RTC revision passed 74 focused CPU tests: 40 existing serving,
network and RoboTwin-adapter cases, 27 simulated wire/model-adapter cases and
seven delay/session/servo cases. The localhost tests use the actual WebSocket
codec and `ServingModelPolicy` with a fake model backend. They verify 32 predicted
rows and 16 returned rows, exact `d=4` and `d=16` old-prefix restoration,
retention of the unreturned prediction tail, stationary calibration, history
capacity/reset and rejection of malformed batches. Action-response environments
produce state and pixels from the commands actually executed; the returned
observations are checked against those results. Additional cases exercise
online fitting from completed samples, the eight-sample buffer, latency spikes,
minimum rates and exact 11/13-servo-step completion without floating-point
endpoint overruns.

These are protocol, causality and execution-logic checks. The new
`--simulation-sync` path has not yet been run with the real checkpoint in a full
RoboTwin scene or on a physical robot, and no task-success rate is established.
The earlier NPU replay below used the wall-clock RTC path. The 74 focused cases
are separate from, and must not be added to, the historical 435-test count.

The real 0930 NPU replay exercises ordinary bootstrap, stationary calibration,
112 client waypoints with background RPCs and a separate full-history request.
That replay explicitly used a 30 Hz starting target, latency factor 1.2 and a
three-step margin; it is not a validation of every default-rate timing condition.
Same-input comparisons check the payload, constructed batch, initial noise and
first forward result. On the tested Ascend backend, ordinary predictions repeat
bitwise, while RTC input VJPs and uncommitted predictions can vary even across
consecutive calls on the same model thread. Full RTC output is therefore not
claimed to be bitwise reproducible. The committed action prefix remains exact;
wire/history-reference input equivalence is checked independently of this NPU
backward variability.

The RoboTwin control loop is tested with simulated environment interfaces. Actual
RoboTwin execution needs its separate simulator dependencies and checkout. Local
network/model replay verifies the serving chain; it does not establish physical
robot dynamics, task success or latency bounds for a remote deployment.

## Recorded issues and remaining risks

The integration's CPU regression run passed 435 tests. The real NPU replay used
the 0930 iteration-20000 checkpoint, CFG=1, ten denoise steps, full compatible
prefix caching and RTC beta=5. It completed 112 waypoints and six RTC publications
without fallback. The requested 30 Hz execution rate was calibrated to about
4.60 Hz and later reduced to about 3.05 Hz. The replay used a three-step margin,
not the client's default one-step margin. A separate 32-history-block request
covered 34.13 model seconds; continuous execution all the way to that history
capacity has not been validated by this network replay. Diagnostic hashing and
first-step capture add overhead, so these runs are not pure latency benchmarks.

### Open: NPU input-gradient repeatability

In short- and full-history cases, three repetitions in each of the model-worker
and main-thread contexts used identical payload bytes, constructed inputs,
initial noise, RTC targets and weights. The first forward velocity was identical;
the first difference appeared in the input VJP. Ordinary inference repeated
bitwise. RTC differences also occurred without changing threads or crossing the
network. This localizes the first observed discrepancy to the backward stage,
but does not identify a particular operator or prove its underlying cause.

For the uncommitted final 16 actions, same-model-worker repetitions showed joint
mean absolute differences of about 0.00042-0.00055 rad (0.024-0.032 degrees) and
mean EEF orientation differences of about 0.32-0.39 degrees. Across the sampled
within-thread comparisons, maxima were about 0.22 degrees for joint components,
0.85 mm for EEF position distance, 1.27 degrees for EEF orientation and 0.46 in
native gripper units. Gripper units and operating range must be checked against
the deployment's statistics; this number is neither a joint angle nor a known
percentage of gripper travel. These are repeat-to-repeat output differences,
not errors against ground truth or measurements of RTC task success.

The committed first 16 actions remained bitwise equal to the supplied prefix.
Model parameter versions were unchanged and parameter gradients remained unset.
The observed variability must not be treated as proven harmless numerical noise
or an established unavoidable loss from RTC. Operator-level isolation and a
deterministic-computation comparison, including its latency cost, remain open.
Changing deterministic settings alone has not yet been verified to resolve it.

### Deployment and resource limits

- Real RoboTwin task execution, physical-robot closed-loop success, WAN transfer
  latency and continuous full-window operation remain untested. The paced replay
  uses recorded observations; it does not simulate changed robot dynamics under
  slower execution. Servo scheduling on the client is not a hard real-time bound.
- The online latency fit and slowdown are estimates. Sudden delay increases can
  still miss a deadline, and fallback can pause action delivery. Camera capture,
  preparation and actuator delivery require their own measurements and budget.
- Historical-image references reduce wire traffic, but `RTCSession` still clones
  retained raw observations before the RPC client selects video-stride frames.
  CPU memory and copy cost therefore grow with history; the transport cache does
  not remove these local costs. Target-controller resource use remains to be
  measured.
- Checkpoint-specific normalization, joint masks, coordinate frames, measured
  state conventions and gripper units must match the deployed robot. The Agibot
  replay validates the inference path, not compatibility with a RoboTwin task.
- The service currently accepts one client and one model rank. A timeout closes
  the connection; an outstanding model call is drained before cleanup. There is
  no transparent reconnection or promise of uninterrupted execution after errors.
- The supplied server uses plain WebSocket without built-in authentication or
  TLS and binds to `0.0.0.0` by default. Deployment assumes a restricted trusted
  network or externally secured transport; do not expose this endpoint publicly.

### Integration issues corrected before this revision

- The model seed is supplied as a per-sample list, as required by the actual
  generation path, and the structured action prompt is serialized as JSON before
  tokenization to match training.
- A cold full-history payload exceeded the initial 256 MiB transport limit. Both
  endpoints now use 512 MiB, and an actual approximately 316 MB request completed.
  Larger image geometries can still exceed that finite limit.
- Localhost connections explicitly bypass proxy auto-detection. Episode/request
  identity checks and acknowledged immutable history references prevent stale
  replies or calibration images from being substituted for actual history.
