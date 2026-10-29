# Causal action client/server with RTC

The `rtc` branch provides a WebSocket model server and a RoboTwin client adapter.
The transport and native robot conversion are adapted from
`feat/robotwin-client-server`; inference uses the current RTC branch's shared
model planner, caches ①＋② and optional RTC guidance.

The protocol is `cosmos-causal-action-rtc-v1`. Run matching client/server code from
this branch. The older synchronous `cosmos-causal-action-v2` client is a different
protocol and fails the handshake rather than silently changing execution.

## Ownership and execution

The server loads the checkpoint, VAE, processor and training statistics. The
client owns the action buffer, observation ticks, servo schedule and online
latency fit. With RTC enabled, the existing `RTCSession` runs on the client CPU;
its background planner performs a blocking RPC while the foreground continues
executing the previous chunk.

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

`--enable_rtc` (alias `--enable-rtc`) is opt-in on both ends. Without it, the client
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
completed waypoints and native servo calls. Slow simulator calls produce reported
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

The client uploads the genuine historical frame indexes required by the current
window. With `raw-history-refs-v1`, the server acknowledges which historical images
it has received; later requests refer to those ticks and upload only new history
images and the fresh current image. The server expands references before calling
the same model planner. This retains raw observations only: VAE encoding and model
prefix computation follow the existing inference path on each request.

Only historical images enter this transport cache. A bootstrap or calibration
current image must not substitute for a later historical observation with the
same tick. Historical images are immutable within an episode; conflicting bytes
for a cached tick or a reference to a missing image require recovery. The cache is
bounded by the configured history horizon and cleared at episode boundaries.
Clients without this capability use complete history payloads.

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
adaptive budget. An unattainable minimum rate, missing acknowledgment or missed
deadline preserves the current warning/fallback behavior. Ordinary fallback can
pause command delivery. The client records whether RTC remained active throughout
the episode; successful inference does not alone imply continuous RTC.

## Validation scope

CPU tests exercise the real localhost WebSocket protocol, ordinary execution,
background RPC progress, round-trip timing, prefix preservation, adaptive pacing,
request/reset isolation, timeout closure and historical-image references. Model
adapter tests check that bootstrap uses ordinary inference and later requests use
the shared RTC planner with genuine observations and zero future placeholders.

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
