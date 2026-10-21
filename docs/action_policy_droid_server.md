# Cosmos3-Policy-DROID Server

Cosmos 3 offers two post-trained policy models for DROID:

1. [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
2. [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)

Each of these policy models is served by a policy **Server** that streams actions to a **Client** driving a simulated or real robot. This example uses [`RoboLab`](https://github.com/NVlabs/RoboLab), a simulation benchmark for task-generalist policies, as the client. Start the server first, then connect the client.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Policy Server](#policy-server)
- [Simulation Client](#simulation-client)
- [RoboTwin EEF Dataset Replay](#robotwin-eef-dataset-replay)
  - [Data and environment requirements](#data-and-environment-requirements)
  - [Export actions without starting the simulator](#export-actions-without-starting-the-simulator)
  - [Run the replay in a RoboTwin environment](#run-the-replay-in-a-robotwin-environment)
  - [Output files](#output-files)

______________________________________________________________________

<!--TOC-->

## Policy Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```bash
docker build \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```bash
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

The `--group=cu130-train` line targets CUDA 13.x drivers. On CUDA 12.x systems, replace it with `--group=cu128-train` (see the [Cosmos3 Cookbooks: Environment Setup](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md) for details).

Inside the container, start the policy server:

1. For [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --port 8000
   ```

2. For [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
     --port 8000 \
     --format-prompt-as-json True
   ```

## Simulation Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```bash
./docker/build_docker.sh latest
```

Launch the container:

```bash
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

To evaluate across multiple sub-environments in parallel in headless mode:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```

Example output:

<video controls width="864" height="480" src="https://github.com/user-attachments/assets/95a16737-5eb9-4b3f-a0ad-3a6b929b423f"></video>

## RoboTwin EEF Dataset Replay

[`robotwin_eef_replay.py`](../examples/action_pretrain/robotwin_eef_replay.py) drives RoboTwin with the EEF targets from one training-data episode and records simulation observations. This is a standalone entry point: it reads the dataset directly and requires no policy server or model checkpoint. It reuses the existing RoboTwin client's environment-loading, camera and video helpers.

### Data and environment requirements

Point `--dataset-root` to one LeRobot dataset containing `meta/info.json` and `data/`, for example `RoboTwin__adjust_bottle/clean`. Select an episode with `--episode-index`; its `frame_index` values must be contiguous and start at zero. The script reads physical, absolute `action_unified` values directly, without normalization, extra temporal shifting or video-stride sampling.

The dataset must contain `episode_index`, `frame_index`, `action_unified` and `mask_action`. The action and mask arrays must have width 55, with both EEFs and grippers enabled in every row:

| Unified slots | Meaning                   | Required representation   |
| ------------- | ------------------------- | ------------------------- |
| `14:21`       | Left EEF pose             | `x,y,z,qx,qy,qz,qw`       |
| `21:28`       | Right EEF pose            | `x,y,z,qx,qy,qz,qw`       |
| `28:30`       | Left/right gripper target | Opening values in `[0,1]` |

By default, `--pose-frame base` interprets EEF poses in the shared robot base frame. The script reads `custom.eef_enrichment.T_world_base` from `meta/info.json`, using its `position` and `quaternion_wxyz` fields, and transforms both position and orientation into the world frame. To override that transform, pass `--world-from-base x y z qw qx qy qz`. If the unified EEF slots already contain world-frame poses, use `--pose-frame world` and omit `--world-from-base`. Both modes interpret the input quaternion as `xyzw` and export `wxyz` for RoboTwin.

For simulation, use a working RoboTwin Python environment with NumPy, PyArrow, OpenCV and msgpack, plus FFmpeg with the `libx264` encoder on `PATH`. The RoboTwin checkout must include `scripts/eval_policy_xpolicylab.py`, which supplies the task/config helpers. The scene seed, task configuration, robot and assets must match the data you intend to replay. The episode index does not identify the simulator seed.

### Export actions without starting the simulator

Run from the Cosmos repository root. Replace the paths and set `SIM_SEED` to the intended scene seed; `0` below is only an example.

```bash
cd /path/to/cosmos-framework
export DATASET_ROOT=/path/to/robotwin_processed/RoboTwin__adjust_bottle/clean
export SIM_SEED=0

python examples/action_pretrain/robotwin_eef_replay.py \
  --dataset-root "$DATASET_ROOT" \
  --episode-index 0 \
  --seed "$SIM_SEED" \
  --prepare-only \
  --output outputs/robotwin_eef_replay
```

`--prepare-only` requires NumPy, PyArrow and msgpack but does not import RoboTwin/SAPIEN. It validates and exports the actions, writing only `action.npy` and `metrics.json` under `outputs/robotwin_eef_replay/seed${SIM_SEED}/`. This checks data preparation, not simulator execution or task success.

### Run the replay in a RoboTwin environment

Using the same dataset and seed variables, run the following on a machine with a working RoboTwin installation:

```bash
python examples/action_pretrain/robotwin_eef_replay.py \
  --robotwin-root /path/to/RoboTwin \
  --dataset-root "$DATASET_ROOT" \
  --episode-index 0 \
  --task-name adjust_bottle \
  --task-config demo_clean \
  --seed "$SIM_SEED" \
  --output outputs/robotwin_eef_replay
```

Use `--task-config demo_randomized` for the corresponding randomized data. `--task-name` can be omitted when `meta/info.json` supplies `custom.source.task`. Add `--max-steps 32` to limit replay to the first 32 action rows; RoboTwin may stop earlier on task success or its task step limit. `--video-fps` sets video playback FPS and defaults to 30.

Each exported action has 16 values: `[left xyz, left qw,qx,qy,qz, left gripper, right xyz, right qw,qx,qy,qz, right gripper]`. The script calls `env.take_action(action, action_type="ee")` and then `env.get_obs()` after each completed action. It saves an initial observation before the first action, so a successful recording of N executed actions contains N+1 observation frames.

### Output files

All files are written directly under `--output/seed{seed}/`:

| File               | Contents                                                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `action.npy`       | World-frame EEF/gripper control targets, shape `[T,16]`; T is the selected action count after applying `--max-steps`.                  |
| `observations.npz` | Initial and subsequent EEF poses, measured arm qpos, joint/gripper commands, success flags, executed actions and source frame indices. |
| `video.mp4`        | H.264 video of simulation RGB observations: head camera on top, resized left/right wrist cameras side by side below.                   |
| `metrics.json`     | Dataset/scene settings, run status, execution count, stop reason, task success and mean/max position and rotation tracking errors.     |

In `observations.npz`, `eef_world_wxyz` has shape `[N+1,14]` (left pose followed by right pose), `measured_arm_qpos` contains actual arm joint positions, and `joint_command` contains drive targets including commanded gripper values. `eval_success` aligns with the N+1 observations. `executed_action` and `source_frame_index` have N rows; their row i corresponds to observation row i+1. Separate RGB image files are not written.

The EEF targets in the enriched RoboTwin data are derived from the next recorded pose. RoboTwin replans a trajectory for each target, so this replay does not guarantee the original physical timing or trajectory. The video FPS controls playback only; it does not set the simulation/control frequency. A finished replay is not necessarily a successful task: check `eval_success`, `stop_reason` and the measured tracking errors in `metrics.json`.
