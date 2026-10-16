#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。

# RoboTwin policy 全段视频预测模板。
# 按训练片段长度滚完每一个 block，
# 解码 video.mp4，并把反归一化后的预测动作写成 action.npy。
# 这是样本真值 state 条件下的整段预览，不是把预测画面喂回去的自由闭环。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

ASCEND_ENV="${ASCEND_ENV:-/path/to/ascend-toolkit/set_env.sh}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework}"

# shellcheck disable=SC1090
source "$ASCEND_ENV"
set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/trained/checkpoint}"
export ACTION_SOURCES_FILE="${ACTION_SOURCES_FILE:-$SCRIPT_DIR/sources/robotwin.json}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/robotwin_processed}"
export ROBOTWIN_GROUP_STATS_PATH="${ROBOTWIN_GROUP_STATS_PATH:-/path/to/robotwin_group_stats.json}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/causal_action/inference_robotwin}"

# 与训练配方对齐：896 action、重叠 161、历史 32 个 block。一次只取一个片段。
export MODE=policy
export SOURCE_INDEX="${SOURCE_INDEX:-0}"
export SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
export MAX_ACTION_STEPS="${MAX_ACTION_STEPS:-896}"
export OVERLAP_ACTION_STEPS="${OVERLAP_ACTION_STEPS:-161}"
export HISTORY_BLOCKS="${HISTORY_BLOCKS:-32}"
export ACTIONS_PER_BLOCK="${ACTIONS_PER_BLOCK:-32}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-4}"
export NUM_STEPS="${NUM_STEPS:-20}"
export GUIDANCE="${GUIDANCE:-3}"

mkdir -p "$OUTPUT_ROOT"

bash "$SCRIPT_DIR/inference_causal_action.sh" \
    --preview-ground-truth-states \
    --decode-video \
    "$@"

[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

python - <<'PY'
import os
from pathlib import Path

import numpy as np
import torch

output = Path(os.environ["OUTPUT_ROOT"])
payload = torch.load(output / "sample.pt", map_location="cpu", weights_only=True)
action = payload["action"][0].detach().float().cpu().numpy()
path = output / "action.npy"
np.save(path, action)
print(f"Saved {path} shape={tuple(action.shape)} dtype={action.dtype}")
PY
