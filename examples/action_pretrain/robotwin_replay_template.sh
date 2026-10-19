#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。
# 完整回放所选片段一次；结果保存到 OUTPUT_ROOT/seed{server 采样 seed}/，默认 outputs/seed42/。
# 结束后由 server 将整段预测 latent 一次解码，沿用 EveryNDrawSample 的 MP4 质量和 16 FPS。
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
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec python -m cosmos_framework.inference.causal_action.replay \
    --dataset-root "${ROBOTWIN_ROOT:-/path/to/robotwin_processed}" \
    --uri "${SERVER_URI:-ws://localhost:8000}" \
    --dataset-index "${DATASET_INDEX:-0}" --index "${SAMPLE_INDEX:-0}" \
    --split-seed "${DATA_SPLIT_SEED:-42}" \
    --max-action-steps "${MAX_ACTION_STEPS:-896}" --overlap-action-steps "${OVERLAP_ACTION_STEPS:-161}" \
    --output "${OUTPUT_ROOT:-outputs}" "$@"
