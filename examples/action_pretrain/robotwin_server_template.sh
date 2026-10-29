#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Replace /path/to values, or override them through environment variables.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/trained/checkpoint}"
COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
ROBOTWIN_GROUP_STATS_PATH="${ROBOTWIN_GROUP_STATS_PATH:-/path/to/robotwin_group_stats.json}"

CMD=(torchrun --standalone --nproc-per-node=1
    -m cosmos_framework.inference.causal_action.server
    --checkpoint "$CHECKPOINT_ROOT" --processor "$COSMOS3_EDGE_PROCESSOR_PATH"
    --vae "$WAN_VAE_PATH" --statistics "$ROBOTWIN_GROUP_STATS_PATH"
    --execute-steps "${EXECUTE_STEPS:-16}" --history "${HISTORY_BLOCKS:-32}"
    --steps "${NUM_STEPS:-10}" --guidance "${GUIDANCE:-1}"
    --fps "${FPS:-30}" --resolution "${RESOLUTION:-480}" --cache-mode "${CACHE_MODE:-full}"
    --seed "${SEED:-42}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}")
[[ -n "${ROBOT_CONFIG:-}" ]] && CMD+=(--robot-config "$ROBOT_CONFIG")
case "${ENABLE_RTC:-0}" in
    1) CMD+=(--enable_rtc) ;;
    0) ;;
    *) printf 'ENABLE_RTC must be 0 or 1\n' >&2; exit 2 ;;
esac
CMD+=("$@")
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

ASCEND_ENV="${ASCEND_ENV:-/path/to/ascend-toolkit/set_env.sh}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos}"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "${CMD[@]}"
