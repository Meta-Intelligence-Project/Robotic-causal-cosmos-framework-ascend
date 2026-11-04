#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Liujin local causal-action mid-training launcher.
#
# Wraps launch_midtrain_action_causal.sh with local paths, preflight checks,
# a compact launch summary, and optional smoke-test overrides.
#
# Current default behavior:
#   SOURCE_SET=mixed
#   RUN_ID=TEST                     (exported for compatibility/logging)
#   LOG_ROOT=/mnt/sfs_turbo/liujin/training_logs/action_midtrain
#   RUN_DIR=/mnt/sfs_turbo/liujin/training_logs/action_midtrain/TEST_20261103_125549
#   STATS_RUN_DIR=/mnt/sfs_turbo/liujin/training_logs/action_midtrain/TEST_20261103_125549
#   OUTPUT_ROOT=${RUN_DIR}/train_${SOURCE_SET}
#
# RUN_DIR is currently assigned unconditionally below, so
# `RUN_DIR=... bash ...` does NOT override it. To write to a new directory,
# edit the RUN_DIR assignment (or restore the commented timestamp-based form).
# RUN_ID and LOG_ROOT are exported but do not derive RUN_DIR in the current
# version; they are kept for parity with liujin_compute_dataset_stats.sh.
#
# Statistics are NOT generated here. They must already exist under
# ${STATS_RUN_DIR}/stats, or be supplied explicitly through
# AGIBOT_GROUP_STATS_PATH / EGOSUITE_GROUP_STATS_PATH. By default this script
# reuses the TEST_20261103_125549 stats produced by
# liujin_compute_dataset_stats.sh.
#
# Usage:
#   # dry-run: run preflight, print the resolved config and final torchrun
#   # command, but do not start training
#   PRINT_ONLY=1 bash examples/action_pretrain/liujin_launch_midtrain_local.sh
#
#   # smoke test using the current smoke defaults:
#   #   trainer.max_iter=10, checkpoint.save_iter=6
#   SMOKE=1 bash examples/action_pretrain/liujin_launch_midtrain_local.sh
#
#   # one-step smoke: override both smoke defaults explicitly
#   SMOKE=1 SMOKE_MAX_ITER=1 SMOKE_SAVE_ITER=1 \
#     bash examples/action_pretrain/liujin_launch_midtrain_local.sh
#
#   # AgiBot-only one-step smoke (avoids mixed-source shard constraints)
#   SOURCE_SET=agibot SMOKE=1 SMOKE_MAX_ITER=1 SMOKE_SAVE_ITER=1 \
#     bash examples/action_pretrain/liujin_launch_midtrain_local.sh
#
#   # normal run (no smoke overrides)
#   bash examples/action_pretrain/liujin_launch_midtrain_local.sh
#
#   # resume: keep the same RUN_DIR/OUTPUT_ROOT and rerun; the trainer
#   # auto-resumes from the latest DCP in the output directory. Since RUN_DIR
#   # is hard-coded, reusing the default directory is the resume path.
#
# Smoke logic:
#   SMOKE=1 only appends these two Hydra overrides:
#     trainer.max_iter=${SMOKE_MAX_ITER:-10}
#     checkpoint.save_iter=${SMOKE_SAVE_ITER:-6}
#   It does NOT reduce ranks/workers, sequence length, or data sources.
#   The overrides are inserted before forwarded CLI args, so an explicit
#   `trainer.max_iter=...` or `checkpoint.save_iter=...` argument still wins.
#
# Other useful overrides:
#   SOURCE_SET=agibot|egosuite|mixed
#   MODE=joint|forward_dynamics|inverse_dynamics|policy
#   STATS_RUN_DIR=... (directory containing stats/*.json)
#   AGIBOT_GROUP_STATS_PATH=..., EGOSUITE_GROUP_STATS_PATH=...
#   LOG_ROOT=..., OUTPUT_ROOT=..., RUN_ID=...
#   ASCEND_RT_VISIBLE_DEVICES=2,3,4,6
#   NPROC_PER_NODE=4
#   SMOKE_MAX_ITER=..., SMOKE_SAVE_ITER=...
#
# All remaining CLI arguments are forwarded to launch_midtrain_action_causal.sh,
# so Hydra overrides such as trainer.max_iter=10 still work.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Run directory convention (same LOG_ROOT/RUN_ID style as the stats script)
# ---------------------------------------------------------------------------
DEFAULT_RUN_ID="TEST"
LOG_ROOT="${LOG_ROOT:-/mnt/sfs_turbo/liujin/training_logs/action_midtrain}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
export LOG_ROOT
export RUN_ID
# export RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}_$(date +%Y%m%d_%H%M%S)}"


export RUN_DIR="/mnt/sfs_turbo/liujin/training_logs/action_midtrain/TEST_20261103_125549"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_DIR/train_${SOURCE_SET:-mixed}}"

# ---------------------------------------------------------------------------
# Dataset / checkpoint paths on this machine
# ---------------------------------------------------------------------------
export AGIBOT_ROOT="${AGIBOT_ROOT:-/mnt/sfs_turbo/public/datasets/agibot_processed/v0}"
export EGOSUITE_ROOT="${EGOSUITE_ROOT:-/mnt/sfs_turbo/public/datasets/egosuite_processed}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

# ---------------------------------------------------------------------------
# Statistics produced by liujin_compute_dataset_stats.sh
# ---------------------------------------------------------------------------
STATS_RUN_DIR="${STATS_RUN_DIR:-/mnt/sfs_turbo/liujin/training_logs/action_midtrain/TEST_20261103_125549}"
export STATS_RUN_DIR
export AGIBOT_GROUP_STATS_PATH="${AGIBOT_GROUP_STATS_PATH:-$STATS_RUN_DIR/stats/agibot_group_stats.json}"
export EGOSUITE_GROUP_STATS_PATH="${EGOSUITE_GROUP_STATS_PATH:-$STATS_RUN_DIR/stats/egosuite_group_stats.json}"

# ---------------------------------------------------------------------------
# Training recipe / source manifest
# ---------------------------------------------------------------------------
SOURCE_SET="${SOURCE_SET:-mixed}"
export ACTION_SOURCES_FILE="${ACTION_SOURCES_FILE:-$SCRIPT_DIR/sources/$SOURCE_SET.json}"
export TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/action_midtrain_edge_causal_tnd.toml}"
export MODE="${MODE:-joint}"
export ALLOW_MOCK_STATISTICS="${ALLOW_MOCK_STATISTICS:-false}"
export LOOKAHEAD_LIMIT="${LOOKAHEAD_LIMIT:-1}"
export MROPE_BASE_FPS="${MROPE_BASE_FPS:-24}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_ROOT="${CONDA_ROOT:-/mnt/sfs_turbo/public/apps/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework-py312}"

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

unset TORCH_DEVICE_BACKEND_AUTOLOAD

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# NPU 0/1/5 had less free HBM; 2/3/6/7 had more free HBM when checked.
# Override with ASCEND_RT_VISIBLE_DEVICES if your allocation changes.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,6}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-50142}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
missing=0
check_path() {
    if [[ ! -e "$1" ]]; then
        printf 'MISSING: %s\n' "$1" >&2
        missing=1
    fi
}

check_path "$ACTION_SOURCES_FILE"
check_path "$TOML_PATH"
check_path "$BASE_CHECKPOINT_PATH"
check_path "$COSMOS3_EDGE_PROCESSOR_PATH"
check_path "$WAN_VAE_PATH"

case "$SOURCE_SET" in
    agibot)
        check_path "$AGIBOT_ROOT"
        check_path "$AGIBOT_GROUP_STATS_PATH"
        ;;
    egosuite)
        check_path "$EGOSUITE_ROOT"
        check_path "$EGOSUITE_GROUP_STATS_PATH"
        ;;
    mixed)
        check_path "$AGIBOT_ROOT"
        check_path "$EGOSUITE_ROOT"
        check_path "$AGIBOT_GROUP_STATS_PATH"
        check_path "$EGOSUITE_GROUP_STATS_PATH"
        ;;
    *)
        printf 'Unknown SOURCE_SET=%s (use agibot, egosuite, or mixed)\n' "$SOURCE_SET" >&2
        exit 2
        ;;
esac

if [[ "$missing" -ne 0 ]]; then
    printf '\n' >&2
    printf 'Some required paths are missing. For statistics:\n' >&2
    printf '  - run examples/action_pretrain/liujin_compute_dataset_stats.sh, or\n' >&2
    printf '  - set STATS_RUN_DIR to the stats output directory, or\n' >&2
    printf '  - set AGIBOT_GROUP_STATS_PATH / EGOSUITE_GROUP_STATS_PATH explicitly.\n' >&2
    if [[ "${PRINT_ONLY:-0}" != "1" ]]; then
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Smoke-test overrides
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(
        "trainer.max_iter=${SMOKE_MAX_ITER:-10}"
        "checkpoint.save_iter=${SMOKE_SAVE_ITER:-6}"
    )
fi

printf 'Repository:        %s\n' "$REPO_ROOT"
printf 'Source set:        %s\n' "$SOURCE_SET"
printf 'Sources file:      %s\n' "$ACTION_SOURCES_FILE"
printf 'TOML:              %s\n' "$TOML_PATH"
printf 'Mode:              %s\n' "$MODE"
printf 'RUN_ID:            %s\n' "$RUN_ID"
printf 'RUN_DIR:           %s\n' "$RUN_DIR"
printf 'OUTPUT_ROOT:       %s\n' "$OUTPUT_ROOT"
printf 'Stats run dir:     %s\n' "$STATS_RUN_DIR"
printf 'AgiBot stats:      %s\n' "$AGIBOT_GROUP_STATS_PATH"
printf 'EgoSuite stats:    %s\n' "$EGOSUITE_GROUP_STATS_PATH"
printf 'NPU cards:         %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
printf 'NPROC_PER_NODE:    %s\n' "$NPROC_PER_NODE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    printf 'Smoke overrides:   %s\n' "${EXTRA_ARGS[*]}"
fi

if [[ "${PRINT_ONLY:-0}" != "1" ]]; then
    mkdir -p "$RUN_DIR"
fi

exec bash "$SCRIPT_DIR/launch_midtrain_action_causal.sh" "${EXTRA_ARGS[@]}" "$@"
