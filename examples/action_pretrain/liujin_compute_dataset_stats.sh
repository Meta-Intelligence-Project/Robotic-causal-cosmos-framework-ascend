#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generic causal-action dataset statistics for Liujin's machine.
#
# Two execution modes:
#
#   MODE=single   (default)
#     python -m tools.compute_causal_action_stats
#     - One process; reads all subdatasets and merges them.
#     - Writes ONLY --output. No files under DATASET_ROOT are created/updated.
#     - Uses --num-workers threads internally for reading/encoding.
#     - Exact quantiles are computed globally across the merged data.
#
#   MODE=parallel
#     python -m tools.compute_causal_action_stats_parallel
#     - One child process per subdataset (--dataset-processes controls concurrency).
#     - Each child writes <subdataset>/meta/causal_action_stats.json as an
#       intermediate/cache file; LeRobot's meta/stats.json is NOT touched.
#     - Then aggregates those per-subset files into --output.
#     - Group q01/q99 are min(subset q01) / max(subset q99), not global
#       sample quantiles; this matches the repo's parallel stats contract.
#     - Requires ALLOW_DATASET_WRITES=1 because it writes into DATASET_ROOT.
#
# Dataset presets:
#   DATASET=egosuite (default)
#     DATASET_ROOT=/mnt/sfs_turbo/public/datasets/egosuite_processed
#     ACTION_FROM_STATE=false, ACTION_TIME_OFFSET_STEPS=0
#   DATASET=agibot
#     DATASET_ROOT=/mnt/sfs_turbo/public/datasets/agibot_processed/v0
#     ACTION_FROM_STATE=true, ACTION_TIME_OFFSET_STEPS=1
#   DATASET=custom
#     Requires DATASET_ROOT; read options default to EgoSuite-like values and
#     can be overridden explicitly.
#
# What to change for a different dataset:
#   - DATASET preset, or DATASET_ROOT + DATASET_NAME.
#   - If the field names differ: STATE_KEY, ACTION_KEY, STATE_MASK_KEY,
#     ACTION_MASK_KEY.
#   - If the target semantics differ: ACTION_FROM_STATE and
#     ACTION_TIME_OFFSET_STEPS.
#   - If the training split changes: SPLIT_VAL_RATIO and SPLIT_SEED.
#
# What usually stays fixed even for another dataset or workdir:
#   - METHOD=exact (formal statistics; reservoir is debug/approximate)
#   - BOUNDS=q01_q99 (or min_max if the training recipe uses it)
#   - SEED=42, SPLIT_SEED=42 (must match the training recipe)
#   - TEMPLATE default unified55 (change only if the template changes)
#   - NUM_WORKERS=4, BATCH_SIZE=512 (performance only; tune for memory/CPU)
#
# What to change per run:
#   - MODE=single|parallel
#   - RUN_ID, RUN_DIR, OUTPUT_PATH, LOG_PATH
#   - DATASET_PROCESSES (parallel only)
#
# Defaults:
#   LOG_ROOT=/mnt/sfs_turbo/liujin/training_logs/action_midtrain
#   RUN_ID=dataset_stats
#   RUN_DIR=${LOG_ROOT}/${RUN_ID}_<timestamp>
#   OUTPUT_PATH=${RUN_DIR}/stats/${DATASET_NAME}_group_stats.json
#   LOG_PATH=${RUN_DIR}/stats/${DATASET_NAME}.log
#
# Usage:
#   bash examples/action_pretrain/liujin_compute_dataset_stats.sh
#   PRINT_ONLY=1 bash examples/action_pretrain/liujin_compute_dataset_stats.sh
#   DATASET=agibot bash examples/action_pretrain/liujin_compute_dataset_stats.sh
#   MODE=parallel ALLOW_DATASET_WRITES=1 DATASET=agibot \
#     bash examples/action_pretrain/liujin_compute_dataset_stats.sh
#   DATASET=custom DATASET_ROOT=/path/to/data DATASET_NAME=my_data \
#     ACTION_FROM_STATE=false ACTION_TIME_OFFSET_STEPS=0 \
#     bash examples/action_pretrain/liujin_compute_dataset_stats.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

DATASET="${DATASET:-egosuite}"
case "$DATASET" in
    egosuite)
        DEFAULT_DATASET_ROOT=/mnt/sfs_turbo/public/datasets/egosuite_processed
        DEFAULT_ACTION_FROM_STATE=false
        DEFAULT_ACTION_TIME_OFFSET_STEPS=0
        ;;
    agibot)
        DEFAULT_DATASET_ROOT=/mnt/sfs_turbo/public/datasets/agibot_processed/v0
        DEFAULT_ACTION_FROM_STATE=true
        DEFAULT_ACTION_TIME_OFFSET_STEPS=1
        ;;
    custom)
        DEFAULT_DATASET_ROOT=""
        DEFAULT_ACTION_FROM_STATE=false
        DEFAULT_ACTION_TIME_OFFSET_STEPS=0
        ;;
    *)
        printf 'Unknown DATASET preset: %s (use egosuite, agibot, or custom)\n' "$DATASET" >&2
        exit 2
        ;;
esac

DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
if [[ -z "$DATASET_ROOT" ]]; then
    printf 'DATASET=custom requires DATASET_ROOT\n' >&2
    exit 2
fi
DATASET_NAME="${DATASET_NAME:-$DATASET}"

STATE_KEY="${STATE_KEY:-state_unified}"
ACTION_KEY="${ACTION_KEY:-action_unified}"
STATE_MASK_KEY="${STATE_MASK_KEY:-mask_state}"
ACTION_MASK_KEY="${ACTION_MASK_KEY:-mask_action}"
ACTION_FROM_STATE="${ACTION_FROM_STATE:-$DEFAULT_ACTION_FROM_STATE}"
ACTION_TIME_OFFSET_STEPS="${ACTION_TIME_OFFSET_STEPS:-$DEFAULT_ACTION_TIME_OFFSET_STEPS}"
SPLIT_VAL_RATIO="${SPLIT_VAL_RATIO:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SEED="${SEED:-42}"
METHOD="${METHOD:-exact}"
BOUNDS="${BOUNDS:-q01_q99}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DATASET_PROCESSES="${DATASET_PROCESSES:-4}"
MODE="${MODE:-single}"

case "${ACTION_FROM_STATE,,}" in
    1|true|yes) ACTION_FROM_STATE_BOOL=true ;;
    0|false|no) ACTION_FROM_STATE_BOOL=false ;;
    *)
        printf 'ACTION_FROM_STATE must be true/false/1/0, got %s\n' "$ACTION_FROM_STATE" >&2
        exit 2
        ;;
esac

case "$MODE" in
    single|parallel) ;;
    *)
        printf 'MODE must be single or parallel, got %s\n' "$MODE" >&2
        exit 2
        ;;
esac

if [[ "$MODE" == "parallel" && "$DATASET_PROCESSES" -lt 1 ]]; then
    printf 'DATASET_PROCESSES must be positive\n' >&2
    exit 2
fi

# RUN_ID is a plain default string; the timestamp is appended to RUN_DIR below.
DEFAULT_RUN_ID="TEST"
LOG_ROOT="${LOG_ROOT:-/mnt/sfs_turbo/liujin/training_logs/action_midtrain}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
export LOG_ROOT
export RUN_ID
export RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-$RUN_DIR/stats/${DATASET_NAME}_group_stats.json}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/stats/${DATASET_NAME}.log}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/sfs_turbo/public/apps/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework-py312}"

READ_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --state-key "$STATE_KEY"
    --action-key "$ACTION_KEY"
    --state-mask-key "$STATE_MASK_KEY"
    --action-mask-key "$ACTION_MASK_KEY"
    --action-time-offset-steps "$ACTION_TIME_OFFSET_STEPS"
    --split-val-ratio "$SPLIT_VAL_RATIO"
    --split-seed "$SPLIT_SEED"
    --method "$METHOD"
    --seed "$SEED"
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
    --bounds "$BOUNDS"
)
if [[ "$ACTION_FROM_STATE_BOOL" == "true" ]]; then
    READ_ARGS+=(--action-from-state)
fi
READ_ARGS+=(--output "$OUTPUT_PATH")

if [[ "$MODE" == "parallel" ]]; then
    CMD=(python -m tools.compute_causal_action_stats_parallel "${READ_ARGS[@]}" --dataset-processes "$DATASET_PROCESSES")
    WRITE_BEHAVIOR="parallel: writes <subdataset>/meta/causal_action_stats.json (intermediate) + group stats"
else
    CMD=(python -m tools.compute_causal_action_stats "${READ_ARGS[@]}")
    WRITE_BEHAVIOR="single: writes only the group stats file; no dataset writes"
fi

printf 'Dataset preset: %s\n' "$DATASET"
printf 'Dataset root:   %s\n' "$DATASET_ROOT"
printf 'Dataset name:   %s\n' "$DATASET_NAME"
printf 'Mode:           %s\n' "$MODE"
printf 'Write behavior: %s\n' "$WRITE_BEHAVIOR"
printf 'Read options:   state=%s action=%s state_mask=%s action_mask=%s action_from_state=%s offset=%s\n' \
    "$STATE_KEY" "$ACTION_KEY" "$STATE_MASK_KEY" "$ACTION_MASK_KEY" "$ACTION_FROM_STATE_BOOL" "$ACTION_TIME_OFFSET_STEPS"
printf 'Run dir:        %s\n' "$RUN_DIR"
printf 'Stats output:   %s\n' "$OUTPUT_PATH"
printf 'Log:            %s\n' "$LOG_PATH"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

if [[ "$MODE" == "parallel" && "${ALLOW_DATASET_WRITES:-0}" != "1" ]]; then
    printf '\n' >&2
    printf 'Refusing parallel mode without ALLOW_DATASET_WRITES=1.\n' >&2
    printf 'Parallel mode writes meta/causal_action_stats.json into every subdataset under:\n' >&2
    printf '  %s\n' "$DATASET_ROOT" >&2
    printf 'Re-run with ALLOW_DATASET_WRITES=1 if that is intended, or use MODE=single.\n' >&2
    exit 2
fi

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/stats" "$(dirname -- "$OUTPUT_PATH")"

set +e
TORCH_DEVICE_BACKEND_AUTOLOAD=0 LD_LIBRARY_PATH='' \
    "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
exit_code=${PIPESTATUS[0]}
set -e

if [[ "$exit_code" -ne 0 ]]; then
    printf '%s statistics failed with exit code %s\n' "$DATASET_NAME" "$exit_code" >&2
    exit "$exit_code"
fi

python - "$OUTPUT_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
provenance = data.get("provenance", {})
print("Wrote:", path)
print("partial:", provenance.get("partial"))
print("template:", provenance.get("template_id"))
print("blocks:", provenance.get("blocks"))
print("available_blocks:", provenance.get("available_blocks"))
if provenance.get("partial") is not False:
    raise SystemExit("Statistics are partial; rerun without --limit")
PY
