#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Activate the existing RoboTwin simulator environment before running this script.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CMD=("${PYTHON_BIN:-python}" "$SCRIPT_DIR/robotwin_client.py"
    --robotwin-root "${ROBOTWIN_ROOT:-/path/to/RoboTwin}"
    --server-uri "${SERVER_URI:-ws://127.0.0.1:8000}"
    --task-name "${TASK_NAME:-adjust_bottle}" --task-config "${TASK_CONFIG:-demo_clean}"
    --control-mode "${CONTROL_MODE:-joint}" --state-source "${STATE_SOURCE:-measured}"
    --episodes "${EPISODES:-1}" --seed "${SEED:-100000}"
    --servo-hz "${SERVO_HZ:-30}" --request-timeout "${REQUEST_TIMEOUT:-120}"
    --latency-factor "${LATENCY_FACTOR:-1.2}" --margin-steps "${MARGIN_STEPS:-1}"
    --extra-latency-seconds "${EXTRA_LATENCY_SECONDS:-0}"
    --minimum-execution-hz "${MINIMUM_EXECUTION_HZ:-0}"
    --output "${OUTPUT_ROOT:-outputs/causal_action/robotwin_sim}")
[[ -n "${EXECUTION_HZ:-}" ]] && CMD+=(--execution-hz "$EXECUTION_HZ")
[[ -n "${MAX_WAYPOINTS:-}" ]] && CMD+=(--max-waypoints "$MAX_WAYPOINTS")
[[ -n "${MAX_STEPS:-}" ]] && CMD+=(--max-steps "$MAX_STEPS")
[[ -n "${INSTRUCTION:-}" ]] && CMD+=(--instruction "$INSTRUCTION")
case "${ENABLE_RTC:-0}" in
    1) CMD+=(--enable_rtc) ;;
    0) ;;
    *) printf 'ENABLE_RTC must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${ADAPTIVE_EXECUTION:-1}" in
    1) ;;
    0) CMD+=(--no-adaptive-execution) ;;
    *) printf 'ADAPTIVE_EXECUTION must be 0 or 1\n' >&2; exit 2 ;;
esac
CMD+=("$@")
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "${CMD[@]}"
