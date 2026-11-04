#!/usr/bin/env bash
# SPDX-License-Identifier: OpenMDW-1.1
# Robot-action / video-only co-training; other environment options match the action launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${HUMAN_VIDEO_DATASET_PATH:?Set HUMAN_VIDEO_DATASET_PATH to a LeRobot video dataset or manifest}"
if [[ "$HUMAN_VIDEO_DATASET_PATH" != /* ]]; then
    HUMAN_VIDEO_DATASET_PATH="$PWD/$HUMAN_VIDEO_DATASET_PATH"
fi
export HUMAN_VIDEO_DATASET_PATH
export TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/action_video_midtrain_edge_causal_tnd.toml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_action/action_video_midtrain}"
exec bash "$SCRIPT_DIR/launch_midtrain_action_causal.sh" "$@"
