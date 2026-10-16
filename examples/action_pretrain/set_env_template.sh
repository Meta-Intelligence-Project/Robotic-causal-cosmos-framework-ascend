#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。
# 使用 source examples/action_pretrain/set_env_template.sh 加载到当前 shell。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ASCEND_ENV="${ASCEND_ENV:-/path/to/ascend-toolkit/set_env.sh}"
export CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
export CONDA_ENV="${CONDA_ENV:-cosmos-framework}"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/robotwin_processed}"
export RUN_DIR="${RUN_DIR:-$REPO_ROOT/outputs/my_robotwin_run}"
export ROBOTWIN_GROUP_STATS_PATH="${ROBOTWIN_GROUP_STATS_PATH:-$RUN_DIR/stats/robotwin_group_stats.json}"
export SOURCE_SET="${SOURCE_SET:-robotwin}"

export TMPDIR="${TMPDIR:-/tmp/cosmos-action-stats}"
mkdir -p "$RUN_DIR/stats" "$TMPDIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
