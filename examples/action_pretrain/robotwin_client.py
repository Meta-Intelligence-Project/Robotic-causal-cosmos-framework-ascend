# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run in the existing RoboTwin Python/CUDA/SAPIEN environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cosmos_framework.inference.causal_action.robotwin_client import main

if __name__ == "__main__":
    main()
