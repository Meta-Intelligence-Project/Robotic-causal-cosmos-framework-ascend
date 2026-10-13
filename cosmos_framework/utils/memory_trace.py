# SPDX-License-Identifier: OpenMDW-1.1
"""Lightweight training memory markers; inactive without the MemoryTrace callback."""

from collections.abc import Callable
from typing import Any

_recorder: Callable[[str, dict[str, Any]], None] | None = None


def set_memory_recorder(recorder: Callable[[str, dict[str, Any]], None] | None) -> None:
    global _recorder
    _recorder = recorder


def record_memory_phase(phase: str, **metadata: Any) -> None:
    # Callers supply only scalar/shape metadata, never tensors or computation graphs.
    if _recorder is not None:
        _recorder(phase, metadata)
