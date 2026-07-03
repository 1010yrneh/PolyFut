"""Keep the OS awake while match analysis runs.

Allows the display/monitor to turn off; prevents system idle sleep so long
runs continue until the machine is fully shut down or the job finishes.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

_lock = threading.Lock()
_active = 0
_state: dict[str, Any] = {}


def _win_acquire() -> None:
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    prev = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )
    _state["win_prev"] = prev


def _win_release() -> None:
    import ctypes

    ES_CONTINUOUS = 0x80000000
    prev = _state.pop("win_prev", None)
    if prev is not None:
        ctypes.windll.kernel32.SetThreadExecutionState(prev)
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _mac_acquire() -> None:
    if _state.get("mac_proc"):
        return
    _state["mac_proc"] = subprocess.Popen(
        ["caffeinate", "-dims"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _mac_release() -> None:
    proc = _state.pop("mac_proc", None)
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _platform_acquire() -> None:
    if sys.platform == "win32":
        _win_acquire()
    elif sys.platform == "darwin":
        _mac_acquire()


def _platform_release() -> None:
    if sys.platform == "win32":
        _win_release()
    elif sys.platform == "darwin":
        _mac_release()


def acquire() -> None:
    """Register one active analysis job; block system sleep when first job starts."""
    global _active
    with _lock:
        _active += 1
        if _active == 1:
            _platform_acquire()


def release() -> None:
    """Unregister a job; allow sleep again when no jobs remain."""
    global _active
    with _lock:
        if _active <= 0:
            return
        _active -= 1
        if _active == 0:
            _platform_release()


class during_analysis:
    """Context manager used around each CV job thread."""

    def __enter__(self) -> during_analysis:
        acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        release()
