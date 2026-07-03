"""Tests for keep_awake reference counting."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keep_awake import acquire, release, during_analysis


def test_wake_lock_refcount():
    acquire()
    acquire()
    release()
    release()
    with during_analysis():
        pass
