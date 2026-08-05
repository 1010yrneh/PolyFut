"""Debug scaffolding must stay off in a shipped build.

_dbg_log defaulted to ON and wrote JSONL to ROOT/.cursor/, which in an installed
build is the app's own directory. Installing 1.0.0 and running one kit detection
dropped a debug-*.log into %LOCALAPPDATA%\\Programs\\PolyFut\\_internal\\.cursor\\.
That path is writable for a per-user install and read-only for an all-users one,
so it was a crash waiting for the first admin install — and it logged upload
sizes and tokens on every run with nobody reading them.
"""

from __future__ import annotations

import importlib
import os
import sys


def _fresh_server(monkeypatch, debug_path=None):
    if debug_path is None:
        monkeypatch.delenv("POLYFUT_DEBUG_LOG", raising=False)
    else:
        monkeypatch.setenv("POLYFUT_DEBUG_LOG", str(debug_path))
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def test_off_by_default(monkeypatch):
    srv = _fresh_server(monkeypatch)
    assert srv._DEBUG_LOG is None


def test_writes_nothing_when_off(monkeypatch, tmp_path):
    """The guarantee that matters: no file appears anywhere."""
    srv = _fresh_server(monkeypatch)
    before = set(os.listdir(tmp_path))
    srv._dbg_log("H1", "test", "should not be written", {"a": 1})
    assert set(os.listdir(tmp_path)) == before
    assert not list(tmp_path.rglob("debug-*.log"))


def test_never_writes_into_the_app_directory(monkeypatch):
    """Whatever it does when enabled, the default must not target the install
    directory — that is the read-only case on an all-users install."""
    srv = _fresh_server(monkeypatch)
    assert srv._DEBUG_LOG is None, (
        "a default path would live under ROOT, which is the application "
        "directory in a frozen build"
    )


def test_can_still_be_switched_on_for_debugging(monkeypatch, tmp_path):
    target = tmp_path / "dbg.log"
    srv = _fresh_server(monkeypatch, target)
    srv._dbg_log("H1", "test", "hello", {"n": 7})
    assert target.is_file()
    assert "hello" in target.read_text(encoding="utf-8")


def test_enabling_it_never_raises(monkeypatch, tmp_path):
    """It is instrumentation; a bad path must not take the server down."""
    srv = _fresh_server(monkeypatch, tmp_path / "no" / "such" / "dir" / "x.log")
    srv._dbg_log("H1", "test", "swallowed", {})   # must not raise
