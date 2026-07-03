"""PolyFut desktop launcher — Flask server + pywebview shell.

Usage (dev):
    pip install pywebview
    python launcher.py

Packaged builds set POLYFUT_DATA_DIR to the user app-data folder automatically.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MUTEX_NAME = "Global\\PolyFut.SingleInstance.v1"


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


ROOT = _app_root()


def _read_version() -> str:
    for candidate in (
        Path(__file__).resolve().parent / "packaging" / "VERSION",
        ROOT / "packaging" / "VERSION",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return "dev"


def _pick_data_dir() -> Path:
    env = os.environ.get("POLYFUT_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "PolyFut"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PolyFut"
    return Path.home() / ".local" / "share" / "PolyFut"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _single_instance_guard() -> bool:
    """Return True if this process should continue; False if another instance is running."""
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    mutex = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        user32.MessageBoxW(
            0,
            "PolyFut is already running.\nCheck your taskbar for the open window.",
            "PolyFut",
            0x40,
        )
        return False
    return True


def _wait_for_server(port: int, timeout_sec: float = 8.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _analysis_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/process/active", timeout=1.5,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("runs"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def _warn_analysis_still_running() -> None:
    msg = (
        "Match analysis is still running.\n\n"
        "You can turn off the display or lock the screen — analysis will keep going.\n"
        "Only a full shutdown or sleep will stop it.\n\n"
        "Minimize this window and leave PolyFut open until analysis finishes."
    )
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "PolyFut", 0x40)
    else:
        print(msg)


def main() -> int:
    if not _single_instance_guard():
        return 1

    version = _read_version()
    data_dir = _pick_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("POLYFUT_DATA_DIR", str(data_dir))

    port = int(os.environ.get("POLYFUT_PORT", "0"))
    if port <= 0:
        port = _free_port()
    os.environ["POLYFUT_PORT"] = str(port)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import server  # noqa: WPS433

    def _run_flask() -> None:
        try:
            from waitress import serve
            serve(
                server.app,
                host="127.0.0.1",
                port=port,
                threads=4,
                channel_timeout=7200,
            )
        except ImportError:
            server.app.run(
                host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False,
            )

    t = threading.Thread(target=_run_flask, daemon=False)
    t.start()

    url = f"http://127.0.0.1:{port}/"
    if not _wait_for_server(port):
        print(f"Server did not start in time. Try opening {url} manually.")
        print(f"Data dir: {data_dir}")
        return 1

    window_title = f"PolyFut {version}" if version != "dev" else "PolyFut"

    try:
        import webview  # noqa: WPS433
    except ImportError:
        print(f"pywebview not installed. Open {url} in your browser.")
        print(f"Data dir: {data_dir}")
        t.join()
        return 0

    window = webview.create_window(
        window_title,
        url,
        width=1280,
        height=800,
        min_size=(960, 640),
        text_select=True,
    )

    def on_closing() -> bool:
        if _analysis_running(port):
            _warn_analysis_still_running()
            return False
        return True

    window.events.closing += on_closing
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
