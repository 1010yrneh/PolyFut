# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PolyFut desktop (Windows/macOS).

Build from repo root:
    powershell -File packaging/build_win.ps1
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent.parent
ICON = ROOT / "packaging" / "icons" / "polyfut.ico"

block_cipher = None

datas = [
    (str(ROOT / "index.html"), "."),
    (str(ROOT / "script.js"), "."),
    (str(ROOT / "style.css"), "."),
    (str(ROOT / "calculations.js"), "."),
    (str(ROOT / "help.js"), "."),
    (str(ROOT / "server.py"), "."),
    (str(ROOT / "keep_awake.py"), "."),
    (str(ROOT / "polyfut_video"), "polyfut_video"),
    (str(ROOT / "sample_data"), "sample_data"),
    (str(ROOT / "packaging" / "VERSION"), "packaging"),
]
logo = ROOT / "PolyFut Logo.png"
if logo.exists():
    datas.append((str(logo), "."))

hiddenimports = [
    "flask",
    "waitress",
    "cv2",
    "numpy",
    "ultralytics",
    "PIL",
    "yaml",
    "webview",
    "polyfut_video.main",
    "polyfut_video.pipeline.detection",
    "sklearn",
    "sklearn.cluster",
]

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PolyFut",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PolyFut",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PolyFut.app",
        icon=str(ICON) if ICON.is_file() else None,
        bundle_identifier="com.polyfut.app",
    )
