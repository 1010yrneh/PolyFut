# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PolyFut desktop (Windows/macOS).

Build from repo root:
    powershell -File packaging/build_win.ps1
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is the directory CONTAINING this spec (packaging/), so the repo root
# is one level up, not two. With .parent.parent every path here pointed at the
# directory above the repo: the build died on "the soccer model is missing"
# looking in <parent-of-repo>/models, and had it somehow got past that it would
# have gone looking for index.html up there too.
ROOT = Path(SPECPATH).resolve().parent
if not (ROOT / "server.py").exists():          # fail loudly, not mysteriously
    raise SystemExit(
        f"pyinstaller.spec: expected the repo root at {ROOT}, but server.py is "
        f"not there. SPECPATH={SPECPATH!r}"
    )
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
    # The v2 pipeline. It used to arrive only because PyInstaller follows
    # launcher.py -> server -> polyfut_v2, which is implicit and breaks the
    # moment an import becomes lazy or conditional. Named outright now.
    (str(ROOT / "polyfut_v2"), "polyfut_v2"),
    (str(ROOT / "sample_data"), "sample_data"),
    (str(ROOT / "packaging" / "VERSION"), "packaging"),
]

# The soccer ball+player model, SHIPPED rather than downloaded on first run.
# The installer targets Program Files, which a normal user cannot write to, so
# a fresh install could not fetch it: the download would fail and the app would
# fall back to the general COCO model, which barely detects a soccer ball. This
# is ~64MB of the installer and buys an instant, offline, admin-free first run.
# Path must match polyfut_v2.ball_model._models_root().
_MODELS = ROOT / "models"
_model_files = [
    _MODELS / "soccer_uisikdag.pt",
    _MODELS / "soccer_uisikdag_openvino_model",
]
_missing = [p for p in _model_files if not p.exists()]
if _missing:
    raise SystemExit(
        "Cannot build a shippable PolyFut: the soccer model is missing.\n"
        + "\n".join(f"  {p}" for p in _missing)
        + "\n\nRun the app once from source to download and export it, then "
          "rebuild. Shipping without it means every install silently "
          "downgrades to the COCO model."
    )
datas += [
    (str(_MODELS / "soccer_uisikdag.pt"), "models"),
    (str(_MODELS / "soccer_uisikdag_openvino_model"),
     "models/soccer_uisikdag_openvino_model"),
]

# The team-kit picker runs a general person detector, not the soccer model:
# server.py sets WEIGHTS = ROOT/"yolov8s.pt". It was never bundled, so in a
# frozen build that path did not exist and Ultralytics quietly downloaded 21.5MB
# from GitHub on first use — caught by watching the packaged app's own log.
# That is a first-run internet dependency for a "runs entirely on your computer"
# app, and it writes into the install directory, which is not writable when
# PolyFut is installed for all users. Bundle it, and refuse to build without it
# for the same reason as the soccer model above.
_YOLO = ROOT / "yolov8s.pt"
if not _YOLO.exists():
    raise SystemExit(
        f"Cannot build a shippable PolyFut: {_YOLO} is missing.\n"
        "It is the detector the team-kit picker uses. Run the app once from "
        "source to fetch it, then rebuild — otherwise every install downloads "
        "it on first use, which fails offline and in a read-only install "
        "directory."
    )
datas.append((str(_YOLO), "."))
logo = ROOT / "PolyFut Logo.png"
if logo.exists():
    datas.append((str(logo), "."))

# Images the MARKUP actually asks for. The line above bundles a file nothing
# references by that name, so the installed app was serving 404 for every image
# it uses: the header logo and all six help screenshots. Found by watching the
# network log of the running installed app - the page still renders, so nothing
# fails loudly; you just get broken images in How to Use.
#
# Derived from index.html and help.js rather than listed by hand, so adding an
# image to the markup cannot silently miss the bundle again. Missing files fail
# the build for the same reason the model guard does: shipping is the wrong
# place to discover it.
import re as _re

_referenced = set()
for _src in (ROOT / "index.html", ROOT / "help.js"):
    if _src.exists():
        _referenced |= set(_re.findall(
            r'src="([^":]+\.(?:png|jpg|jpeg|svg|gif|webp))"',
            _src.read_text(encoding="utf-8")))

_missing_img = []
for _rel in sorted(_referenced):
    _path = ROOT / _rel
    if not _path.exists():
        _missing_img.append(_rel)
        continue
    # Preserve the sub-path so assets/logo.png resolves as assets/logo.png.
    _dest = str(Path(_rel).parent) if Path(_rel).parent != Path(".") else "."
    datas.append((str(_path), _dest))

if _missing_img:
    raise SystemExit(
        "Cannot build a shippable PolyFut: the markup references images that "
        "are not in the repo.\n"
        + "\n".join(f"  {m}" for m in _missing_img)
        + "\n\nEither add them or stop referencing them - a build without them "
          "ships broken images."
    )

# AI proxy wiring (see ai_backend/README.md). Gitignored and per-deployment, so
# it won't exist until someone has run through the Modal setup — bundle it when
# present, but don't fail the build without it: a build with no ai_config.json
# just ships with the AI proxy disabled and falls back to the bring-your-own-key
# flow, same as before this feature existed.
ai_config = ROOT / "ai_config.json"
if ai_config.exists():
    datas.append((str(ai_config), "."))

hiddenimports = [
    "flask",
    "waitress",
    "cv2",
    "numpy",
    "ultralytics",
    "PIL",
    "yaml",
    "webview",
    # (polyfut_video.main was listed here and does not exist — the build logged
    # "ERROR: Hidden import 'polyfut_video.main' not found" on every run.
    # collect_submodules("polyfut_video") below covers the real modules.)
    "polyfut_video.pipeline.detection",
    "sklearn",
    "sklearn.cluster",
    # pitch_calibration solves the camera pose with least_squares, imported
    # inside the function so it never showed up in the import graph.
    "scipy.optimize",
]
# Every module of both pipelines, so a lazy or conditional import cannot leave
# one out of the build.
hiddenimports += collect_submodules("polyfut_v2")
hiddenimports += collect_submodules("polyfut_video")

# OpenVINO is loaded by name from inside ultralytics, so the import graph never
# sees it. Without it the soccer model runs on PyTorch instead — measured at
# roughly 6x slower on this CPU, i.e. the difference between a 5-minute clip
# taking 8 minutes and taking most of an hour. collect_all because it ships
# binaries and plugin descriptors, not just Python.
try:
    _ov_datas, _ov_binaries, _ov_hidden = collect_all("openvino")
    datas += _ov_datas
    hiddenimports += _ov_hidden
except Exception as exc:  # noqa: BLE001 - build must say so, not guess
    raise SystemExit(
        f"Cannot build a shippable PolyFut: openvino is not importable ({exc}).\n"
        "Install it (pip install -r requirements.txt) and rebuild — without it "
        "every analysis runs several times slower."
    )
else:
    binaries_extra = _ov_binaries

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries_extra,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Development-only weight that would otherwise ride along inside the
    # scientific stack. Nothing in the shipped app imports these.
    excludes=[
        "matplotlib", "tkinter", "IPython", "jupyter", "notebook",
        "pytest", "_pytest", "pandas", "seaborn",
    ],
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
