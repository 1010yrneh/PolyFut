#!/usr/bin/env bash
# Build PolyFut.app on macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt
pip install -e cv/
pip install pyinstaller pywebview

pyinstaller packaging/pyinstaller.spec --noconfirm

echo "Output: dist/PolyFut.app (or dist/PolyFut/)"
echo "Copy bundled ffmpeg + OpenVINO weights before codesign/notarization."
