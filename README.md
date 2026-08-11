# PolyFut

A self-contained match-analysis app. Upload a broadcast match video and the **Level 1**
`polyfut_video` pipeline finds team-possession windows, then filters to your team's key
moments for review and action logging.

Everything runs on your machine — no video is uploaded anywhere.

## User journey

1. **Import** — pick the match video (+ optional position, opponent, score). Read off disk; nothing is uploaded.
2. **Playing time** — mark the stretches you were on the pitch. Bounds the analysis, and stops the seed step asking you to identify yourself in a passage you were not in.
3. **Team colour** — both kits are read off the footage; pick yours, the other becomes the opposition.
4. **Pitch calibration** — click a few painted-line corners so "who touched it" is judged in metres, not pixels. Skippable.
5. **Choose who you are** — tap yourself in a few short clips. This is the identity anchor everything downstream hangs on.
6. **CV analysis** — the local pipeline runs (shot filter → ball trajectory → contacts → attribution → hotspots). Roughly half the footage length on a laptop CPU.
7. **Review** — flick through candidate touches and drop the ones that are not you. The pipeline over-includes on purpose and asks.
8. **Score** — log actions during possession windows; live hybrid stats update.
9. **Report** — FINISH → WPA chart, hybrid breakdown, optional AI scout report.

## Quick start (developer)

```bash
git clone https://github.com/YOUR_USERNAME/PolyFut.git
cd PolyFut
pip install -r requirements.txt
python server.py
# open http://localhost:5000
```

First real run downloads `yolov8n.pt` (~6 MB) via Ultralytics automatically.

Set `POLYFUT_DEVICE=cuda` when a GPU is available (RTX 3060+ recommended for 90-min matches).

### Desktop app

```bash
pip install pywebview
python launcher.py
```

Opens a native window (taskbar app) — no browser tab or PowerShell needed.

### Ship Windows installer (polyfut.com)

```powershell
# One-time: install Inno Setup 6 from https://jrsoftware.org/isinfo.php
powershell -File packaging/build_win.ps1
```

Output goes to `%LOCALAPPDATA%\PolyFut-build\` — **not** into the repo. Building inside the
OneDrive-synced working tree fails: OneDrive holds file handles mid-sync and PyInstaller dies
clearing its own output directory. Override with `POLYFUT_BUILD_DIR`.

- `%LOCALAPPDATA%\PolyFut-build\dist\PolyFut\PolyFut.exe` — portable folder
- `%LOCALAPPDATA%\PolyFut-build\dist\PolyFut-Setup-<VERSION>.exe` — the installer

The build **runs what it just built** before Inno Setup packages it: it launches the exe and
hits `/api/selftest`, which loads the real model and runs inference on several threads. A build
that cannot analyse does not become an installer. Four bugs shipped past a green build before
this existed — a stale icon, a stale `index.html`, an `msvcp140.dll` too old for torch, and a
shared OpenVINO infer request that crashed nine minutes into a run. Set `POLYFUT_SKIP_SMOKE=1`
to bypass while iterating, never for a build anyone else installs.

**Releasing.** Bump `packaging/VERSION`; `sync_version.ps1` propagates it to the Inno script,
`website/version.json`, and the download button in `website/index.html`. Then attach the
installer to a GitHub Release tagged `v<VERSION>`:

```powershell
gh release create v1.0.1 "$env:LOCALAPPDATA\PolyFut-build\dist\PolyFut-Setup-1.0.1.exe" --title "PolyFut 1.0.1"
```

The installer is a **Release asset, not a file on the site**: Pages caps a site at 1 GB and git
rejects anything over 100 MB, so a ~400 MB installer can never live in `website/`.
`website/downloads/*.exe` is gitignored and is only a local staging spot.

The site deploys itself — pushing to `main` runs `.github/workflows/deploy-website.yml`, which
publishes `website/` to GitHub Pages. It only fires when `website/` actually changes.

User data (videos, exports, saved jobs) lives in `%APPDATA%\PolyFut\` after install.

### Try the UI without the CV stack

```powershell
$env:POLYFUT_FAKE_CV=1; python server.py
```

### Run pipeline CLI directly

```bash
python -m polyfut_video.main --video match.mp4 --out output --my-team team_a
```

### Tests

```bash
cd PolyFut
pip install pytest
pytest polyfut_video/tests -q
```

## Layout

```
PolyFut/
├── polyfut_v2/             # current ball-anchored single-player pipeline
├── polyfut_video/          # v1 pipeline + shared decode/shot/team utilities
├── index.html / script.js / style.css / calculations.js
├── server.py / launcher.py
└── uploads/ exports/
```
