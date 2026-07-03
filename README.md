# PolyFut

A self-contained match-analysis app. Upload a broadcast match video and the **Level 1**
`polyfut_video` pipeline finds team-possession windows, then filters to your team's key
moments for review and action logging.

Everything runs on your machine — no video is uploaded anywhere.

## User journey

1. **Import** — upload match video (+ optional position, opponent, score).
2. **Team pick** — choose Team A or Team B (the side you played for).
3. **CV analysis** — 9-stage local pipeline (shot filter → detect → track → DBSCAN teams → possession).
4. **Review** — click a key moment; playback clamps to that window.
5. **Score** — log actions during possession windows; live hybrid stats update.
6. **Report** — FINISH → WPA chart, hybrid breakdown, optional Groq AI scout report.

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

Outputs:

- `dist/PolyFut/PolyFut.exe` — portable folder
- `dist/PolyFut-Setup-1.0.0.exe` — Windows installer (if Inno Setup is installed)
- `website/downloads/` — copy installer here for the download page

Deploy the `website/` folder to Vercel, Netlify, or GitHub Pages and point **polyfut.com** at it.
Bump `packaging/VERSION` before each release; `sync_version.ps1` updates the installer and site metadata.

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
├── polyfut_video/          # Level 1 pipeline (active)
├── cv/                     # legacy polyfut_cv (reference)
├── index.html / script.js / style.css / calculations.js
├── server.py / launcher.py
└── uploads/ exports/
```
