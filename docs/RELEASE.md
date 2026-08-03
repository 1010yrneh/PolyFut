# Releasing PolyFut

The shippable artifact is **one Windows installer**, `PolyFut-Setup-<version>.exe`,
served from `polyfut.com/downloads/`. There is no source distribution and no
separate "user version" of the tree — the build script reads this repo directly,
so a parallel copy would only drift from the thing that actually gets built.

```
build_win.ps1
  ├─ pip install -r requirements.txt + pyinstaller + pywebview
  ├─ make_icon.py                    → packaging/icons/polyfut.ico
  ├─ sync_version.ps1                → packaging/VERSION into the .iss
  ├─ pyinstaller packaging/pyinstaller.spec
  │                                  → dist/PolyFut/            (portable folder)
  ├─ copies ffmpeg.exe if one is on PATH
  └─ Inno Setup (ISCC.exe)
                                     → dist/PolyFut-Setup-<version>.exe
                                     → copied into website/downloads/
```

The site's download button needs no editing: `website/index.html` reads
`website/version.json` and builds the filename from it.

---

## One-time setup on the build machine

| | |
|---|---|
| Python 3.12 with `requirements.txt` installed | the build script does this |
| `pip install pyinstaller "pywebview>=5.0"` | the build script does this |
| **Inno Setup 6** — https://jrsoftware.org/isinfo.php | **manual, one time** |
| ffmpeg on `PATH` (optional) | bundled into the app if found |

Without Inno Setup the build still produces `dist/PolyFut/`, a working portable
folder — but no installer, so nothing to publish.

---

## The model is shipped, not downloaded

`models/` (~64MB: the soccer `.pt` plus its OpenVINO export) is bundled into the
installer, and the spec **refuses to build without it**.

This is not an optimisation. The installer targets `Program Files`, which a
normal user cannot write to, so an app that tried to download the model on first
run would fail there and silently fall back to the general COCO model — which
barely detects a soccer ball. Every analysis would look broken for reasons the
user could never diagnose.

If `models/` is missing on the build machine, run the app once from source: it
downloads and exports on first use, into the same folder the spec then bundles.

`polyfut_v2.ball_model._models_root()` resolves this for both cases — from
source it walks up from `__file__`, and in a frozen build it reads
`sys._MEIPASS`. If you move `models/` in the spec, move it there too.

---

## Release checklist

1. `pytest polyfut_v2/tests polyfut_video/tests -q` — all green. CI runs this on
   every push, but run it locally before tagging.
2. **A full run through the app on real footage**, start to finish: upload →
   team pick → seed → (optional) pitch calibration → review → hotspots.
   Unit tests do not cover the wiring between screens.
3. Bump `packaging/VERSION` **and** `website/version.json` together. They are
   the same number in two places and nothing enforces that.
4. `powershell -File packaging/build_win.ps1`
5. **Install the result on a machine that has never run PolyFut.** This is the
   only test that catches a missing bundled dependency — a dev machine has the
   model, the weights and the Python packages lying around already, and will
   happily hide all three.
6. Confirm on that clean machine: the app opens, an analysis runs, and the run's
   warnings contain **no** "soccer ball model unavailable" or "running on
   PyTorch (OpenVINO unavailable)". Either means the bundle is incomplete and
   every user gets a slow or near-useless analysis.
7. Publish `website/` (the installer is already copied into
   `website/downloads/` by the build script).

---

## What ships and what does not

Bundled: both pipelines, the soccer model, OpenVINO, `sample_data/`, the web UI
(`index.html`, `script.js`, `style.css`, `calculations.js`, `help.js`).

Excluded in the spec: `matplotlib`, `tkinter`, `IPython`, `jupyter`, `pytest`,
`pandas`, `seaborn` — development weight that otherwise rides along inside the
scientific stack.

Never in the installer or in git: `uploads/`, `exports/`, `notebooks/`, `cv/`,
`output_v2*/`, and the `.exe` itself (it is hosted, not committed).

---

## Known limits to set expectations against

- **Windows only.** `packaging/build_mac.sh` exists but is untested; the site
  copy says "Requires Windows" and should keep saying so.
- **Footage quality is the ceiling.** At 640×360 a player is ~26px tall; kit
  colour and same-kit identity are unreliable there by measurement, not by bug,
  and the review queue is how the product stays correct anyway. See
  `docs/detection-issues.md` for the numbers.
- **Runtime is roughly 1.5–2× the clip length** on a CPU with OpenVINO, and
  several times that if the machine is busy — disk contention hits the decode
  stages hardest.
