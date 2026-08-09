# NOTE: keep this file ASCII-only and BOM-prefixed. Windows
# PowerShell 5.1 reads a BOM-less .ps1 as the ANSI codepage, where a
# UTF-8 em-dash decodes to a smart quote -- which PowerShell accepts
# as a string delimiter, so one dash in a comment or message broke
# parsing of the whole script before any step ran.
# Build PolyFut desktop app on Windows (PyInstaller + optional Inno Setup).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Packaging = Join-Path $Root "packaging"
Set-Location $Root

# Build OUTSIDE the repo. This tree is inside a synced OneDrive folder, and
# dist/ is 1.4GB of disposable output: syncing it silently killed a build with a
# PermissionError, stretched an Inno compile from ~13 minutes to 9.75 hours, and
# set off a bulk-delete warning on the user's cloud storage. It is git-ignored,
# so syncing it buys nothing. Override with POLYFUT_BUILD_DIR if you want it
# somewhere else; only the finished installer is copied back into the repo.
$BuildRoot = $env:POLYFUT_BUILD_DIR
if (-not $BuildRoot) { $BuildRoot = Join-Path $env:LOCALAPPDATA "PolyFut-build" }
$Dist = Join-Path $BuildRoot "dist"
$Work = Join-Path $BuildRoot "build"
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Write-Host "Build output: $BuildRoot"

Write-Host "== PolyFut Windows build ==" -ForegroundColor Cyan

Write-Host "Installing build dependencies..."
pip install -r requirements.txt
pip install pyinstaller "pywebview>=5.0"

Write-Host "Generating app icon..."
python (Join-Path $Packaging "make_icon.py")

Write-Host "Syncing installer version..."
& (Join-Path $Packaging "sync_version.ps1")

# Clear dist first. PyInstaller cleans it itself, but it does so mid-run and
# dies with PermissionError if anything holds a handle - OneDrive syncing the
# 1.4GB tree is enough, and so is having just run the app from it. Retrying
# helps because the lock is transient.
# Both dist AND the work dir. PyInstaller caches an icon-patched copy of the
# bootloader at build/<spec>/PolyFut.exe and reuses it when it believes nothing
# relevant changed - a changed .ico is NOT part of that cache key. Clearing only
# dist meant a new icon could never reach the exe: the build reported success,
# wrote a fresh dist, and the binary in it was a byte copy of the cached one
# carrying the previous icon. Caught by extracting the RT_ICON resource from
# the built exe and finding the old artwork still in it.
foreach ($dir in @($Dist, $Work)) {
  if (Test-Path $dir) {
    Write-Host "Clearing $dir..."
    foreach ($attempt in 1..5) {
        try {
            Remove-Item $dir -Recurse -Force -ErrorAction Stop
            break
        } catch {
            if ($attempt -eq 5) {
                throw ("Could not clear $dir after 5 attempts: " +
                       $_.Exception.Message +
                       " - close PolyFut.exe and pause OneDrive sync, then retry.")
            }
            Write-Host "  locked, retrying in 3s ($attempt/5)..."
            Start-Sleep -Seconds 3
        }
    }
  }
}

$SpecPath = Join-Path $Packaging "pyinstaller.spec"
Write-Host "Running PyInstaller..."
pyinstaller $SpecPath --noconfirm --distpath $Dist --workpath $Work

# A native command's exit code does NOT trip $ErrorActionPreference, so this
# has to be explicit. Without it a failed PyInstaller was invisible and Inno
# Setup packaged whatever happened to be in dist from a previous build.
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$DistExe = Join-Path $Dist "PolyFut\PolyFut.exe"
if (-not (Test-Path $DistExe)) {
    throw "Build failed: $DistExe not found"
}
# Existence is not enough - it was the stale-output trap. Require the exe to be
# newer than the spec that was supposed to produce it.
$ExeTime = (Get-Item $DistExe).LastWriteTime
$SpecTime = (Get-Item $SpecPath).LastWriteTime
if ($ExeTime -lt $SpecTime) {
    throw ("Build failed: $DistExe ($ExeTime) is older than " +
           "pyinstaller.spec ($SpecTime) - PyInstaller did not rebuild.")
}
Write-Host "Built: $DistExe" -ForegroundColor Green

# Optional: bundle ffmpeg if present on PATH or in packaging/bin
$Ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($Ffmpeg) {
    Copy-Item $Ffmpeg.Source (Join-Path $Dist "PolyFut\ffmpeg.exe") -Force
    Write-Host "Bundled ffmpeg.exe from PATH"
} elseif (Test-Path (Join-Path $Packaging "bin\ffmpeg.exe")) {
    Copy-Item (Join-Path $Packaging "bin\ffmpeg.exe") (Join-Path $Dist "PolyFut\ffmpeg.exe") -Force
    Write-Host "Bundled ffmpeg.exe from packaging/bin"
}

# ---------------------------------------------------------------------------
# Smoke test: RUN the thing we just built.
#
# Four bugs have reached a finished installer while every build step reported
# success: a stale icon, a stale index.html, an msvcp140.dll too old for torch
# (so 1.0.0 shipped unable to analyse anything), and a shared OpenVINO infer
# request that crashed a real run after nine minutes. Inspecting the output
# caught none of them. Running it catches three.
#
# So: launch the frozen exe and make it do real inference on several threads,
# before Inno Setup is allowed to wrap it. Set POLYFUT_SKIP_SMOKE=1 to bypass
# while iterating - but never for a build anyone else will install.
if ($env:POLYFUT_SKIP_SMOKE -eq "1") {
    Write-Host "Smoke test SKIPPED (POLYFUT_SKIP_SMOKE=1)" -ForegroundColor Yellow
} else {
    Write-Host "Smoke test: launching the built app..."
    $smokeProc = Start-Process $DistExe -PassThru
    $smokePort = $null
    try {
        foreach ($i in 1..60) {
            Start-Sleep -Seconds 2
            if ($smokeProc.HasExited) {
                throw "Smoke test failed: the app exited immediately (code $($smokeProc.ExitCode))"
            }
            $conn = Get-NetTCPConnection -OwningProcess $smokeProc.Id -State Listen -ErrorAction SilentlyContinue |
                    Select-Object -First 1
            if ($conn) { $smokePort = $conn.LocalPort; break }
        }
        if (-not $smokePort) { throw "Smoke test failed: the app never opened a port" }
        Write-Host "  app is up on port $smokePort, running self-test..."

        # Loads the real model and runs concurrent inferences. Slow on purpose:
        # first call compiles the OpenVINO graph.
        $res = Invoke-RestMethod "http://127.0.0.1:$smokePort/api/selftest" -Method Post -TimeoutSec 600
        if (-not $res.ok) { throw "Smoke test FAILED: $($res.error)" }
        Write-Host "  self-test passed in $($res.elapsed_sec)s" -ForegroundColor Green
        Write-Host "  ball model: $(Split-Path $res.ball_weights -Leaf)"
        if ($res.roi_weights) {
            Write-Host "  roi model : $(Split-Path $res.roi_weights -Leaf) @ $($res.roi_imgsz)"
        } else {
            throw ("Smoke test FAILED: no ROI model in the build. The 320 export " +
                   "is what keeps the warm path from costing more than a full scan.")
        }
        # A PyTorch fallback still "works" but is 5.4x slower - not shippable.
        if ($res.degraded) { throw "Smoke test FAILED (degraded): $($res.degraded)" }
    }
    finally {
        if ($smokeProc -and -not $smokeProc.HasExited) {
            Stop-Process -Id $smokeProc.Id -Force -ErrorAction SilentlyContinue
        }
        Get-Process -Name "PolyFut" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}

# Inno Setup lands in different places depending on how it was installed.
# `winget install JRSoftware.InnoSetup` without an admin prompt installs
# per-user into LOCALAPPDATA, which the two Program Files paths miss entirely -
# and the only symptom was "Inno Setup not found", i.e. a build that quietly
# produced no installer at all.
$Iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Iscc) {
    $OnPath = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($OnPath) { $Iscc = $OnPath.Source }
}

if ($Iscc) {
    Write-Host "Building installer with Inno Setup..."
    & $Iscc "/DDistDir=$Dist" (Join-Path $Packaging "polyfut_installer.iss")
    $Version = (Get-Content (Join-Path $Packaging "VERSION") -Raw).Trim()
    $Setup = Join-Path $Dist "PolyFut-Setup-$Version.exe"
    if (Test-Path $Setup) {
        Write-Host "Installer: $Setup" -ForegroundColor Green
        Copy-Item $Setup (Join-Path $Root "website\downloads\PolyFut-Setup-$Version.exe") -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Inno Setup not found - skip installer. Install from https://jrsoftware.org/isinfo.php" -ForegroundColor Yellow
    Write-Host "Portable app folder: $Dist\PolyFut\"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test: $Dist\PolyFut\PolyFut.exe"
Write-Host "  2. Upload website\downloads\PolyFut-Setup-*.exe to polyfut.com/downloads/"
Write-Host "  3. Deploy website/ to your static host (Vercel, Netlify, GitHub Pages)"
