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
$Dist = Join-Path $Root "dist"
if (Test-Path $Dist) {
    Write-Host "Clearing dist..."
    foreach ($attempt in 1..5) {
        try {
            Remove-Item $Dist -Recurse -Force -ErrorAction Stop
            break
        } catch {
            if ($attempt -eq 5) {
                throw ("Could not clear $Dist after 5 attempts: " +
                       $_.Exception.Message +
                       " - close PolyFut.exe and pause OneDrive sync, then retry.")
            }
            Write-Host "  locked, retrying in 3s ($attempt/5)..."
            Start-Sleep -Seconds 3
        }
    }
}

$SpecPath = Join-Path $Packaging "pyinstaller.spec"
Write-Host "Running PyInstaller..."
pyinstaller $SpecPath --noconfirm

# A native command's exit code does NOT trip $ErrorActionPreference, so this
# has to be explicit. Without it a failed PyInstaller was invisible and Inno
# Setup packaged whatever happened to be in dist from a previous build.
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$DistExe = Join-Path $Root "dist\PolyFut\PolyFut.exe"
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
    Copy-Item $Ffmpeg.Source (Join-Path $Root "dist\PolyFut\ffmpeg.exe") -Force
    Write-Host "Bundled ffmpeg.exe from PATH"
} elseif (Test-Path (Join-Path $Packaging "bin\ffmpeg.exe")) {
    Copy-Item (Join-Path $Packaging "bin\ffmpeg.exe") (Join-Path $Root "dist\PolyFut\ffmpeg.exe") -Force
    Write-Host "Bundled ffmpeg.exe from packaging/bin"
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
    & $Iscc (Join-Path $Packaging "polyfut_installer.iss")
    $Version = (Get-Content (Join-Path $Packaging "VERSION") -Raw).Trim()
    $Setup = Join-Path $Root "dist\PolyFut-Setup-$Version.exe"
    if (Test-Path $Setup) {
        Write-Host "Installer: $Setup" -ForegroundColor Green
        Copy-Item $Setup (Join-Path $Root "website\downloads\PolyFut-Setup-$Version.exe") -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Inno Setup not found - skip installer. Install from https://jrsoftware.org/isinfo.php" -ForegroundColor Yellow
    Write-Host "Portable app folder: dist\PolyFut\"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test: dist\PolyFut\PolyFut.exe"
Write-Host "  2. Upload dist\PolyFut-Setup-*.exe to polyfut.com/downloads/"
Write-Host "  3. Deploy website/ to your static host (Vercel, Netlify, GitHub Pages)"
