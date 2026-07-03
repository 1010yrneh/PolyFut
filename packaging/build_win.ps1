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

Write-Host "Running PyInstaller..."
pyinstaller (Join-Path $Packaging "pyinstaller.spec") --noconfirm

$DistExe = Join-Path $Root "dist\PolyFut\PolyFut.exe"
if (-not (Test-Path $DistExe)) {
    throw "Build failed: $DistExe not found"
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

$Iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

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
    Write-Host "Inno Setup not found — skip installer. Install from https://jrsoftware.org/isinfo.php" -ForegroundColor Yellow
    Write-Host "Portable app folder: dist\PolyFut\"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test: dist\PolyFut\PolyFut.exe"
Write-Host "  2. Upload dist\PolyFut-Setup-*.exe to polyfut.com/downloads/"
Write-Host "  3. Deploy website/ to your static host (Vercel, Netlify, GitHub Pages)"
