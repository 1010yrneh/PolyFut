# NOTE: keep this file ASCII-only and BOM-prefixed - see build_win.ps1.
# Sync packaging/VERSION into Inno Setup script and website metadata.
$ErrorActionPreference = "Stop"
$Packaging = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Packaging
$Version = (Get-Content (Join-Path $Packaging "VERSION") -Raw).Trim()

# Only the #define is rewritten. OutputBaseFilename already reads
# {#MyAppVersion}, so substituting a literal version there just destroyed the
# indirection and made the file churn on every build.
$Iss = Join-Path $Packaging "polyfut_installer.iss"
$content = Get-Content $Iss -Raw
$content = $content -replace '(?m)^#define MyAppVersion ".*"', "#define MyAppVersion `"$Version`""
[System.IO.File]::WriteAllText($Iss, $content, (New-Object System.Text.UTF8Encoding $false))

# WriteAllText with UTF8Encoding($false), not Set-Content -Encoding UTF8: the
# latter emits a BOM, and this file is fetched over HTTP by the site, where a
# leading BOM breaks strict JSON parsers and is invisible in every editor.
$webVersion = Join-Path $Root "website\version.json"
# installer_url is what the download button actually uses. The installer is a
# GitHub Release asset, not a file on the site: Pages caps the whole site at
# 1GB and git rejects any file over 100MB, so a ~400MB installer can never be
# served from website/. Kept in step with $Version here so a release cannot
# silently keep offering the previous build.
$Repo = $env:POLYFUT_RELEASE_REPO
if (-not $Repo) { $Repo = "1010yrneh/Polyfut" }
$json = @{
    version = $Version
    windows_installer = "downloads/PolyFut-Setup-$Version.exe"
    installer_url = "https://github.com/$Repo/releases/download/v$Version/PolyFut-Setup-$Version.exe"
} | ConvertTo-Json
[System.IO.File]::WriteAllText($webVersion, $json, (New-Object System.Text.UTF8Encoding $false))

Write-Host "Version $Version synced to installer + website/version.json"
