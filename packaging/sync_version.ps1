# Sync packaging/VERSION into Inno Setup script and website metadata.
$ErrorActionPreference = "Stop"
$Packaging = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Packaging
$Version = (Get-Content (Join-Path $Packaging "VERSION") -Raw).Trim()
$Iss = Join-Path $Packaging "polyfut_installer.iss"
$content = Get-Content $Iss -Raw
$content = $content -replace '(?m)^#define MyAppVersion ".*"', "#define MyAppVersion `"$Version`""
$content = $content -replace '(?m)^OutputBaseFilename=PolyFut-Setup-.*', "OutputBaseFilename=PolyFut-Setup-$Version"
Set-Content -Path $Iss -Value $content -NoNewline

$webVersion = Join-Path $Root "website\version.json"
@{
    version = $Version
    windows_installer = "downloads/PolyFut-Setup-$Version.exe"
} | ConvertTo-Json | Set-Content -Path $webVersion -Encoding UTF8
Write-Host "Version $Version synced to installer + website/version.json"
