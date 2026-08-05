; Inno Setup script - build after PyInstaller (packaging/build_win.ps1).
; Requires Inno Setup 6: https://jrsoftware.org/isinfo.php

#define MyAppName "PolyFut"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "PolyFut"
#define MyAppURL "https://polyfut.com"
#define MyAppExeName "PolyFut.exe"
; Build output lives outside the repo (see build_win.ps1); the
; build passes /DDistDir. Default keeps a standalone compile working.
#ifndef DistDir
  #define DistDir "..\dist"
#endif
#define MyAppAssocName MyAppName
#define MyAppAssocExt ".polyfut"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
AppId={{A7B3C9D1-4E2F-5A6B-8C9D-0E1F2A3B4C5D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=
OutputDir={#DistDir}
OutputBaseFilename=PolyFut-Setup-{#MyAppVersion}
SetupIconFile=icons\polyfut.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#DistDir}\PolyFut\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; There is deliberately no [Code] / InitializeSetup check here.
;
; It used to test FileExists({#SourcePath}\..\dist\PolyFut\PolyFut.exe) and
; abort with "Build PolyFut.exe first: run packaging\build_win.ps1". That reads
; like a build guard, but {#SourcePath} is a PREPROCESSOR variable: it is
; substituted when ISCC compiles, so the shipped installer carried an absolute
; path from the build machine and tested it on the END USER's computer, where it
; cannot exist. Every download would have aborted, telling the user to run a
; PowerShell script. It only ever passed here because the path happened to exist
; on the build machine -- and it started failing the moment the build output
; moved out of the repo, which is how it was found.
;
; The guard it was trying to be already exists and is free: ISCC fails at
; COMPILE time if the [Files] Source pattern matches nothing, so an installer
; cannot be produced without the app in the first place.
