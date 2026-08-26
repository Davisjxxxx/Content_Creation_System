$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.venv')) {
  py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[desktop,dev]'
& .\.venv\Scripts\python.exe scripts\make_icon.py

if (Test-Path 'dist\AvatarV2') { Remove-Item -Recurse -Force 'dist\AvatarV2' }
if (Test-Path 'build\pyinstaller') { Remove-Item -Recurse -Force 'build\pyinstaller' }

& .\.venv\Scripts\pyinstaller.exe `
  --noconfirm `
  --clean `
  --windowed `
  --name AvatarV2 `
  --icon build\avatar_v2.ico `
  --add-data 'avatar_v2\ui;avatar_v2\ui' `
  --collect-submodules webview `
  --distpath dist `
  --workpath build\pyinstaller `
  scripts\desktop_entry.py

Write-Host ''
Write-Host 'Avatar V2 build complete:' -ForegroundColor Magenta
Write-Host (Resolve-Path 'dist\AvatarV2\AvatarV2.exe')
