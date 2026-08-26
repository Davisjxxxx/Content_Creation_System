$ErrorActionPreference = 'Stop'

$exe = Resolve-Path 'dist\AvatarV2\AvatarV2.exe'
$icon = Resolve-Path 'build\avatar_v2.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Avatar V2.lnk'

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exe.Path
$shortcut.WorkingDirectory = Split-Path $exe.Path
$shortcut.IconLocation = $icon.Path
$shortcut.Description = 'Avatar V2 H3 Desktop Studio'
$shortcut.Save()

Write-Host "Desktop shortcut created: $shortcutPath" -ForegroundColor Magenta
