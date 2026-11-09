# Create a desktop shortcut to the published AVIS app.
# Usage: tools\shortcut.ps1 [-Name AVIS]
param([string]$Name = 'AVIS')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'dist\AVIS\Avis.App.exe'
if (-not (Test-Path $exe)) { throw "publish first, exe missing: $exe" }
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop ($Name + '.lnk')
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $exe
$sc.WorkingDirectory = Split-Path $exe
$sc.IconLocation = "$exe,0"
$sc.Description = 'AVIS - neural audio visualizer'
$sc.Save()
Write-Host ("shortcut: {0}" -f $lnk)
