# Install AVIS from this folder (portable package).
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Dir "D:\Apps\AVIS"
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
param(
    [string]$Dir = (Join-Path $env:LOCALAPPDATA 'Programs\AVIS'),
    [switch]$Uninstall,
    [switch]$NoShortcuts
)
$ErrorActionPreference = 'Stop'
$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$appName = 'AVIS'
$exeName = 'Avis.App.exe'

function New-Shortcut([string]$Path, [string]$Target, [string]$WorkDir) {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($Path)
    $sc.TargetPath = $Target
    $sc.WorkingDirectory = $WorkDir
    $sc.IconLocation = "$Target,0"
    $sc.Description = 'AVIS - neural audio visualizer'
    $sc.Save()
}

$desktop = [Environment]::GetFolderPath('Desktop')
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) $appName
$desktopLnk = Join-Path $desktop "$appName.lnk"
$startLnk = Join-Path $startMenu "$appName.lnk"

if ($Uninstall) {
    Get-Process 'Avis.App' -ErrorAction SilentlyContinue | Stop-Process -Force
    Remove-Item -Force $desktopLnk -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $startMenu -ErrorAction SilentlyContinue
    if (Test-Path $Dir) {
        if ((Resolve-Path $Dir).Path -eq (Resolve-Path $src).Path) {
            Write-Host "Package folder is the install folder; delete it manually: $Dir"
        } else {
            Remove-Item -Recurse -Force $Dir
        }
    }
    Write-Host "AVIS uninstalled. User settings in $env:LOCALAPPDATA\AVIS were kept."
    exit 0
}

if (-not (Test-Path (Join-Path $src $exeName))) {
    throw "$exeName not found next to install.ps1. Unpack the whole zip first."
}

Get-Process 'Avis.App' -ErrorAction SilentlyContinue | Stop-Process -Force
if ((Resolve-Path $src).Path -ne (Resolve-Path (New-Item -ItemType Directory -Force -Path $Dir)).Path) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $src '*') $Dir
}

$exe = Join-Path $Dir $exeName
if (-not (Test-Path $exe)) { throw "install failed: $exe missing" }

if (-not $NoShortcuts) {
    New-Shortcut $desktopLnk $exe $Dir
    New-Item -ItemType Directory -Force -Path $startMenu | Out-Null
    New-Shortcut $startLnk $exe $Dir
}

Write-Host "AVIS installed: $Dir"
Write-Host "Shortcuts: Desktop and Start Menu (AVIS)"
Write-Host "Run: $exe"
