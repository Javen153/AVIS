# Launch AVIS: dist\AVIS when published, otherwise the build output.
# Usage: tools\run.ps1 [app arguments...]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArgs)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root 'dist\AVIS\Avis.App.exe'
$dev = Join-Path $root 'runtime\src\Avis.App\bin\x64\Release\net10.0-windows10.0.19041.0\win-x64\Avis.App.exe'
$exe = if (Test-Path $dist) { $dist } elseif (Test-Path $dev) { $dev } else { throw 'Avis.App.exe not built (run tools\build.ps1)' }
Write-Host ("run: {0}" -f $exe)
& $exe @AppArgs
exit $LASTEXITCODE
