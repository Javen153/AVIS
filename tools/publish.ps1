# Publish AVIS to dist\AVIS (unpackaged, self-contained, x64).
# Usage: tools\publish.ps1 [-Model v1] [-SkipBuild]
param(
    [string]$Model = 'v1',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

if (-not $SkipBuild) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root 'tools\build.ps1')
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$modelDir = Join-Path $root "models\$Model"
if (-not (Test-Path (Join-Path $modelDir 'model.onnx'))) { throw "model not found: $modelDir" }

$dist = Join-Path $root 'dist\AVIS'
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force -Path $dist | Out-Null

$log = Join-Path $root 'logs\publish.log'
& dotnet publish 'runtime\src\Avis.App\Avis.App.csproj' -c Release -p:Platform=x64 -o $dist *> $log
if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 15; throw "publish failed" }

Copy-Item -Recurse -Force (Join-Path $root 'shared') (Join-Path $dist 'shared')
New-Item -ItemType Directory -Force -Path (Join-Path $dist 'models') | Out-Null
Copy-Item -Recurse -Force $modelDir (Join-Path $dist "models\$Model")
# Copy user docs (ASCII names only: Windows PowerShell reads non-BOM scripts
# as ANSI, and the start-prompt .txt in the root must not be published).
foreach ($doc in 'README.md', 'USER_GUIDE.md', 'FINAL_REPORT.md') {
    $src = Join-Path $root $doc
    if (Test-Path $src) { Copy-Item -Force $src $dist }
}

$exe = Join-Path $dist 'Avis.App.exe'
if (-not (Test-Path $exe)) { throw "Avis.App.exe missing in $dist" }
$size = (Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("publish: PASS -> {0} ({1:N0} MB, model {2})" -f $exe, $size, $Model)
