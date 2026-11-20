# Build a downloadable AVIS release: publish, bundle every exported model,
# add install scripts and zip it under artifacts\release.
# Usage: tools\release.ps1 [-Version 1.0.0] [-Model v1] [-SkipBuild] [-SkipZip]
param(
    [string]$Version = '1.0.0',
    [string]$Model = 'v1',
    [switch]$SkipBuild,
    [switch]$SkipZip
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

$pubArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'tools\publish.ps1'), '-Model', $Model)
if ($SkipBuild) { $pubArgs += '-SkipBuild' }
& powershell @pubArgs
if ($LASTEXITCODE -ne 0) { throw "publish failed" }

$dist = Join-Path $root 'dist\AVIS'

# Bundle every exported model so the settings drop-down works out of the box.
$models = Get-ChildItem (Join-Path $root 'models') -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'model.onnx') }
foreach ($m in $models) {
    $dst = Join-Path $dist "models\$($m.Name)"
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse -Force $m.FullName $dst
}

# Installer + user docs.
Copy-Item -Force (Join-Path $root 'packaging\install.ps1') $dist
Copy-Item -Force (Join-Path $root 'packaging\README-INSTALL.txt') $dist
foreach ($doc in 'USER_GUIDE.md', 'TROUBLESHOOTING.md') {
    $src = Join-Path $root $doc
    if (Test-Path $src) { Copy-Item -Force $src $dist }
}
$commit = 'unknown'
try { $commit = (git rev-parse --short HEAD).Trim() } catch { }
@(
    "AVIS $Version",
    "commit $commit",
    "built $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
    "models $((Get-ChildItem (Join-Path $dist 'models') -Directory | ForEach-Object Name) -join ', ')"
) | Set-Content -Encoding UTF8 (Join-Path $dist 'VERSION.txt')

$exe = Join-Path $dist 'Avis.App.exe'
if (-not (Test-Path $exe)) { throw "Avis.App.exe missing in $dist" }

$outDir = Join-Path $root 'artifacts\release'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$zip = Join-Path $outDir "AVIS-$Version-win-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
if (-not $SkipZip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $sw = [Diagnostics.Stopwatch]::StartNew()
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $dist, $zip, [IO.Compression.CompressionLevel]::Fastest, $false)
    $sw.Stop()
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash
    $mb = (Get-Item $zip).Length / 1MB
    $hash | Set-Content -Encoding ASCII ($zip + '.sha256')
    @{
        version = $Version; commit = $commit; zip = (Split-Path -Leaf $zip)
        bytes = (Get-Item $zip).Length; sha256 = $hash
        models = @(Get-ChildItem (Join-Path $dist 'models') -Directory | ForEach-Object Name)
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $outDir 'release.json')
    Write-Host ("release: {0} ({1:N0} MB, sha256 {2}) in {3:N0} s" -f $zip, $mb, $hash.Substring(0, 12), $sw.Elapsed.TotalSeconds)
} else {
    Write-Host "release: dist ready at $dist (zip skipped)"
}
