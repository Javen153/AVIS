$ErrorActionPreference = 'Stop'
# AVIS environment check (P0). Writes a short report to logs/check_env.log
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'logs\check_env.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$lines = New-Object System.Collections.Generic.List[string]
function Say($s) { $lines.Add($s); Write-Host $s }

$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')

Say ("OS: " + [Environment]::OSVersion.VersionString)
Say ("Arch: " + $env:PROCESSOR_ARCHITECTURE)
$gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
if ($LASTEXITCODE -eq 0) { Say ("GPU: " + ($gpu -join ' | ')) } else { Say "GPU: nvidia-smi not available" }

Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    if ($_.Free -gt 0) { Say ("Disk {0}: free {1:N1} GB" -f $_.Name, ($_.Free/1GB)) }
}

foreach ($c in 'git','dotnet','uv','curl','tar') {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { Say ("{0}: {1}" -f $c, $cmd.Source) } else { Say ("{0}: MISSING" -f $c) }
}
if (Get-Command dotnet -ErrorAction SilentlyContinue) {
    Say ("dotnet SDKs: " + ((& dotnet --list-sdks) -join '; '))
}
$avisData = [Environment]::GetEnvironmentVariable('AVIS_DATA','User')
if (-not $avisData) { $avisData = 'E:\avis_data' }
$driveLetter = ([IO.Path]::GetPathRoot($avisData)).Substring(0,1)
$freeGb = (Get-PSDrive $driveLetter).Free / 1GB
Say ("AVIS_DATA: " + $avisData + " (free " + ("{0:N1}" -f $freeGb) + " GB)")

$lines | Set-Content -Encoding UTF8 $log
Write-Host "Report: $log"
