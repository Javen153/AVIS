# Run scenario WAVs through the CLI analysis + check engine.
# Usage: tools\scenarios.ps1 [-Source dsp|neural|mock] [-Only S01,S02] [-Out artifacts\scenarios\dsp]
param(
    [string]$Source = 'dsp',
    [string]$Only = '',
    [string]$Out = '',
    [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'runtime\src\Avis.Cli\bin\x64\Release\net10.0\avis-cli.exe'
if (-not (Test-Path $exe)) { throw "avis-cli not built: $exe (run tools\build.ps1)" }
if (-not $Out) { $Out = "artifacts\scenarios\$Source" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$want = @()
if ($Only) { $want = $Only.Split(',') | ForEach-Object { $_.Trim().ToUpperInvariant() } }
$wavs = Get-ChildItem testdata\scenarios\S*.wav | Sort-Object Name
$rows = @()
$failed = 0
foreach ($wav in $wavs) {
    $id = $wav.BaseName
    if ($want.Count -gt 0 -and $want -notcontains $id) { continue }
    $jsonl = Join-Path $Out "$id.$Source.jsonl"
    $result = Join-Path $Out "$id.result.json"
    $args = @('analyze', '--in', $wav.FullName, '--source', $Source, '--out', $jsonl)
    if ($Model) { $args += @('--model', $Model) }
    & $exe @args | Out-Null
    if ($LASTEXITCODE -ne 0) { $rows += [pscustomobject]@{ Id = $id; Pass = $false; Detail = 'analyze failed' }; $failed++; continue }
    $checkArgs = @('check', '--frames', $jsonl, '--expect', (Join-Path $root "testdata\scenarios\$id.expect.json"), '--out', $result)
    $text = & $exe @checkArgs
    $code = $LASTEXITCODE
    $last = ($text | Select-Object -Last 1)
    $rows += [pscustomobject]@{ Id = $id; Pass = ($code -eq 0); Detail = $last }
    if ($code -ne 0) { $failed++ }
}
$rows | Format-Table -AutoSize
"summary: {0}/{1} passed, out {2}" -f ($rows.Count - $failed), $rows.Count, $Out
if ($failed -gt 0) { exit 1 }
