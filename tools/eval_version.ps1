# Full evaluation pipeline for one model version: export -> parity -> scenarios ->
# real clips -> tracking -> latency bench. Written for M-v3 (P15) but version-agnostic.
#
# Usage:
#   tools\eval_version.ps1 -Name v3                       # full pipeline
#   tools\eval_version.ps1 -Name v3 -SkipExport           # model dir already exists
#   tools\eval_version.ps1 -Name v3 -Clips 6 -SkipTracking
#
# Note: export.py writes C# golden model IO into a per-version tree under
# artifacts\golden\<Name>\model, so shared\golden\model (v1, the committed default)
# stays untouched. Pass -GoldenDir to point at an existing tree instead.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Ckpt = '',
    [string]$ModelDir = '',
    [string]$GoldenDir = '',
    [int]$Clips = 20,
    [switch]$SkipExport,
    [switch]$SkipScenarios,
    [switch]$SkipReal,
    [switch]$SkipTracking,
    [switch]$SkipBench
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Ckpt) { $Ckpt = "models\$Name\ckpt_best.pt" }
if (-not $ModelDir) { $ModelDir = Join-Path $root "models\$Name" }
if (-not (Test-Path $ModelDir) -and -not $SkipExport) { New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null }
$ModelDir = (Resolve-Path $ModelDir).Path
if (-not $GoldenDir) { $GoldenDir = Join-Path $root "artifacts\golden\$Name" }
$GoldenDir = [System.IO.Path]::GetFullPath($GoldenDir)
$eval = Join-Path $root 'artifacts\eval'
New-Item -ItemType Directory -Force -Path $eval | Out-Null
$exe = Join-Path $root 'runtime\src\Avis.Cli\bin\x64\Release\net10.0\avis-cli.exe'
$summary = Join-Path $eval "$Name`_summary.md"
$lines = [System.Collections.Generic.List[string]]::new()
$failed = 0

function Step([string]$title, [scriptblock]$body) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
    $script:lines.Add("")
    $script:lines.Add("## $title")
    # native stderr must not abort the pipeline, so relax Stop for the duration of the step
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = @(& $body 2>&1)
        $code = $LASTEXITCODE
        $text = ($out | Out-String).Trim()
        if ($text) { Write-Host $text }
        $script:lines.Add('```')
        $script:lines.Add($text)
        $script:lines.Add('```')
        if ($code -ne 0) {
            $script:failed++
            $script:lines.Add("**exit code: $code**")
        }
        return $code
    }
    catch {
        $script:failed++
        Write-Host $_.Exception.Message -ForegroundColor Red
        $script:lines.Add("**ERROR: $($_.Exception.Message)**")
        return 1
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

$lines.Add("# Evaluation: $Name")
$lines.Add("")
$lines.Add("- date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("- ckpt: ``$Ckpt``")
$lines.Add("- model dir: ``$ModelDir``")
$lines.Add("- golden tree: ``$GoldenDir``")

# golden tree = frontend golden files + a per-version model/ subdir, so parity can run
# for any version without touching shared\golden\model.
New-Item -ItemType Directory -Force -Path $GoldenDir | Out-Null
foreach ($f in 'frontend_input.f32', 'frontend_expected.f32', 'frontend_meta.json') {
    $src = Join-Path $root "shared\golden\$f"
    if ((Test-Path $src) -and -not (Test-Path (Join-Path $GoldenDir $f))) { Copy-Item $src $GoldenDir }
}

if (-not $SkipExport) {
    Step 'export' {
        Push-Location ml
        try {
            & uv run python -m avis_ml.export --ckpt $Ckpt --out $ModelDir --device cpu `
                --golden-out (Join-Path $GoldenDir 'model')
        } finally { Pop-Location }
    } | Out-Null
}

if (-not (Test-Path (Join-Path $ModelDir 'model.onnx'))) { throw "model.onnx missing in $ModelDir (run export first)" }
if (-not (Test-Path $exe)) { throw "avis-cli not built: $exe (run tools\build.ps1)" }

Step 'onnx parity (C# vs golden)' { & $exe parity --golden $GoldenDir --model $ModelDir } | Out-Null

if (-not $SkipScenarios) {
    $outDir = Join-Path $eval "scenarios_$Name"
    Step 'scenarios S01-S13' {
        & (Join-Path $root 'tools\scenarios.ps1') -Source neural -Model $ModelDir -Out $outDir
    } | Out-Null
}

if (-not $SkipReal) {
    Step "real clips ($Clips)" {
        Push-Location ml
        try {
            & uv run python -m avis_ml.eval_real --ckpt $Ckpt --clips $Clips --device cpu `
                --out (Join-Path $eval "real_$Name`_c$Clips.json")
        } finally { Pop-Location }
    } | Out-Null
}

if (-not $SkipTracking) {
    Step 'tracking' {
        Push-Location ml
        try {
            & uv run python -m avis_ml.eval_tracking --model $ModelDir --out (Join-Path $eval "tracking_$Name.json")
        } finally { Pop-Location }
    } | Out-Null
}

if (-not $SkipBench) {
    Step 'bench model latency' { & $exe bench --what model --model $ModelDir } | Out-Null
}

$lines.Add('')
$lines.Add("result: $($(if ($failed -eq 0) { 'PASS' } else { "FAIL ($failed step(s))" }))")
Set-Content -Path $summary -Value $lines -Encoding utf8
Write-Host ""
Write-Host "summary: $summary" -ForegroundColor Cyan
if ($failed -gt 0) { exit 1 }
