# AVIS verification: build, unit tests, frontend parity, model, scenarios, render, app.
# Writes artifacts\verify\<stamp>\report.md and report.json. Exit 0 only when no FAIL.
# Usage: tools\verify.ps1 [-Only frontend,scenarios] [-Source auto|dsp|neural] [-Model v1] [-NoBuild]
param(
    [string]$Only = '',
    [string]$Source = 'auto',
    [string]$Model = 'v1',
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = Join-Path $root "artifacts\verify\$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$cli = Join-Path $root 'runtime\src\Avis.Cli\bin\x64\Release\net10.0\avis-cli.exe'
$app = Join-Path $root 'runtime\src\Avis.App\bin\x64\Release\net10.0-windows10.0.19041.0\win-x64\Avis.App.exe'
$modelDir = Join-Path $root "models\$Model"
$hasModel = Test-Path (Join-Path $modelDir 'model.onnx')

if ($Only) { $want = $Only.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() } }
else { $want = @('build', 'unit', 'frontend', 'model', 'scenarios', 'render', 'app') }
if ($want -contains 'ml') { $want += 'model' }

$results = New-Object System.Collections.Generic.List[object]
function Add-Result([string]$Name, [string]$Status, [string]$Detail, [double]$Ms) {
    $results.Add([pscustomobject]@{ name = $Name; status = $Status; detail = $Detail; ms = [math]::Round($Ms, 1) })
    $color = switch ($Status) { 'PASS' { 'Green' } 'FAIL' { 'Red' } default { 'Yellow' } }
    Write-Host ("{0,-12} {1} {2}" -f $Name, $Status, $Detail) -ForegroundColor $color
}

function Measure-Check([string]$Name, [scriptblock]$Body) {
    if ($want -notcontains $Name) { return }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $r = & $Body
        $sw.Stop()
        Add-Result $Name $r.Status $r.Detail $sw.Elapsed.TotalMilliseconds
    }
    catch {
        $sw.Stop()
        Add-Result $Name 'FAIL' $_.Exception.Message $sw.Elapsed.TotalMilliseconds
    }
}

# ------------------------------------------------------------------ build
$buildOk = $true
if (-not $NoBuild) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root 'tools\build.ps1') *> (Join-Path $outDir 'build.log')
    $buildOk = $LASTEXITCODE -eq 0
    $sw.Stop()
    if ($want -contains 'build') {
        Add-Result 'build' $(if ($buildOk) { 'PASS' } else { 'FAIL' }) $(if ($buildOk) { 'solution x64 Release' } else { 'see build.log' }) $sw.Elapsed.TotalMilliseconds
    }
    if (-not $buildOk) { Write-Host 'build failed, skipping remaining checks' -ForegroundColor Red }
}

Measure-Check 'unit' {
    $log = Join-Path $outDir 'unit.log'
    & dotnet test 'runtime\tests\Avis.Tests\Avis.Tests.csproj' -c Release -p:Platform=x64 --no-build *> $log
    $ok = $LASTEXITCODE -eq 0
    $line = (Get-Content $log -Encoding UTF8 | Select-String 'Passed!|Failed!|Пройден|не пройдено' | Select-Object -Last 1)
    @{ Status = $(if ($ok) { 'PASS' } else { 'FAIL' }); Detail = if ($line) { $line.Line.Trim() } else { 'see unit.log' } }
}

Measure-Check 'frontend' {
    if (-not (Test-Path $cli)) { return @{ Status = 'FAIL'; Detail = 'avis-cli.exe missing' } }
    $text = & $cli parity --golden (Join-Path $root 'shared\golden') 2>&1
    $ok = $LASTEXITCODE -eq 0
    $line = ($text | Select-String 'frontend' | Select-Object -First 1)
    @{ Status = $(if ($ok) { 'PASS' } else { 'FAIL' }); Detail = if ($line) { $line.Line.Trim() } else { ($text | Select-Object -Last 1) } }
}

Measure-Check 'model' {
    if (-not $hasModel) { return @{ Status = 'SKIP'; Detail = "models\$Model\model.onnx missing" } }
    $par = & $cli parity --golden (Join-Path $root 'shared\golden') --model $modelDir 2>&1
    $pOk = $LASTEXITCODE -eq 0
    $bench = & $cli bench --what model --model $modelDir 2>&1
    $bOk = $LASTEXITCODE -eq 0
    $pl = ($par | Select-String 'model max' | Select-Object -First 1)
    $bl = ($bench | Select-String 'model:' | Select-Object -First 1)
    @{ Status = $(if ($pOk -and $bOk) { 'PASS' } else { 'FAIL' });
       Detail = (@($(if ($pl) { $pl.Line.Trim() }), $(if ($bl) { $bl.Line.Trim() })) -join '; ') }
}

Measure-Check 'scenarios' {
    if ($Source -eq 'neural' -and -not $hasModel) { return @{ Status = 'SKIP'; Detail = "neural model missing" } }
    $sdir = Join-Path $outDir 'scenarios'
    $sources = @()
    if ($Source -eq 'auto' -or $Source -eq 'dsp') { $sources += 'dsp' }
    if (($Source -eq 'auto' -and $hasModel) -or $Source -eq 'neural') { $sources += 'neural' }
    if ($sources.Count -eq 0) { return @{ Status = 'SKIP'; Detail = 'no source selected' } }
    $details = @(); $allOk = $true
    foreach ($src in $sources) {
        $od = Join-Path $sdir $src
        $sargs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'tools\scenarios.ps1'), '-Source', $src, '-Out', $od)
        if ($src -eq 'neural') { $sargs += @('-Model', $modelDir) }
        & powershell @sargs *> (Join-Path $outDir "scenarios-$src.log")
        $files = Get-ChildItem (Join-Path $od 'S*.result.json') -ErrorAction SilentlyContinue
        $checks = @($files | ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json })
        $passed = @($checks | Where-Object { $_.Pass })
        $count = @($files).Count
        if ($src -eq 'dsp') {
            $required = @('S01', 'S02', 'S03', 'S04', 'S05', 'S07', 'S08')
            $ok = $true
            foreach ($id in $required) {
                if ($passed.Scenario -notcontains $id) { $ok = $false }
            }
            if (-not $ok) { $allOk = $false }
            $details += "dsp $($passed.Count)/$count (required 7/7)"
        }
        else {
            if ($passed.Count -lt $count) { $allOk = $false }
            $details += "$src $($passed.Count)/$count"
        }
    }
    @{ Status = $(if ($allOk) { 'PASS' } else { 'FAIL' }); Detail = ($details -join ', ') }
}

Measure-Check 'render' {
    $rd = Join-Path $outDir 'render'
    New-Item -ItemType Directory -Force -Path $rd | Out-Null
    $frames = Join-Path $rd 's01.jsonl'
    & $cli analyze --in (Join-Path $root 'testdata\scenarios\S01.wav') --source dsp --out $frames | Out-Null
    & $cli render --frames $frames --out $rd --view main --every 100 --w 480 --h 180 | Out-Null
    $png = Get-ChildItem (Join-Path $rd 'frame_*.png') | Sort-Object Name | Select-Object -Last 1
    if (-not $png) { return @{ Status = 'FAIL'; Detail = 'no PNG produced' } }
    Add-Type -AssemblyName System.Drawing
    $bmp = [System.Drawing.Bitmap]::FromFile($png.FullName)
    $lit = 0
    try {
        for ($y = 0; $y -lt $bmp.Height; $y += 2) {
            for ($x = 0; $x -lt $bmp.Width; $x += 2) {
                $c = $bmp.GetPixel($x, $y)
                if ($c.A -gt 8 -and ($c.R + $c.G + $c.B) -gt 24) { $lit++ }
            }
        }
    }
    finally { $bmp.Dispose() }
    $bench = & $cli bench --what render --frames 60 2>&1
    $ok = $LASTEXITCODE -eq 0
    $bl = ($bench | Select-String 'render submit' | Select-Object -First 1)
    if ($lit -lt 10) { return @{ Status = 'FAIL'; Detail = "offscreen PNG has $lit lit samples" } }
    @{ Status = $(if ($ok) { 'PASS' } else { 'FAIL' }); Detail = "png $($png.Name) lit=$lit; " + $(if ($bl) { $bl.Line.Trim() } else { '' }) }
}

Measure-Check 'app' {
    if (-not (Test-Path $app)) { return @{ Status = 'FAIL'; Detail = 'Avis.App.exe missing' } }
    $bd = Join-Path $outDir 'app'
    New-Item -ItemType Directory -Force -Path $bd | Out-Null
    $shot = Join-Path $bd 'mock.png'
    $benchFile = Join-Path $bd 'bench.json'
    Remove-Item $shot, $benchFile -ErrorAction SilentlyContinue
    $argLine = "--source mock --bench 3 --bench-out `"$benchFile`" --screenshot `"$shot`" --screenshot-at 2 --autoclose"
    $proc = Start-Process -FilePath $app -ArgumentList $argLine -PassThru -Wait
    $exit = $proc.ExitCode
    if ($exit -ne 0) { return @{ Status = 'FAIL'; Detail = "exit $exit, see app\run.log" } }
    if (-not (Test-Path $benchFile)) { return @{ Status = 'FAIL'; Detail = 'no bench.json' } }
    $b = Get-Content $benchFile -Raw | ConvertFrom-Json
    Add-Type -AssemblyName System.Drawing
    $lit = 0
    if (Test-Path $shot) {
        $bmp = [System.Drawing.Bitmap]::FromFile($shot)
        try {
            for ($y = 0; $y -lt $bmp.Height; $y += 4) {
                for ($x = 0; $x -lt $bmp.Width; $x += 4) {
                    $c = $bmp.GetPixel($x, $y)
                    if ($c.A -gt 8 -and ($c.R + $c.G + $c.B) -gt 24) { $lit++ }
                }
            }
        }
        finally { $bmp.Dispose() }
    }
    $fps = [math]::Round($b.fps_p50, 1)
    $an = [math]::Round($b.analysis_ms_p99, 2)
    $ok = $lit -ge 10
    @{ Status = $(if ($ok) { 'PASS' } else { 'FAIL' });
       Detail = "fps p50 $fps, analysis p99 $an ms, objects $([math]::Round($b.objects_avg,1)), lit=$lit, device=$($b.device)" }
}

# ----------------------------------------------------------------- report
$fail = @($results | Where-Object { $_.status -eq 'FAIL' }).Count
$pass = @($results | Where-Object { $_.status -eq 'PASS' }).Count
$skip = @($results | Where-Object { $_.status -eq 'SKIP' }).Count

$json = [pscustomobject]@{ date = $stamp; source = $Source; model = $Model; pass = $pass; fail = $fail; skip = $skip; checks = $results }
$json | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $outDir 'report.json')

$md = New-Object System.Collections.Generic.List[string]
$md.Add('# Отчёт проверки AVIS')
$md.Add('')
$md.Add("Дата: $stamp · источник: $Source · модель: $Model")
$md.Add('')
$md.Add('| Проверка | Статус | Детали | Время, мс |')
$md.Add('|---|---|---|---|')
foreach ($r in $results) { $md.Add("| $($r.name) | $($r.status) | $($r.detail) | $($r.ms) |") }
$md.Add('')
$md.Add("Итого: PASS $pass, FAIL $fail, SKIP $skip.")
$md | Set-Content -Encoding UTF8 (Join-Path $outDir 'report.md')

Write-Host ''
Write-Host ("verify: PASS {0}, FAIL {1}, SKIP {2} -> {3}" -f $pass, $fail, $skip, $outDir)
exit $(if ($fail -gt 0) { 1 } else { 0 })
