# Acceptance table for one model version, computed from the eval artifacts.
#
#   tools\acceptance.ps1 -Name v3_1                 # uses artifacts\eval files
#   tools\acceptance.ps1 -Name v3 -Clips 20
#
# Reads only files written by tools\eval_version.ps1:
#   artifacts\eval\scenarios_<Name>\*.result.json
#   artifacts\eval\real_<Name>_c<Clips>.json
#   artifacts\eval\tracking_<Name>.json
#   artifacts\eval\<Name>_summary.md  (parity / latency lines)
# Baseline numbers are the fixed ones from docs\REQUIREMENTS_V3.md (v1 / v2.1);
# nothing here changes a threshold, it only reports PASS/FAIL against them.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Clips = 20
)
$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = 'en-US'
$root = Split-Path -Parent $PSScriptRoot
$eval = Join-Path $root 'artifacts\eval'

$baseScen = 10            # >=10/13 and at least one of S04/S09/S13
$baseMusdb = 0.1298       # v1, same 6 clips
$baseSlakh = 0.3996       # v1, same 6 clips
$baseIdSw = 0.343         # v2.1, per second
$baseReid = 0.955         # v2.1
$baseLatency = 1.5        # ms, p50
$baseParity = 1e-3

$rows = [System.Collections.Generic.List[object]]::new()
function Add-Row([string]$item, [string]$value, [string]$limit, [bool]$ok) {
    $rows.Add([pscustomobject]@{ Criterion = $item; Value = $value; Limit = $limit
                                 Result = $(if ($ok) { 'PASS' } else { 'FAIL' }) })
}

# ---------------------------------------------------------------- scenarios
$sdir = Join-Path $eval "scenarios_$Name"
$passed = @()
$failed = @()
if (Test-Path $sdir) {
    Get-ChildItem $sdir -Filter *.result.json | Sort-Object Name | ForEach-Object {
        $j = Get-Content $_.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        if ($j.Pass) { $passed += $j.Scenario } else { $failed += $j.Scenario }
    }
}
$nPass = $passed.Count
$hard = @($passed | Where-Object { $_ -in 'S04', 'S09', 'S13' })
Add-Row 'scenarios' ("$nPass/13 (pass: " + ($passed -join ',') + ')') ">=$baseScen/13" ($nPass -ge $baseScen)
Add-Row 'one of S04/S09/S13' ($(if ($hard) { $hard -join ',' } else { 'none' })) 'at least one' ([bool]$hard)

# ---------------------------------------------------------------- real clips
$real = Join-Path $eval "real_$Name`_c$Clips.json"
if (Test-Path $real) {
    $j = Get-Content $real -Raw -Encoding utf8 | ConvertFrom-Json
    foreach ($d in 'musdb', 'slakh') {
        $all = @($j.details | Where-Object { $_.dataset -eq $d })
        $apAll = ($all | Measure-Object pitch_ap -Average).Average
        $ap6 = (@($all | Select-Object -First 6) | Measure-Object pitch_ap -Average).Average
        $base = if ($d -eq 'musdb') { $baseMusdb } else { $baseSlakh }
        Add-Row "$d AP ($Clips clips)" ('{0:N4}' -f $apAll) 'informational' $true
        Add-Row "$d AP (same 6 clips)" ('{0:N4}' -f $ap6) ">= $base" ($ap6 -ge $base)
    }
} else {
    Add-Row "real clips ($Clips)" 'missing' 'run eval_version.ps1' $false
}

# ---------------------------------------------------------------- tracking
$track = Join-Path $eval "tracking_$Name.json"
if (Test-Path $track) {
    $tj = Get-Content $track -Raw -Encoding utf8 | ConvertFrom-Json
    $mean = $tj.mean
    if ($mean) {
        Add-Row 'tracking id_switches/s' ('{0:N3}' -f $mean.id_switches_per_s) "<= $baseIdSw" ($mean.id_switches_per_s -le $baseIdSw)
        Add-Row 'tracking reid_acc' ('{0:N3}' -f $mean.reid_acc) ">= $baseReid" ($mean.reid_acc -ge $baseReid)
        Add-Row 'tracking count_in_range' ('{0:N3}' -f $mean.count_in_range) 'informational' $true
    }
} else {
    Add-Row 'tracking' 'missing' 'run eval_version.ps1' $false
}

# ---------------------------------------------------------------- summary
$sum = Join-Path $eval "$Name`_summary.md"
if (Test-Path $sum) {
    $txt = Get-Content $sum -Raw -Encoding utf8
    $par = [regex]::Match($txt, 'model max\|diff\|=([0-9.,Ee+\-]+)')
    if ($par.Success) {
        $raw = $par.Groups[1].Value.Replace(',', '.').TrimEnd('.')
        $v = [double]::Parse($raw, [Globalization.CultureInfo]::InvariantCulture)
        Add-Row 'ONNX parity (C#)' ('{0:E2}' -f $v) "<= $baseParity" ($v -le $baseParity)
    }
    $lat = [regex]::Match($txt, 'infer p50 ([0-9.,]+)')
    if ($lat.Success) {
        $v = [double]::Parse($lat.Groups[1].Value.Replace(',', '.'), [Globalization.CultureInfo]::InvariantCulture)
        Add-Row 'latency p50' ('{0:N2} ms' -f $v) "<= $baseLatency ms" ($v -le $baseLatency)
    }
}

# ---------------------------------------------------------------- verdict
"== acceptance: $Name ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) =="
$rows | Format-Table -AutoSize
$mandatory = @($rows | Where-Object { $_.Limit -ne 'informational' })
$bad = @($mandatory | Where-Object { $_.Result -eq 'FAIL' })
"mandatory: $($mandatory.Count - $bad.Count)/$($mandatory.Count) PASS"
if ($bad.Count -eq 0) {
    "VERDICT: ACCEPTED"
} else {
    'VERDICT: NOT ACCEPTED - ' + (($bad | ForEach-Object { $_.Criterion }) -join '; ')
}
