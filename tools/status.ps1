# Live status of background jobs started with tools\bg.ps1.
#
#   tools\status.ps1                  # one-shot table of every job
#   tools\status.ps1 -Name v3         # details for one job (phases, last ckpts)
#   tools\status.ps1 -Watch 10        # refresh every 10 s in place (Ctrl+C to stop)
#
# Reads only logs\bg\*.log and *.pid; never touches the jobs themselves.
param(
    [string]$Name = '',
    [int]$Watch = 0,
    [int]$Tail = 5,
    [switch]$All,
    # keep refreshing until the job process exits, then print the final state
    [switch]$UntilDone,
    # keep the window open on exit (used by bg.ps1 so the result is readable)
    [switch]$Pause
)
$ErrorActionPreference = 'Stop'
# fixed formatting so the table does not switch to '0,00' on ru-RU hosts
[System.Threading.Thread]::CurrentThread.CurrentCulture = 'en-US'
$root = Split-Path -Parent $PSScriptRoot
$bgDir = Join-Path $root 'logs\bg'

function Read-Parts([string]$path) {
    $head = @(Get-Content -Path $path -Encoding utf8 -TotalCount 60 -ErrorAction SilentlyContinue)
    $tail = @(Get-Content -Path $path -Encoding utf8 -Tail 500 -ErrorAction SilentlyContinue)
    return @($head, $tail)
}

function Parse-Job([string]$log) {
    $item = Get-Item $log
    $parts = Read-Parts $log
    $head = $parts[0]; $tail = $parts[1]
    $j = [pscustomobject]@{
        Log = $log; Step = -1; Total = 0; Rate = 0.0; Start = $item.CreationTime
        Updated = $item.LastWriteTime; Done = $false; Ckpt = ''; Mix = ''
        Warn = 0; Recover = 0; Skip = 0; Last = ''
    }
    foreach ($l in $head) {
        if ($l -match 'steps=(\d+)') { $j.Total = [int]$matches[1] }
    }
    foreach ($l in $tail) {
        if ($l -match '^step (\d+) ') { $j.Step = [int]$matches[1] }
        if ($l -match 'steps/s') {
            if ($l -match '([\d.]+) steps/s') { $j.Rate = [double]$matches[1] }
        }
        if ($l -match '\[ckpt\]') { $j.Ckpt = $l.Trim() }
        if ($l -match '\[mix\]') { $j.Mix = $l.Trim() }
        if ($l -match '\[warn\]') { $j.Warn++ }
        if ($l -match '\[recover\]') { $j.Recover++ }
        if ($l -match '\[skip\]|nonfinite') { $j.Skip++ }
        if ($l -match '^done in') { $j.Done = $true; $j.Last = $l.Trim() }
    }
    if (-not $j.Done -and $tail.Count -gt 0) { $j.Last = $tail[-1].Trim() }
    if ($j.Rate -le 0 -and $j.Step -gt 0) {
        $sec = ([datetime]::Now - $j.Start).TotalSeconds
        if ($sec -gt 30) { $j.Rate = $j.Step / $sec }
    }
    return $j
}

function Job-Table {
    $rows = @()
    Get-ChildItem -Path $bgDir -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object {
        $j = Parse-Job $_.FullName
        $pidFile = [IO.Path]::ChangeExtension($j.Log, '.pid')
        $alive = $false
        if (Test-Path $pidFile) {
            $procId = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
            $alive = [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
        }
        $state = if ($alive) { 'RUN' } elseif ($j.Done) { 'done' } else { 'stopped' }
        $steps = $j.Step + 1
        $prog = ''
        $eta = ''
        $finish = ''
        if ($j.Total -gt 0) {
            $frac = [Math]::Min(1.0, $steps / [double]$j.Total)
            $n = [int][Math]::Round($frac * 16)
            $bar = ('#' * $n) + ('-' * (16 - $n))
            $prog = '[' + $bar + '] {0,3}%' -f [int]($frac * 100)
            if ($alive -and $j.Rate -gt 0) {
                $left = ($j.Total - $steps) / $j.Rate / 60.0
                $eta = '{0:N0}min' -f $left
                $finish = (Get-Date).AddMinutes($left).ToString('HH:mm')
            }
        }
        $ck = ''
        if ($j.Ckpt -match 'step (\d+) composite ([\d.]+) best ([\d.]+)') {
            $ck = '{0} (best {1})' -f $matches[2], $matches[3]
        }
        $rows += [pscustomobject]@{
            Job = [IO.Path]::GetFileNameWithoutExtension($j.Log)
            State = $state
            Step = $(if ($j.Total -gt 0) { "$steps/$($j.Total)" } elseif ($steps -gt 0) { "$steps" } else { '' })
            Progress = $prog; Rate = $(if ($j.Rate) { '{0:N2}/s' -f $j.Rate } else { '' })
            ETA = $eta; Finish = $finish; LastCkpt = $ck
            'warn/rec/skip' = $(if ($j.Warn + $j.Recover + $j.Skip) { '{0}/{1}/{2}' -f $j.Warn, $j.Recover, $j.Skip } else { '' })
            Updated = $j.Updated.ToString('HH:mm:ss')
        }
    }
    $sorted = $rows | Sort-Object @{ e = { $_.State -ne 'RUN' } }, @{ e = { $_.Updated }; Descending = $true }
    if (-not $All) {
        $run = @($sorted | Where-Object { $_.State -eq 'RUN' })
        $rest = @($sorted | Where-Object { $_.State -ne 'RUN' } | Select-Object -First 5)
        $sorted = @($run) + @($rest)
    }
    $sorted | Format-Table -AutoSize -Property Job, State, Step, Progress, Rate, ETA, Finish, LastCkpt, 'warn/rec/skip', Updated |
        Out-String -Width 200
}

function Get-TreeInfo([int]$procId) {
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { return 'not running' }
    # cmd.exe -> uv -> python: report the whole tree so CPU/WS are meaningful
    $cpu = 0.0
    $ws = 0.0
    $names = @()
    $queue = [System.Collections.Generic.Queue[int]]::new()
    $queue.Enqueue($procId)
    for ($depth = 0; $depth -lt 4 -and $queue.Count -gt 0; $depth++) {
        $level = $queue.Count
        for ($i = 0; $i -lt $level; $i++) {
            $cur = $queue.Dequeue()
            $p = Get-Process -Id $cur -ErrorAction SilentlyContinue
            if (-not $p) { continue }
            $cpu += [double]$p.CPU
            $ws = [Math]::Max($ws, [double]$p.WorkingSet64 / 1MB)
            if ($p.ProcessName -ne 'conhost') { $names += ('{0} {1}' -f $p.ProcessName, $p.Id) }
            foreach ($k in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$cur" -ErrorAction SilentlyContinue)) {
                $queue.Enqueue([int]$k.ProcessId)
            }
        }
    }
    return ('{0} | CPU {1:N0}s, peak WS {2:N0}MB' -f ($names -join ' -> '), $cpu, $ws)
}

function Job-Detail([string]$log) {
    $j = Parse-Job $log
    $name = [IO.Path]::GetFileNameWithoutExtension($log)
    $pidFile = [IO.Path]::ChangeExtension($log, '.pid')
    $procId = if (Test-Path $pidFile) { [int](Get-Content $pidFile) } else { 0 }
    $elapsed = ([datetime]::Now - $j.Start).TotalMinutes
    $steps = $j.Step + 1
    "job:      $name"
    "process:  $(if ($procId) { Get-TreeInfo $procId } else { 'no pid file' })"
    "log:      $log  (updated $($j.Updated.ToString('HH:mm:ss')))"
    "elapsed:  {0:N1} min" -f $elapsed
    if ($j.Total) {
        $frac = [Math]::Min(1.0, $steps / [double]$j.Total)
        $bar = '#' * [int][Math]::Round($frac * 30) + '-' * (30 - [int][Math]::Round($frac * 30))
        "progress: [$bar] {0}/{1} ({2:N1}%)" -f $steps, $j.Total, ($frac * 100)
        $aliveNow = [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
        if ($j.Rate -gt 0 -and $aliveNow) {
            $left = ($j.Total - $steps) / $j.Rate / 60.0
            "rate:     {0:N2} steps/s" -f $j.Rate
            "eta:      {0:N0} min, finish about {1}" -f $left, (Get-Date).AddMinutes($left).ToString('HH:mm')
        }
        elseif (-not $aliveNow) {
            'state:    stopped before the last step (see log tail)'
        }
    }
    $cfg = Join-Path $root "ml\configs\$name.yaml"
    if (Test-Path $cfg) {
        $stageA = (Select-String -Path $cfg -Pattern '^stage_a_steps:\s*(\d+)').Matches
        if ($stageA.Count -gt 0) {
            $a = [int]$stageA[0].Groups[1].Value
            $phase = if ($steps -le $a) { "A: procedural + hard cases (real data starts at $a)" } else { 'B: real stems + hard cases' }
            "phase:    $phase"
        }
    }
    "warnings: $($j.Warn)  recoveries: $($j.Recover)  skipped steps: $($j.Skip)"
    if ($j.Ckpt) { "last ckpt: $($j.Ckpt)" }
    if ($j.Mix) { "last mix:  $($j.Mix)" }
    if ($j.Done) { "result:   $($j.Last)" }
    ''
    '--- last lines ---'
    Get-Content -Path $log -Encoding utf8 -Tail $Tail | ForEach-Object { '  ' + $_ }
}

function Show-All {
    if ($Watch -gt 0) { Clear-Host }
    "== AVIS background jobs == $(Get-Date -Format 'HH:mm:ss') =="
    ''
    Job-Table
    if ($Name) {
        $log = Join-Path $bgDir "$Name.log"
        if (Test-Path $log) { Job-Detail $log }
    }
    $gpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($gpu) {
        $u = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
        if ($u) { "gpu: $u" }
    }
    if ($Watch -gt 0) { "refreshing every $Watch s, Ctrl+C to stop" }
}

function Test-JobAlive([string]$jobName) {
    $pidFile = Join-Path $bgDir "$jobName.pid"
    if (-not (Test-Path $pidFile)) { return $false }
    $procId = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
    return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
}

if ($Watch -gt 0) {
    if ($Name -and $Host.UI.RawUI) {
        try { $Host.UI.RawUI.WindowTitle = "AVIS: $Name" } catch { }
    }
    while ($true) {
        Show-All
        if ($UntilDone -and $Name -and -not (Test-JobAlive $Name)) { break }
        Start-Sleep -Seconds $Watch
    }
    Show-All
    "job '$Name' is no longer running ($(Get-Date -Format 'HH:mm:ss'))"
} else {
    Show-All
}

if ($Pause -and $Host.Name -ne 'ServerRemoteHost') {
    try { Read-Host 'press Enter to close this window' | Out-Null } catch { }
}
