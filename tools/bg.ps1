# Start a long background job (training, build, eval) and open a live progress window.
#
#   tools\bg.ps1 -Name v3 -Cmd "uv run python -m avis_ml.train --config configs/v3.yaml" -WorkDir "C:\OpenCode\AVIS\ml"
#
# The job runs without a console (all output goes to logs\bg\<name>.log), so no blank
# black window appears. A separate window titled "AVIS: <name>" shows a live dashboard
# (step, %, rate, ETA, phase, log tail) and stays open when the job ends.
# Closing the dashboard does NOT stop the job.
#   -NoMonitor     no dashboard window
#   -Refresh N     dashboard refresh in seconds (default 5)
#   -Visible       give the job its own console (it will look empty: text goes to the log)
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Cmd,
    [string]$WorkDir = '',
    [switch]$Visible,
    [switch]$NoMonitor,
    [int]$Refresh = 5
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$bgDir = Join-Path $root 'logs\bg'
New-Item -ItemType Directory -Force -Path $bgDir | Out-Null
if (-not $WorkDir) { $WorkDir = $root }
$log = Join-Path $bgDir "$Name.log"
$pidFile = Join-Path $bgDir "$Name.pid"
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Set-Content -Path $log -Value '' -Encoding utf8

# The job: a cmd wrapper that appends every stream to the log. -WindowStyle Hidden
# means no console is attached at all (verified: no conhost child), which is what
# removed the empty black windows.
$argList = @('/d', '/s', '/c', ('"' + $Cmd.Replace('"', '""') + ' >> "' + $log + '" 2>&1"'))
$style = if ($Visible) { 'Normal' } else { 'Hidden' }
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList $argList -WorkingDirectory $WorkDir `
    -WindowStyle $style -PassThru
Set-Content -Path $pidFile -Value $p.Id
Write-Host ("Started '{0}' PID={1}" -f $Name, $p.Id)
Write-Host ("Log: {0}" -f $log)

# The dashboard: a real console window (Windows Terminal opens a tab for it, so it
# is visible even though the job itself has no console).
if (-not $NoMonitor) {
    $status = Join-Path $PSScriptRoot 'status.ps1'
    $mon = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -Name {1} -Watch {2} -UntilDone -Pause -Tail 8' -f `
        $status, $Name, $Refresh
    $m = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $mon
        CurrentDirectory = $root
    }
    if ($m.ReturnValue -eq 0) {
        Write-Host ("Progress window 'AVIS: {0}' opened (refresh {1}s). Closing it does not stop the job." -f $Name, $Refresh)
    } else {
        Write-Host ("Progress window failed ({0}); run tools\status.ps1 -Name {1}" -f $m.ReturnValue, $Name)
    }
} else {
    Write-Host ("Status: tools\status.ps1 -Name {0}" -f $Name)
}
