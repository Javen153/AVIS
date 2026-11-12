# Wait for a background job (logs\bg\<name>.pid) to exit, then print its result.
#
#   powershell -NoProfile -File tools\wait_job.ps1 -Name v3 -Tail 25
#
# Meant to be run as a background shell: the agent gets a completion notification
# when the job it watches is over. Prints a heartbeat every -Every minutes so the
# waiter itself is visibly alive; exits 0 when the job ended, 2 when it timed out.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Tail = 25,
    [int]$Every = 5,
    [int]$TimeoutMin = 720
)
$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = 'en-US'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "logs\bg\$Name.pid"
$log = Join-Path $root "logs\bg\$Name.log"

if (-not (Test-Path $pidFile)) { Write-Host "no pid file: $pidFile"; exit 2 }
$procId = [int](Get-Content $pidFile)
if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
    Write-Host "job '$Name' (PID $procId) is already finished"
} else {
    Write-Host ("waiting for '{0}' PID {1} (up to {2} min, heartbeat every {3} min)" -f $Name, $procId, $TimeoutMin, $Every)
    $deadline = (Get-Date).AddMinutes($TimeoutMin)
    while ($true) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $p) { break }
        if ((Get-Date) -gt $deadline) {
            Write-Host "timeout after $TimeoutMin min; job is still running"
            exit 2
        }
        Start-Sleep -Seconds ($Every * 60)
        $last = (Get-Content $log -Encoding utf8 -Tail 1 -ErrorAction SilentlyContinue)
        Write-Host ("[{0}] still running: {1}" -f (Get-Date -Format 'HH:mm:ss'), $last)
    }
    Write-Host ("job '{0}' finished at {1}" -f $Name, (Get-Date -Format 'HH:mm:ss'))
}

Write-Host ''
Write-Host '--- last lines ---'
Get-Content $log -Encoding utf8 -Tail $Tail -ErrorAction SilentlyContinue
$ck = Join-Path $root "ml\models\$Name\ckpt_best.pt"
if (Test-Path $ck) {
    $mb = [Math]::Round((Get-Item $ck).Length / 1MB, 1)
    Write-Host ''
    Write-Host ("ckpt_best.pt present ({0} MB), last write {1:HH:mm:ss}" -f $mb, (Get-Item $ck).LastWriteTime)
} else {
    Write-Host ''
    Write-Host "no ckpt_best.pt for '$Name'"
}
exit 0
