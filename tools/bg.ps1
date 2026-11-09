param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Cmd,
    [string]$WorkDir = ''
)
$ErrorActionPreference = 'Stop'
# Start a detached background process via Win32_Process Create (not tied to this shell).
$root = Split-Path -Parent $PSScriptRoot
$bgDir = Join-Path $root 'logs\bg'
New-Item -ItemType Directory -Force -Path $bgDir | Out-Null
if (-not $WorkDir) { $WorkDir = $root }
$log = Join-Path $bgDir "$Name.log"
$pidFile = Join-Path $bgDir "$Name.pid"
$full = 'cmd.exe /d /s /c "' + $Cmd.Replace('"','""') + ' >> "' + $log + '" 2>&1"'
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = 0 }
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $full
    CurrentDirectory = $WorkDir
    ProcessStartupInformation = $startup
}
if ($res.ReturnValue -ne 0) { throw ("Win32_Process.Create failed: " + $res.ReturnValue) }
Set-Content -Path $pidFile -Value $res.ProcessId
Write-Host ("Started '{0}' PID={1}" -f $Name, $res.ProcessId)
Write-Host ("Log: {0}" -f $log)
