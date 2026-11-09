# Build the whole AVIS solution (CLI, Core, Audio, Render, tests, App).
# Usage: tools\build.ps1 [-Test] [-Configuration Release]
param(
    [switch]$Test,
    [string]$Configuration = 'Release'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'build.log'
Set-Content -Path $log -Value ("# build {0} {1}" -f (Get-Date -Format s), $Configuration)

# Build explicit x64 outputs (bin\x64\Release); solution-level platform mapping
# for the .slnx file does not propagate to the projects.
$projects = @(
    'runtime\src\Avis.Core\Avis.Core.csproj',
    'runtime\src\Avis.Audio\Avis.Audio.csproj',
    'runtime\src\Avis.Render\Avis.Render.csproj',
    'runtime\src\Avis.Cli\Avis.Cli.csproj',
    'runtime\tests\Avis.Tests\Avis.Tests.csproj',
    'runtime\src\Avis.App\Avis.App.csproj'
)

$sw = [Diagnostics.Stopwatch]::StartNew()
$code = 0
foreach ($proj in $projects) {
    & dotnet build $proj -c $Configuration -p:Platform=x64 -v:m *>> $log
    if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; break }
}
$sw.Stop()

Get-Content $log -Tail 4
Write-Host ("build: {0} in {1:N1} s -> {2}" -f $(if ($code -eq 0) { 'PASS' } else { 'FAIL' }), $sw.Elapsed.TotalSeconds, $log)

if ($Test -and $code -eq 0) {
    $testLog = Join-Path $logDir 'test.log'
    & dotnet test 'runtime\tests\Avis.Tests\Avis.Tests.csproj' -c $Configuration -p:Platform=x64 --no-build *> $testLog
    $tcode = $LASTEXITCODE
    Get-Content $testLog -Tail 4
    Write-Host ("tests: {0} -> {1}" -f $(if ($tcode -eq 0) { 'PASS' } else { 'FAIL' }), $testLog)
    if ($tcode -ne 0) { $code = $tcode }
}

exit $code
