param(
    [string[]]$Only = @('slakh','musdb','groove'),
    [switch]$SkipSlakh,
    [switch]$SkipMusdb
)
$ErrorActionPreference = 'Stop'
# Download AVIS datasets with resume. Long-running: launch through tools\bg.ps1.
$root = Split-Path -Parent $PSScriptRoot
$data = [Environment]::GetEnvironmentVariable('AVIS_DATA','User')
if (-not $data) { $data = 'E:\avis_data' }
$raw = Join-Path $data 'raw'
New-Item -ItemType Directory -Force -Path $raw | Out-Null
$minFreeGb = 40

function FreeGb([string]$p) {
    $d = ([IO.Path]::GetPathRoot($p)).Substring(0,1)
    return (Get-PSDrive $d).Free / 1GB
}
function NeedSpace([double]$gb, [string]$what) {
    $f = FreeGb $data
    Write-Host ("[space] {0:N1} GB free, need ~{1:N1} GB for {2}" -f $f, $gb, $what)
    if ($f - $gb -lt $minFreeGb) { throw "Not enough free space for $what (would leave < ${minFreeGb} GB)" }
}
function Get-File([string]$url, [string]$dest) {
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    & curl.exe -L -C - --retry 10 --retry-delay 5 -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) for $url" }
}
function Check-Md5([string]$path, [string]$expected) {
    Write-Host "[md5] $path"
    $h = (Get-FileHash -Algorithm MD5 -Path $path).Hash.ToLower()
    if ($h -ne $expected) { throw "md5 mismatch: $h != $expected" }
    Write-Host "[md5] OK"
}

# 1. Slakh2100-redux (~104 GB)
if ($Only -contains 'slakh' -and -not $SkipSlakh) {
    $dir = Join-Path $raw 'slakh'
    $marker = Join-Path $dir '.done'
    if (-not (Test-Path $marker)) {
        NeedSpace 110 'Slakh2100'
        $tgz = Join-Path $data 'slakh2100_flac_redux.tar.gz'
        if (-not (Test-Path $tgz)) {
            Get-File 'https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1' $tgz
        }
        Check-Md5 $tgz 'f4b71b6c45ac9b506f59788456b3f0c4'
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host '[slakh] extracting (excluding mix.flac)'
        & tar.exe -xzf $tgz -C $dir --exclude '*/mix.flac'
        if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
        Set-Content -Path $marker -Value (Get-Date -Format o)
        Remove-Item -Force $tgz
        Write-Host '[slakh] done'
    } else { Write-Host '[slakh] already extracted' }
}

# 2. MUSDB18-HQ (~23 GB). Zenodo record 3338373 files are per-track zip (musdb18hq.zip).
if ($Only -contains 'musdb' -and -not $SkipMusdb) {
    $dir = Join-Path $raw 'musdb18hq'
    $marker = Join-Path $dir '.done'
    if (-not (Test-Path $marker)) {
        NeedSpace 30 'MUSDB18-HQ'
        $zip = Join-Path $data 'musdb18hq.zip'
        if (-not (Test-Path $zip)) {
            # Try the record's single archive; fall back to per-file listing handled by user action.
            Get-File 'https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1' $zip
        }
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        & tar.exe -xf $zip -C $dir
        if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
        Set-Content -Path $marker -Value (Get-Date -Format o)
        Remove-Item -Force $zip
        Write-Host '[musdb] done'
    } else { Write-Host '[musdb] already extracted' }
}

# 3. Groove MIDI (~20 MB)
if ($Only -contains 'groove') {
    $dir = Join-Path $raw 'groove'
    $marker = Join-Path $dir '.done'
    if (-not (Test-Path $marker)) {
        $zip = Join-Path $data 'groove-midionly.zip'
        Get-File 'https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip' $zip
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        & tar.exe -xf $zip -C $dir
        if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
        Set-Content -Path $marker -Value (Get-Date -Format o)
        Remove-Item -Force $zip
        Write-Host '[groove] done'
    } else { Write-Host '[groove] already extracted' }
}
Write-Host '[fetch_data] finished'
