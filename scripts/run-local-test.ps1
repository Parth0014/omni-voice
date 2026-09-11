param(
    [string]$PythonPath = "",
    [string]$SettingsPath = "",
    [switch]$CheckOnly,
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
try {
    $candidates = @()
    if ($PythonPath) { $candidates += $PythonPath }
    if ($env:OMNIVOICE_PYTHON) { $candidates += $env:OMNIVOICE_PYTHON }
    $candidates += @(
        (Join-Path $projectRoot '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs\omnivoice\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\envs\omnivoice\python.exe')
    )
    $pathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPython) { $candidates += $pathPython.Source }
    $selectedPython = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ('gradio_client','scipy','soundfile','numpy','bs4','imageio_ffmpeg')) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $selectedPython = $candidate; break }
    }
    if (-not $selectedPython) {
        throw 'No ready OmniVoice Python environment found. Create the omnivoice Conda environment from environment.yml, install the project dependencies, or set OMNIVOICE_PYTHON to its python.exe.'
    }
    Write-Host "Python: $selectedPython"
    $runnerArguments = @('-u', (Join-Path $projectRoot 'local_test.py'))
    if ($SettingsPath) { $runnerArguments += @('--settings', $SettingsPath) }
    if ($CheckOnly) { $runnerArguments += '--check' }
    if ($NoOpen) { $runnerArguments += '--no-open' }
    & $selectedPython @runnerArguments
    exit $LASTEXITCODE
} catch {
    Write-Host "Local test could not start: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
