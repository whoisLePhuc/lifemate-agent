$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskBaseline = Join-Path $taskRepo '.venv\Scripts\python.exe'
$taskTraining = Join-Path $taskRepo '.venv-training'
if (-not (Test-Path -LiteralPath $taskBaseline)) { throw 'Create the frozen baseline environment first.' }
if (-not (Test-Path -LiteralPath $taskTraining)) {
    & uv venv --python $taskBaseline $taskTraining
    if ($LASTEXITCODE -ne 0) { throw 'Training environment creation failed.' }
}
$taskSite = Join-Path $taskTraining 'Lib\site-packages'
if (-not (Test-Path -LiteralPath $taskSite)) { throw 'Unexpected Windows training environment layout.' }
$taskBaselineSite = (Resolve-Path -LiteralPath (Join-Path $taskRepo '.venv\Lib\site-packages')).Path
[System.IO.File]::WriteAllText((Join-Path $taskSite 'baseline-dependencies.pth'), $taskBaselineSite + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
& uv pip install --python (Join-Path $taskTraining 'Scripts\python.exe') --no-deps -r (Join-Path $taskRepo 'requirements-training.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training overlay installation failed.' }
Write-Output 'Training overlay ready; baseline packages were not upgraded.'
