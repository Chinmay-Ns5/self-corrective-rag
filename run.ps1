$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $projectRoot '..\..\.venv\Scripts\python.exe')
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $python) {
    throw 'No local virtual environment found. Create .venv in this project or the PE workspace.'
}

Push-Location $projectRoot
try {
    # The existing local index was built with a cached embedding model.
    # Skip the model hub check that can stall when offline or sandboxed.
    if (Test-Path -LiteralPath (Join-Path $projectRoot 'data\vectorstore\chroma.sqlite3')) {
        $env:HF_HUB_OFFLINE = '1'
    }
    Write-Host "Using Python: $python"
    Write-Host 'Starting Streamlit at http://localhost:8501 (Ctrl+C to stop)'
    & $python -m streamlit run app.py
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
