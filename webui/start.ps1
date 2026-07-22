# start.ps1 -- starts engine, backend + frontend together
# Usage: ./start.ps1

$ENGINE_DIR = (Join-Path (Split-Path $PSScriptRoot -Parent) "Manga Translator")
$BACKEND = Join-Path $PSScriptRoot "backend"
$FRONTEND = Join-Path $PSScriptRoot "frontend"
$DEBUG_LOG = Join-Path (Split-Path $PSScriptRoot -Parent) "debug-8edb45.log"
$DEBUG_RUN_ID = "post-fix"

#region agent log
function Write-DebugLog {
    param([string]$HypothesisId, [string]$Location, [string]$Message, [hashtable]$Data)
    $entry = @{
        sessionId    = "8edb45"
        runId        = $DEBUG_RUN_ID
        hypothesisId = $HypothesisId
        location     = $Location
        message      = $Message
        data         = $Data
        timestamp    = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    } | ConvertTo-Json -Compress
    Add-Content -Path $DEBUG_LOG -Value $entry -Encoding UTF8
}
#endregion

function Resolve-Python {
    param([string[]]$Candidates, [string]$ServiceName)

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    $systemPython = if ($pythonCmd) { $pythonCmd.Source } else { $null }
    if ($systemPython -and (Test-Path -LiteralPath $systemPython)) {
        Write-Host "WARNING: No dedicated venv for $ServiceName - using system Python: $systemPython" -ForegroundColor Yellow
        return $systemPython
    }

    return $null
}

$ENGINE_PYTHON = Resolve-Python -ServiceName "Engine" -Candidates @(
    (Join-Path $ENGINE_DIR "venv\Scripts\python.exe")
)

$BACKEND_PYTHON = Resolve-Python -ServiceName "Backend" -Candidates @(
    (Join-Path $BACKEND ".venv\Scripts\python.exe"),
    (Join-Path $ENGINE_DIR "venv\Scripts\python.exe")
)

Write-Host "MangaFlow - Starting services..." -ForegroundColor Cyan

#region agent log
Write-DebugLog -HypothesisId "H1" -Location "start.ps1:paths" -Message "Resolved python paths" -Data @{
    enginePython       = $ENGINE_PYTHON
    backendPython      = $BACKEND_PYTHON
    enginePythonExists = [bool]$ENGINE_PYTHON -and (Test-Path -LiteralPath $ENGINE_PYTHON)
    backendPythonExists = [bool]$BACKEND_PYTHON -and (Test-Path -LiteralPath $BACKEND_PYTHON)
    engineDirExists    = (Test-Path -LiteralPath $ENGINE_DIR)
    engineApiExists    = (Test-Path -LiteralPath (Join-Path $ENGINE_DIR "engine_api.py"))
    backendDirExists   = (Test-Path -LiteralPath $BACKEND)
}
#endregion

function Start-ServiceProcess {
    param(
        [string]$PythonPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$ServiceName
    )

    if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath)) {
        Write-Host "ERROR: Could not find Python for $ServiceName." -ForegroundColor Red
        Write-Host "  Create the engine venv:" -ForegroundColor Yellow
        Write-Host "    cd `"$ENGINE_DIR`"" -ForegroundColor Gray
        Write-Host "    python -m venv venv" -ForegroundColor Gray
        Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
        Write-Host "    pip install -r requirements.txt -r requirements-cli.txt fastapi uvicorn" -ForegroundColor Gray
        return $false
    }

    #region agent log
    Write-DebugLog -HypothesisId "H1" -Location "start.ps1:start-service" -Message "Launching service" -Data @{
        serviceName = $ServiceName
        pythonPath = $PythonPath
        arguments = $Arguments
        workingDirectory = $WorkingDirectory
    }
    #endregion

    Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -WindowStyle Normal
    return $true
}

# 1. Start Engine on port 8001 (if not already running)
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8001/api/engine/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
    Write-Host "Engine already running on :8001" -ForegroundColor Green
} catch {
    Write-Host "Starting FastAPI Engine on :8001..." -ForegroundColor Yellow
    $started = Start-ServiceProcess -PythonPath $ENGINE_PYTHON -Arguments "engine_api.py" -WorkingDirectory $ENGINE_DIR -ServiceName "Engine"
    if (-not $started) { Write-Host "Engine failed to start." -ForegroundColor Red }
}

# 2. Start FastAPI Backend on port 8000 (if not already running)
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
    Write-Host "Backend already running on :8000" -ForegroundColor Green
} catch {
    Write-Host "Starting FastAPI Backend on :8000..." -ForegroundColor Yellow
    $started = Start-ServiceProcess -PythonPath $BACKEND_PYTHON -Arguments "-m uvicorn main:app --reload --port 8000" -WorkingDirectory $BACKEND -ServiceName "Backend"
    if (-not $started) { Write-Host "Backend failed to start." -ForegroundColor Red }
}

# 3. Start Next.js frontend on port 3000 (if not already running)
try {
    $null = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
    Write-Host "Frontend already running on :3000" -ForegroundColor Green
} catch {
    Write-Host "Starting Next.js Frontend on :3000..." -ForegroundColor Yellow
    Start-Process -FilePath "npm" -ArgumentList "run dev" -WorkingDirectory $FRONTEND -WindowStyle Normal
}

Write-Host ""
Write-Host "Services initialized!" -ForegroundColor Green
Write-Host "MangaFlow UI: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Engine API:   http://localhost:8001" -ForegroundColor Cyan

# Open in browser
Start-Sleep 2
Start-Process "http://localhost:3000"
