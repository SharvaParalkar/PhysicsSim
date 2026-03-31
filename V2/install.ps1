# ─── Genesis Simulation Environment Installer ────────────────────────────────
# Run from your project folder:  .\install.ps1
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$VENV = "venv_sim"
$PY   = ".\$VENV\Scripts\python.exe"
$PIP  = ".\$VENV\Scripts\pip.exe"

function Write-Step($msg) {
    Write-Host "`n>>> $msg" -ForegroundColor Cyan
}

function Write-OK($msg) {
    Write-Host "  ✔  $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "  ✗  $msg" -ForegroundColor Red
    exit 1
}

# ── 0. Find Python 3.11 ───────────────────────────────────────────────────────
Write-Step "Looking for Python 3.11 ..."

$py311 = $null
foreach ($candidate in @("py", "python3.11", "python")) {
    try {
        $ver = & $candidate -3.11 --version 2>&1
        if ($ver -match "3\.11") { $py311 = "$candidate -3.11"; break }
    } catch {}
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "3\.11") { $py311 = $candidate; break }
    } catch {}
}

if (-not $py311) {
    Write-Host "  Python 3.11 not found. Installing via winget ..." -ForegroundColor Yellow
    winget install Python.Python.3.11 --silent
    $py311 = "py -3.11"
}

Write-OK "Using: $py311"

# ── 1. Create venv ────────────────────────────────────────────────────────────
Write-Step "Creating virtual environment '$VENV' ..."

if (Test-Path "$VENV\Scripts\python.exe") {
    Write-OK "venv already exists, skipping creation"
} else {
    Invoke-Expression "$py311 -m venv $VENV"
    if (-not (Test-Path "$VENV\Scripts\python.exe")) {
        Write-Fail "venv creation failed"
    }
    Write-OK "venv created"
}

# ── 2. Upgrade pip inside venv ────────────────────────────────────────────────
Write-Step "Upgrading pip ..."
& $PY -m pip install --upgrade pip --quiet
Write-OK "pip upgraded"

# ── 3. PyTorch (CPU) ──────────────────────────────────────────────────────────
Write-Step "Installing PyTorch (CPU) ..."
& $PIP install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "PyTorch install failed" }
Write-OK "PyTorch installed"

# ── 4. Core simulation dependencies ──────────────────────────────────────────
Write-Step "Installing core dependencies ..."
$core = @(
    "numpy",
    "scipy",
    "pandas",
    "networkx",
    "h5py",
    "trimesh",
    "coacd",
    "imageio[ffmpeg]",
    "fastapi",
    "uvicorn",
    "websockets"
)
& $PIP install @core --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Core dependency install failed" }
Write-OK "Core dependencies installed"

# ── 5. Genesis ────────────────────────────────────────────────────────────────
Write-Step "Installing Genesis ..."
& $PIP install genesis-world --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Genesis install failed" }
Write-OK "Genesis installed"

# ── 6. Quick smoke test ───────────────────────────────────────────────────────
Write-Step "Running smoke test ..."

$test = @"
import sys
failures = []

deps = ["numpy", "scipy", "pandas", "networkx", "h5py", "trimesh", "coacd",
        "torch", "genesis"]
for d in deps:
    try:
        __import__(d)
    except Exception as e:
        failures.append(f"{d}: {e}")

if failures:
    print("FAIL")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)
else:
    print("OK")
"@

$result = & $PY -c $test
if ($result -match "^OK") {
    Write-OK "All packages importable"
} else {
    Write-Host "`n  Some packages failed to import:" -ForegroundColor Yellow
    Write-Host $result
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n" + ("─" * 55) -ForegroundColor Cyan
Write-Host "  Install complete!" -ForegroundColor Green
Write-Host "  To activate your environment:" -ForegroundColor White
Write-Host "    .\$VENV\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  To run the checker:" -ForegroundColor White
Write-Host "    .\$VENV\Scripts\python.exe gen.py" -ForegroundColor Yellow
Write-Host "  To run the simulation:" -ForegroundColor White
Write-Host "    .\$VENV\Scripts\python.exe simulation.py" -ForegroundColor Yellow
Write-Host ("─" * 55) -ForegroundColor Cyan