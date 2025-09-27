Param()
$ErrorActionPreference = "Stop"
Write-Host "[init] Windows bootstrap starting…" -ForegroundColor Cyan

# 1) Create virtualenv (idempotent)
python -m venv .venv

# 2) Use venv's interpreter directly (без активации)
$VENV_PY = Join-Path $PWD ".venv\Scripts\python.exe"

# 3) Tools and deps
& $VENV_PY -m pip install --upgrade pip setuptools wheel
if (Test-Path "requirements.txt") {
  & $VENV_PY -m pip install -r requirements.txt
}
& $VENV_PY -m pip install pytest pytest-cov pytest-asyncio pandas ruff

# 4) Diagnostics
& $VENV_PY -V
& $VENV_PY -m pip -V

Write-Host ""
Write-Host "✅ Windows env ready." -ForegroundColor Green
Write-Host "To activate (optional): .\.venv\Scripts\Activate.ps1"
Write-Host "Unit-only local test: .\.venv\Scripts\python -m pytest -q -m 'not integration' --maxfail=1"
Write-Host ""
