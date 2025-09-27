#!/usr/bin/env bash
set -euo pipefail
echo "[init] POSIX bootstrap starting…"

# 1) Create virtualenv (idempotent)
python3 -m venv .venv

# 2) Use venv's interpreter directly (без 'source')
VENV_PY="$(pwd)/.venv/bin/python"

# 3) Tools and deps
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
if [[ -f requirements.txt ]]; then
  "$VENV_PY" -m pip install -r requirements.txt
fi
"$VENV_PY" -m pip install pytest pytest-cov pytest-asyncio pandas ruff

# 4) Diagnostics
"$VENV_PY" -V
"$VENV_PY" -m pip -V

echo
echo "✅ POSIX env ready."
echo "To activate manually: 'source .venv/bin/activate' (optional)"
echo "To run tests locally (unit-only): '.venv/bin/python -m pytest -q -m \"not integration\" --maxfail=1'"
echo
