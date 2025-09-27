#!/usr/bin/env bash
# Initial setup for Jules AI (Prosperous Bot)
# Usage:
#   bash setup_env.sh --probe   # быстрый прогон (unit-only)
#   bash setup_env.sh --strict  # строгий прогон (coverage >= 90%, артефакты в ./reports)
#   bash setup_env.sh --reinstall  # пересоздать venv
set -Eeuo pipefail

# --- Guard: auto-fix CRLF and re-exec -----------------------------------------
if grep -q $'' "$0" 2>/dev/null; then
  echo "[setup] CRLF detected; re-executing sanitized script..."
  exec /usr/bin/env bash <(tr -d '\r' < "$0") "$@"
fi

# --- Repo-State Header ---------------------------------------------------------
if git rev-parse --git-dir >/dev/null 2>&1; then
  DEFAULT_BRANCH="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || echo "prosperous_bot")"
  LAST_SHA="$(git rev-parse HEAD)"
  LAST_MSG="$(git log -1 --pretty=%s)"
  echo "=== Repo-State Header ==="
  echo "Default branch: ${DEFAULT_BRANCH}"
  echo "Last commit: ${LAST_SHA} — ${LAST_MSG}"
  echo "Commit link: https://github.com/FMProducer/prosperous_bot/commit/${LAST_SHA}"
  echo "========================="
fi

PROJECT_ROOT="$(pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
REPORTS_DIR="${PROJECT_ROOT}/reports"
CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
mkdir -p "${REPORTS_DIR}" "${CACHE_DIR}"
export PIP_CACHE_DIR="${CACHE_DIR}"
export PYTHONUTF8=1

# Cross-platform path to python inside venv
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin || "$OSTYPE" == \"win32\" || "${OS:-}" == "Windows_NT" ]]; then
  PYBIN="${VENV_DIR}/Scripts/python.exe"
else
  PYBIN="${VENV_DIR}/bin/python"
fi

# Args
MODE="probe"   # probe|strict
REINSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) MODE="probe"; shift;;
    --strict) MODE="strict"; shift;;
    --reinstall) REINSTALL=1; shift;;
    *) echo "[setup] Unknown arg: $1"; exit 2;; 
  esac
done

# Create/refresh venv
echo "[setup] Creating/using virtualenv..."
if [[ ! -x "${PYBIN}" || "${REINSTALL}" -eq 1 ]]; then
  BASE_PY="${PYTHON:-python3}"
  command -v "${BASE_PY}" >/dev/null 2>&1 || BASE_PY="python"
  "${BASE_PY}" -m venv "${VENV_DIR}"
fi

echo "[setup] Upgrading pip/setuptools/wheel..."
"${PYBIN}" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

echo "[setup] Installing dependencies..."
"${PYBIN}" -m pip install --disable-pip-version-check -r "${PROJECT_ROOT}/requirements.txt"

echo "[setup] Installing project (editable)..."
"${PYBIN}" -m pip install --disable-pip-version-check -e "${PROJECT_ROOT}"

echo "[setup] Environment verification..."
"${PYBIN}" --version
"${PYBIN}" -m pip --version
"${PYBIN}" -m pytest --version || true
"${PYBIN}" -m pip freeze > "${REPORTS_DIR}/pip_freeze.txt"

COMMON_PYTEST="-q --maxfail=1 -ra"
set +e
if [[ "${MODE}" == "probe" ]]; then
  echo "[setup] Running tests (probe: unit-only, no coverage)..."
  "${PYBIN}" -m pytest ${COMMON_PYTEST} -m "not integration"
elif [[ "${MODE}" == "strict" ]]; then
  echo "[setup] Running tests (strict: coverage >= 90%)..."
  "${PYBIN}" -m pytest ${COMMON_PYTEST} \
    --cov=src/prosperous_bot \
    --cov-report=xml:"${REPORTS_DIR}/coverage.xml" \
    --junitxml="${REPORTS_DIR}/junit.xml" \
    --cov-fail-under=90
else
  echo "[setup] Unknown MODE=${MODE}"; exit 2
fi
EXIT_CODE=$?
set -e



if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[setup] Tests failed with code ${EXIT_CODE}"
  exit $EXIT_CODE
fi

echo "[setup] Initial Setup Finished Successfully"