#!/bin/bash
set -e

# setup_env.sh - Script to prepare the environment and run tests

STRICT_MODE=false
if [[ "$1" == "--strict" ]]; then
    STRICT_MODE=true
fi

echo "🚀 Setting up environment..."

# Install dependencies from requirements.txt
if [[ -f requirements.txt ]]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found!"
fi

# Install the package in editable mode
if [[ -f pyproject.toml ]]; then
    echo "📦 Installing package in editable mode..."
    pip install -e .
else
    echo "⚠️ pyproject.toml not found!"
fi

echo "✅ Environment setup complete."

# Run tests
echo "🧪 Running tests..."
# Create reports directory if it doesn't exist
mkdir -p reports

if [[ "$STRICT_MODE" == true ]]; then
    echo "🛡️ Running tests in strict mode (coverage >= 90%)..."
    # PYTHONPATH is already set in ci.yml but we add it here for local runs as well
    # We use --cov=src to measure coverage of the source code
    # We use --cov-report=html:reports/coverage to generate a report for artifact upload
    PYTHONPATH=.:futures_portfolio:src python3 -m pytest tests/ --ignore=tests/test_ensemble_logic.py --cov=src --cov-report=term-missing --cov-report=html:reports/coverage --cov-fail-under=90
else
    PYTHONPATH=.:futures_portfolio:src python3 -m pytest tests/ --ignore=tests/test_ensemble_logic.py
fi

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌ Tests failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "🎉 All tests passed!"
fi
