#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Environment Setup for prosperous_bot ---

echo "Setting up Python environment..."

# 1. Create a virtual environment to isolate project dependencies.
python3 -m venv .venv

# 2. Activate the virtual environment.
source .venv/bin/activate

# 3. Upgrade pip to the latest version.
pip install --upgrade pip

# 4. Install the application's dependencies from requirements.txt.
# This file has a UTF-16 encoding, but pip handles it correctly.
pip install -r requirements.txt

# 5. Install additional dependencies for testing and static analysis,
# as indicated by the CI workflow (.github/workflows/ci.yml) and the project roadmap.
pip install pytest pytest-cov pytest-asyncio pandas ruff

# 6. Set the PYTHONPATH to include the project's root directory.
# This is necessary because the project uses a `src` layout, and this
# configuration is specified in `pytest.ini`.
export PYTHONPATH=.

echo ""
echo "✅ Environment setup complete."
echo "A virtual environment has been created in the '.venv' directory."
echo "To activate it in a new terminal, run: source .venv/bin/activate"
echo ""
echo "You can now run the following commands:"
echo " - To run tests: pytest"
echo " - To check code style: ruff check ."
echo ""
