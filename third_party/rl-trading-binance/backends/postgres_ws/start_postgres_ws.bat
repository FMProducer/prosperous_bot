@echo off
REM This script starts the Binance WebSocket data collector.
REM Make sure you have a Python virtual environment set up and activated.
REM If not, you can create one and install dependencies using:
REM python -m venv .venv
REM .venv\Scripts\activate
REM pip install -r ..\..\..\..\requirements.txt

set SCRIPT_DIR=%~dp0
cd "%SCRIPT_DIR%"

REM Activate the virtual environment
call "..\..\..\..\.venv\Scripts\activate.bat"

IF EXIST "ref_config.yml" (
    echo Starting collector.py with ref_config.yml...
    python collector.py --config ref_config.yml
) ELSE (
    echo Error: ref_config.yml not found in %SCRIPT_DIR%
)

pause
