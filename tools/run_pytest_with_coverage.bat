@echo off
REM Change current directory to the project root (one level up from where this script is)
cd /d %~dp0..\

REM Now the CWD is the project root (e.g., C:\Python\Prosperous_Bot\)
REM Execute python from the venv, using a path relative to the new CWD
.\.venv\Scripts\python.exe -m pytest -q --cov=exchange_api --cov=rebalance_engine --cov-fail-under=90
pause
