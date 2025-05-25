@echo off
cd /d %~dp0
.venv\Scripts\python.exe -m pytest -q --cov=exchange_api --cov-fail-under=90
pause
