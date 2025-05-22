@echo off
cd /d %~dp0
%~dp0.venv\Scripts\python.exe -m pytest -q --cov=adaptive_agent.exchange_api --cov-fail-under=90
pause
