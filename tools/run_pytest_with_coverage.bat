@echo off
cd /d %~dp0
%~dp0.venv\Scripts\python.exe -m pytest -q --cov=exchange_api --cov=rebalance_engine --cov-fail-under=90
pause
