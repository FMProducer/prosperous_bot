@echo off
echo ============================================
echo  Prosperous Bot - Full Restart Script
echo ============================================
cd /d C:\Python\Prosperous_Bot\futures_portfolio

echo.
echo [1/5] Stopping all paper bots...
pm2 stop supervisor-service
timeout /t 2 /nobreak >nul

echo.
echo [2/5] Deleting paper state files (fresh start)...
del /q paper_state_*.json 2>nul
del /q paper_shadow_*.json 2>nul
echo Done.

echo.
echo [3/5] Verify config.json is correct:
type config.json | findstr "BASE_LONG BASE_SHARE VIRTUAL"
type config.json | findstr "supervisor_interval"

echo.
echo [4/5] Starting supervisor-service...
pm2 start supervisor-service
timeout /t 3 /nobreak >nul

echo.
echo [5/5] Checking status...
pm2 list | findstr "supervisor\|paper-"

echo.
echo ============================================
echo  Checking latest supervisor log...
echo ============================================
for /f "tokens=*" %%i in ('dir /o-d /b logs\supervisor.log 2^>nul') do (
    type logs\supervisor.log
    goto :done
)
:done

echo.
echo ============================================
echo  DONE. Check PnL in 5-10 minutes.
echo ============================================
pause
