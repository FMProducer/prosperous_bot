@echo off
cd /d C:\Python\Prosperous_Bot\futures_portfolio
echo Stopping all paper bots...
pm2 stop supervisor-service
timeout /t 2 /nobreak >nul
echo Starting supervisor-service...
pm2 start supervisor-service
timeout /t 5 /nobreak >nul
echo Checking status...
pm2 list | findstr supervisor
echo Done.
pause
