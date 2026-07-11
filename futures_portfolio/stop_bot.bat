@echo off
title Prosperous Bot — Stopping...
color 0C

echo ========================================
echo   Остановка всех процессов...
echo ========================================
echo.

pm2 stop all
pm2 save

echo.
echo Все процессы остановлены.
timeout /t 3 /nobreak >nul
