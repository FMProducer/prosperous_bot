@echo off
title Prosperous Bot — Restart...
color 0E

echo ========================================
echo   Перезапуск всех процессов...
echo ========================================
echo.

cd /d C:\Python\Prosperous_Bot\futures_portfolio

REM --- Загрузка .env ---
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

call pm2 restart all --update-env
call pm2 save

echo.
echo Все процессы перезапущены.
timeout /t 3 /nobreak >nul
