@echo off
title Prosperous Bot — Status
color 0E

echo ========================================
echo   Статус системы
echo ========================================
echo.

echo --- Процессы ---
pm2 list
echo.

echo --- Использование пам/CPU ---
pm2 monit --no-interaction 2>nul || echo (monit требует интерактивный режим)
echo.

echo Нажмите любую клавишу для закрытия...
pause >nul
