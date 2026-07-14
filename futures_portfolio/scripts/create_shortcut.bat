@echo off
title Создание ярлыка на рабочем столе
echo.
echo Создание ярлыка "Prosperous Bot" на рабочем столе...
echo.

set "SCRIPT_DIR=%~dp0"
set "DESKTOP=%USERPROFILE%\Desktop"
set "SHORTCUT=%DESKTOP%\Prosperous Bot.bat"

echo @echo off > "%SHORTCUT%"
echo cd /d "%SCRIPT_DIR%" >> "%SHORTCUT%"
echo call start_bot.bat >> "%SHORTCUT%"

echo [OK] Ярлык создан: %SHORTCUT%
echo.
pause
