@echo off
title Prosperous Bot — Starting...
color 0A

echo ========================================
echo   Prosperous Bot — Futures Portfolio
echo ========================================
echo.

cd /d C:\Python\Prosperous_Bot\futures_portfolio

REM --- Проверка Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден в PATH
    pause
    exit /b 1
)

REM --- Проверка PM2 ---
pm2 --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Установка PM2...
    call npm install -g pm2
    call pm2 install pm2-windows-startup
)

REM --- Загрузка переменных окружения из .env ---
if exist .env (
    echo [OK] Загрузка .env...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
) else (
    echo [WARN] Файл .env не найден, используются системные переменные
)

REM --- Проверка API ключей ---
if "%BINANCE_API_KEY%"=="" (
    echo [ERROR] BINANCE_API_KEY не установлен
    echo Создайте файл .env или установите переменные через setx
    pause
    exit /b 1
)
if "%BINANCE_SECRET_KEY%"=="" (
    echo [ERROR] BINANCE_SECRET_KEY не установлен
    pause
    exit /b 1
)

echo [OK] API ключи загружены
echo.

REM --- Запуск сервисов (supervisor, aggregator, telegram) ---
echo [1/2] Запуск сервисов...
call pm2 start ecosystem.config.js --update-env --silent
timeout /t 2 /nobreak >nul

REM --- Сохранение состояния ---
echo [2/2] Сохранение состояния PM2...
call pm2 save --silent

echo.
echo ========================================
echo   Все процессы запущены!
echo ========================================
echo.
call pm2 list
echo.
echo Команды:
echo   pm2 logs          — все логи
echo   pm2 logs real-yfi — логи конкретного бота
echo   pm2 restart all   — перезапуск всех
echo   pm2 stop all      — остановка всех
echo.
echo Нажмите любую клавишу для закрытия...
pause >nul
