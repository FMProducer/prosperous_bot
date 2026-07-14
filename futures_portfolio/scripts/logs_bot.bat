@echo off
title Prosperous Bot — Logs
color 0B

echo ========================================
echo   Просмотр логов (Ctrl+C для выхода)
echo ========================================
echo.
echo Команды:
echo   pm2 logs               — все логи
echo   pm2 logs real-yfi      — логи YFI
echo   pm2 logs real-grass    — логи GRASS
echo   pm2 logs supervisor-service — логи супервизора
echo.

pm2 logs
