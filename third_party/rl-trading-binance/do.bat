@echo off
:: Быстрый запуск AgentRouter
:: Использование: do "Текст задачи" [тип_задачи]
:: Пример: do "Напиши функцию RSI" code_writing

set TYPE=%2
if "%TYPE%"=="" set TYPE=code_writing

python main.py --prompt "%~1" --type %TYPE%