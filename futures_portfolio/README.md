# Futures Rebalance Bot

## Описание
Скрипт для динамической ребалансировки фьючерсного портфеля на Binance Futures. 
Алгоритм рассчитывает текущие доли активов на основе спот-цен, проверяет отклонения от целевых долей и при необходимости отправляет рыночные ордера для восстановления целевых распределений.

## Структура проекта
```
futures_portfolio/
├── main.py               # Асинхронный цикл ребалансировки
├── config.json           # Конфигурация портфеля
├── requirements.txt      # Зависимости
├── connector.py          # Инициализация Binance Futures и Spot клиентов
├── calculator.py         # Расчёт стоимости, долей, отклонений
├── executor.py           # Расчёт ордеров и отправка на Binance Futures
├── logs/
│   └── rebalance.log     # Логи ребалансировки
└── tests/
    ├── test_calculator.py
    └── test_executor.py
```

## Установка
1. Клонируйте репозиторий или скопируйте папку `futures_portfolio`.
2. Создайте виртуальное окружение (рекомендуется Python 3.10+):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Установите зависимости:
   ```powershell
   pip install -r requirements.txt
   ```

## Конфигурация
Отредактируйте `config.json`:
- `api_key` и `secret_key` — ваши учетные данные Binance.
- `testnet` — `true` для тестовой сети, `false` для продакшена.
- `portfolios[].targets` — целевые доли активов (сумма должна быть 1.0).
- `portfolios[].rebalance_threshold` — порог отклонения (по умолчанию 0.02 = 2%).
- `portfolios[].check_interval_sec` — интервал проверки в секундах.
- `tickers` — список трейдинговых пар.

## Запуск
```powershell
python main.py --config config.json --interval 60
```
Параметры:
- `--config` — путь к конфиг-файлу (по умолчанию `config.json`).
- `--interval` — интервал проверки в секундах (по умолчанию 60).

Логи будут записаны в `logs/rebalance.log`.

## Тестирование
Для запуска unit-тестов:
```powershell
pytest
```
Тесты проверяют расчёты в `calculator.py` и `executor.py` с использованием моков.

## Важно
- Перед запуском на реальном аккаунте убедитесь, что `testnet` установлен в `true` и протестируйте на Testnet.
- Обработка ошибок и точная настройка `reduceOnly` могут быть доработаны в дальнейшем.
- Текущая реализация не включает точное округление количества контрактов (precision) — это может быть добавлено при необходимости.
