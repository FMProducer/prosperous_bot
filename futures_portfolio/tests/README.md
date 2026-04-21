# Тестирование системы ребалансировки портфеля

Данная директория содержит набор юнит-тестов для обеспечения надежности системы высокочастотного ребалансировки.

## Установка зависимостей

Для запуска тестов необходимо установить дополнительные пакеты:

```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov python-binance aiohttp python-dotenv
```

## Запуск тестов

Для запуска всех тестов с генерацией отчета о покрытии используйте следующую команду из корня проекта:

```bash
PYTHONPATH=futures_portfolio pytest --cov=futures_portfolio futures_portfolio/tests/
```

## Структура тестов

- `test_calculator.py`: Математические расчеты долей, PnL и отклонений. Покрытие 100%.
- `test_executor.py`: Исполнение ордеров (Market/Limit), логика fallback и проверки минимальной стоимости.
- `test_connector.py`: Взаимодействие с Binance API, обработка сетевых ошибок и повторные попытки.
- `test_notifier.py`: Отправка уведомлений в Telegram. Покрытие 100%.
- `test_rank_tickers.py`: Логика сканирования и скоринга тикеров.
- `test_main.py`: Основной цикл ребалансировки, логика siphoning, Trailing Stop и защита по марже.

## Особенности реализации

Тесты полностью изолированы и используют моки (mocks) для всех внешних взаимодействий:
- Binance API (через моки `BinanceConnector` и `Client`).
- Telegram API (через моки `aiohttp.ClientSession`).
- Файловая система (через перехват `load_json` и `save_json`).
