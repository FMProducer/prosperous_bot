# Prosperous Bot — Map of Content

> Единый навигационный центр проекта ребалансировщика.
> Обновлено: 2026-07-17

---

## 🏗️ Архитектура

| Файл | Назначение |
|------|------------|
| `main.py` | Entry point, 24h auto-pilot loop |
| `supervisor.py` | Оркестратор: сканер → ротация → PM2 |
| `connector.py` | Binance Futures API wrapper |
| `calculator.py` | Позиции, TPV, ребалансировка math |
| `executor.py` | Исполнение ордеров |
| `rank_tickers.py` | Сканер тикеров (Selection Strategy 3.0) |
| `notifier.py` | Telegram уведомления |
| `storage.py` | Атомарные JSON операции |
| `config.json` | Конфигурация системы |
| `state.json` | Состояние сессии |

---

## 📊 Документация (docs/)

| Файл | Описание |
|------|----------|
| `docs/Ticker Rotation Algorithm.md` | **7 mermaid-диаграмм** алгоритма supervisor.py |
| `docs/Supervisor Architecture and Bugs.md` | Баги, guards, параметры |
| `docs/AUTOMATION_LOGIC.md` | Логика автоматизации |
| `docs/CHANGES_SUMMARY.md` | Сводка изменений |
| `docs/TICKER_MIGRATION.md` | Миграция тикеров |
| `docs/backtests_060626.md` | Результаты бэктестов 344 тикеров |
| `docs/Описание_логики_Супервайзера.md` | Описание логики супервайзера (RU) |
| `docs/Данные с Binance.md` | Источники данных Binance |
| `docs/Бэктесты _10_дней.md` | Бэктесты 10 дней |
| `docs/Code Review бота ребалансировщика.md` | Code review |
| `docs/Dialog.md` | Диалог 1 |
| `docs/Dialog_2.md` | Диалог 2 |
| `docs/log.md` | Лог изменений |
| `docs/Рекомендации по документации.md` | Рекомендации по документации |

---

## 🔬 Анализ и отчёты

| Файл | Описание |
|------|----------|
| `logs/supervisor.log` | Лог супервайзера |
| `logs/real_INJUSDT.log` | Лог реальной торговли INJUSDT |
| `logs/paper_*.log` | Логи paper-ботов (30+ тикеров) |
| `backtest_rebalance.py` | Бэктест-движок |
| `equity_visualizer.py` | Визуализация equity |
| `swarm_analyzer.py` | Анализатор роя |
| `health_check.py` | Health check системы |

---

## ⚙️ Оптимизация

| Файл | Описание |
|------|----------|
| `optimize_stops.py` | Оптимизация стопов |
| `optimize_takeprofit_stop.py` | Оптимизация тейк-профита |
| `optimize_threshold.py` | Оптимизация порогов |
| `optimize_trend_guard.py` | Оптимизация trend guard |
| `best_takeprofit_stop.json` | Лучшие параметры тейк-профита |
| `best_trend_guard.json` | Лучшие параметры trend guard |

---

## 🧪 Тесты (tests/)

| Файл | Описание |
|------|----------|
| `tests/test_supervisor.py` | Тесты супервайзера |
| `tests/test_supervisor_rotation_whitelist.py` | Тесты ротации с whitelist |
| `tests/test_supervisor_whitelist.py` | Тесты whitelist |
| `tests/test_supervisor_service.py` | Тесты сервиса супервайзера |
| `tests/test_calculator.py` | Тесты калькулятора |
| `tests/test_connector.py` | Тесты коннектора |
| `tests/test_executor.py` | Тесты исполнителя |
| `tests/test_main.py` | Тесты main.py |
| `tests/test_rebalance_v37.py` | Тесты ребалансировки |
| `tests/test_rebalance_logic_v378.py` | Тесты логики ребалансировки |
| `tests/test_scoring_logic.py` | Тесты скоринга |
| `tests/test_guards_*.py` | Тесты guards (3 файла) |
| `tests/test_swarm_*.py` | Тесты роя (2 файла) |
| `tests/test_siphoning.py` | Тесты сифонирования |
| `tests/test_rank_tickers.py` | Тесты сканера |
| `tests/test_storage.py` | Тесты хранилища |
| `tests/test_notifier.py` | Тесты нотификатора |
| `tests/test_telegram_sender.py` | Тесты Telegram |
| `tests/test_risk_engine.py` | Тесты risk engine |
| `tests/test_pattern.py` | Тесты паттернов |
| `tests/test_exact.py` | Тесты точности |
| `tests/test_fix.py` | Тесты фиксов |
| `tests/test_fetch.py` | Тесты fetch |
| `tests/test_import.py` | Тесты импорта |
| `tests/test_get_bot_efficiency.py` | Тесты эффективности |
| `tests/test_scan_v2.py` | Тесты сканера v2 |
| `tests/test_surplus_first.py` | Тесты surplus |
| `tests/conftest.py` | Фикстуры pytest |

---

## 📁 Состояние (state/)

| Файл | Описание |
|------|----------|
| `config.json` | Конфигурация |
| `paper_state_*.json` | Состояния paper-ботов (30+ тикеров) |
| `paper_shadow_*.json` | Тени paper-ботов |
| `real_state_INJUSDT.json` | Состояние REAL INJUSDT |
| `shadow_state_INJUSDT.json` | Тень REAL INJUSDT |
| `scan_cache.json` | Кэш сканера |

---

## 📜 История (history/)

| Файл | Описание |
|------|----------|
| `history/archive_*_paper_state_*.json` | Архивы paper-состояний |
| `history/archive_*_paper_shadow_*.json` | Архивы paper-теней |
| `history/archive_INJUSDT_*.txt` | Архивы INJUSDT |

> Архивы создаются при ротации тикеров (supervisor.py → reset_bot_state_files)

---

## 🔧 Утилиты

| Файл | Описание |
|------|----------|
| `debug_scan.py` | Отладка сканера |
| `check_json_encoding.py` | Проверка кодировки JSON |
| `check_market_data.py` | Проверка рыночных данных |
| `download_30d.py` | Загрузка 30д данных |
| `full_reset.py` | Полный сброс |
| `phase1_validate.py` | Валидация фазы 1 |
| `run_all_backtests.py` | Запуск всех бэктестов |
| `run_backtest_30d.py` | Бэктест 30 дней |
| `run_debug.py` | Отладка |
| `save_cache.py` | Сохранение кэша |
| `send_monitor_report.py` | Отправка отчёта |
| `update_white_list.py` | Обновление whitelist |
| `visualize_bot_log.py` | Визуализация логов |
| `voice_recognizer.py` | Распознавание голоса |
| `telegram_sender.py` | Отправка в Telegram |
| `aggregator.py` | Агрегатор |
| `supervisor_service.py` | Сервис супервайзера |

---

## 📋 Конфигурация

| Файл | Описание |
|------|----------|
| `config.json` | Основная конфигурация |
| `config.txt` | Текстовая копия конфига |
| `tickers.txt` | Список тикеров |
| `tickers_all.txt` | Все тикеры |
| `tickers_cut.txt` | Отфильтрованные тикеры |
| `requirements.txt` | Python зависимости |

---

## 📖 Справка

| Файл | Описание |
|------|----------|
| `README.md` | Описание проекта |
| `CHANGELOG.md` | История изменений |
| `CLAUDE.md` | Инструкции для AI |
| `progress.md` | Прогресс проекта |

---

## 🗂️ Структура папок

```
futures_portfolio/
├── docs/           # Документация (13 файлов)
├── history/        # Архивы состояний (200+ файлов)
├── logs/           # Логи (35+ файлов)
├── tests/          # Тесты (25+ файлов)
├── dashboard/      # Дашборд (5 файлов)
└── *.py            # Основной код (20+ файлов)
```

---

## 🔗 Быстрые ссылки

- [[Ticker Rotation Algorithm]] — диаграммы алгоритма
- [[Supervisor Architecture and Bugs]] — баги и архитектура
- [[INJUSDT Real vs Backtest]] — анализ INJUSDT
- [[Backtest Analysis 060626]] — бэктесты 344 тикеров
- [[prosperous-bot]] — главная заметка проекта

---

## 📊 Текущий статус

| Параметр | Значение |
|----------|----------|
| Режим | REAL |
| LIVE тикеры | INJUSDT |
| Paper боты | 19 |
| Леверидж | x7 |
| Стратегия | 50/50 LONG/SHORT hedge |
| Trailing Stop | ОТКЛЮЧЁН |
| Последнее обновление | 2026-07-17 |
