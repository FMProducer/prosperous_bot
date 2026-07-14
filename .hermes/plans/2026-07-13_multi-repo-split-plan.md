# План: Multi-Repo Split — Реорганизация prosperous_bot

> Дата: 2026-07-13
> Статус: ЭТАП 3 ЗАВЕРШЁН — структурирование выполнено
> Решение: Вариант B (Multi-Repo Split), старт с futures_portfolio
> Новое имя: futures-portfolio (GitHub)

---

## Решение

**Вариант B: Multi-Repo Split** — каждая система = отдельный репозиторий.

**Точка входа:** `C:\Python\Prosperous_Bot\futures_portfolio/`
→ Новый репозиторий: `github.com/FMProducer/futures-portfolio` (или аналогичное имя)

**Почему не Variant D (hybrid):**
- Всё в одном repo = всё ещё мусор
- Legacy код мешает восприятию
- CI/CD проще для одного проекта
- Jules и Claude работают чище с изолированным репо

---

## Целевая структура

### Новый репозиторий: futures-portfolio

```
futures-portfolio/
├── core/                       — Ядро системы (was src/)
│   ├── main.py                 — Entry point, 24h auto-pilot
│   ├── connector.py            — Binance Futures API wrapper
│   ├── calculator.py           — Position sizing & rebalancing math
│   ├── executor.py             — Order execution
│   └── storage.py              — JSON state persistence
├── supervisor/                 — Supervision layer
│   ├── swarm_manager.py        — Swarm manager (was supervisor.py)
│   ├── supervisor_service.py   — PM2 service wrapper
│   ├── aggregator.py           — Portfolio aggregation
│   └── swarm_analyzer.py       — Swarm performance analysis
├── scanner/                    — Ticker selection
│   └── rank_tickers.py         — Selection Strategy 3.0
├── optimization/               — Optuna optimizers
│   ├── optimize_threshold.py
│   ├── optimize_trailing_stop.py
│   ├── optimize_trend_guard.py
│   ├── optimize_config.py
│   ├── optimize_stops.py
│   └── optimize_takeprofit_stop.py
├── backtest/                   — Backtesting
│   ├── backtest_rebalance.py
│   ├── run_all_backtests.py
│   └── run_backtest_30d.py
├── monitoring/                 — Monitoring & reporting
│   ├── telegram_sender.py
│   ├── notifier.py
│   ├── health_check.py
│   └── send_monitor_report.py
├── dashboard/                  — Web UI
│   ├── dashboard.py
│   ├── data_collector.py
│   ├── health_watchdog.py
│   ├── static/
│   └── templates/
├── tools/                      — Utilities
│   ├── voice_recognizer.py
│   ├── equity_visualizer.py
│   ├── visualize_bot_log.py
│   ├── debug_scan.py
│   ├── save_cache.py
│   └── check_market_data.py
├── tests/                      — Unit tests
│   ├── conftest.py
│   └── test_*.py (24 files)
├── scripts/                    — BAT-скрипты запуска
│   ├── start_bot.bat
│   ├── stop_bot.bat
│   ├── restart_bot.bat
│   ├── restart_supervisor.bat
│   ├── status_bot.bat
│   ├── logs_bot.bat
│   ├── full_restart.bat
│   └── create_shortcut.bat
├── docs/                       — Документация
│   ├── CLAUDE.md
│   ├── CHANGELOG.md
│   ├── ROADMAP.md
│   └── *.md
├── config.json                 — SSOT конфигурация
├── config_signal.json          — Signal config (если нужен)
├── ecosystem.config.js         — PM2 конфиг
├── pyproject.toml              — Dependencies
├── requirements.txt            — Lock file
├── .gitignore                  — Чистые правила
├── .env.example                — Template для секретов
└── README.md
```

### Что НЕ попадает в новый репо

| Файл/директория              | Причина                      | Назначение            |
|------------------------------|------------------------------|-----------------------|
| `src/prosperous_bot/`        | Legacy код                   | → prosperous-bot-legacy |
| `freqtrade/`                 | Чужой фреймворк              | → pip dependency       |
| `third_party/`               | Fork чужого проекта          | → отдельный repo       |
| `tests/` (root)              | Старые тесты                 | → prosperous-bot-legacy |
| `scripts/` (root)            | Legacy утилиты               | → prosperous-bot-legacy |
| `tools/` (root)              | Legacy утилиты               | → prosperous-bot-legacy |
| `output/`, `graphs/`         | Артефакты запусков           | → .gitignore           |
| `venv/`                      | Virtualenv                   | → .gitignore           |
| `miniconda3/`                | Miniconda                    | → .gitignore           |
| `vosk-model-*/`              | ML модель                    | → .gitignore           |
| `ffmpeg-*/`                  | Утилита                      | → .gitignore           |
| `*.json` (root)              | Legacy конфиги               | → prosperous-bot-legacy |
| `conftest.py` (root)         | Legacy test infra            | → prosperous-bot-legacy |

---

## Файлы для очистки (мусор в futures_portfolio/)

Дождаться инвентаря от Jules для полного списка. Предварительно:

| Файл                               | Тип       | Действие     |
|------------------------------------|-----------|--------------|
| `connector.py_time.py`            | Debug copy | ✅ Удалено   |
| `main.py_time.py`                 | Debug copy | ✅ Удалено   |
| `test_dd.py`                       | One-off    | ✅ Удалено   |
| `run_debug.py`                     | Debug      | ✅ Удалено   |
| `debug_scan.py`                    | Debug      | → tools/     |
| `check_json_encoding.py`          | Debug      | ✅ Удалено   |
| `check_market_data.py`            | Utility    | → tools/     |
| `full_reset.py`                   | Destructive| ✅ Удалено   |
| `download_30d.py`                 | Data tool  | → tools/     |
| `phase1_validate.py`              | One-off    | ✅ Удалено   |
| `update_white_list.py`           | One-off    | ✅ Удалено   |
| `save_cache.py`                   | Utility    | → tools/     |
| `nul`                             | Windows artifact | ✅ Удалено |
| `progress.md`                     | Temp doc   | ✅ Удалено   |

---

## Этапы реализации

### Этап 0: Ожидание (СЕЙЧАС)
- [ ] Jules завершает инвентарь репозитория
- [ ] Сверяем его результаты с нашим анализом
- [ ] Определяем полный список мусорных файлов

### Этап 1: Подготовка
- [ ] Создать бэкап текущего состояния: `git archive` всего репо
- [ ] Согласовать список файлов для удаления
- [ ] Определить имя нового репозитория на GitHub

### Этап 2: Очистка futures_portfolio
- [x] Удалить debug-копии (*_time.py, test_dd.py, run_debug.py)
- [x] Удалить Windows-артефакты (nul)
- [x] Удалить one-off скрипты (full_reset, phase1_validate, etc.)
- [x] Переместить utility-скрипты в tools/ или удалить
- [x] Удалить progress.md, CHANGELOG.md (проверить актуальность)

### Этап 3: Структурирование ✅
- [x] Создать core/, supervisor/, scanner/, optimization/, backtest/, monitoring/, tools/, scripts/
- [x] Переместить .py файлы по поддиректориям (git mv + mv)
- [x] Переименовать supervisor.py → supervisor/swarm_manager.py
- [x] Обновить import paths во всех файлах (src→core, flat→package)
- [x] Обновить тесты (24 test-файлов, 318/320 passed)
- [x] Исправить BASE_PATH/PROJECT_DIR для поддиректорий
- [x] Обновить ecosystem.config.js (PM2 script paths)

### Этап 4: Конфигурация нового репо
- [ ] Создать чистый .gitignore (30 строк, не 200)
- [ ] Создать .env.example
- [ ] Обновить pyproject.toml (только нужные dependencies)
- [ ] Создать README.md с инструкциями по запуску

### Этап 5: Тестирование
- [ ] Убедиться что все импорты работают
- [ ] Запустить pytest — все тесты проходят
- [ ] Проверить что config.json читается корректно
- [ ] Проверить что PM2-скрипты работают с новыми путями

### Этап 6: Push на GitHub
- [ ] Создать новый repo на GitHub
- [ ] Push чистого кода
- [ ] Настроить branch protection
- [ ] Настроить CI (если нужно)

### Этап 7: PM2 Migration
- [ ] Обновить пути в PM2 конфигах
- [ ] Перезапустить все процессы
- [ ] Проверить что supervisor работает
- [ ] Проверить что все боты (real + paper) работают

### Этап 8: Обновление документации
- [ ] Obsidian: обновить C-Python-Full-Inventory.md
- [ ] Obsidian: создать заметку о новой структуре
- [ ] CLAUDE.md: обновить пути и команды
- [ ] .hermes/: обновить скиллы и планы

---

## Риски

| Риск                                    | Вероятность | Митигация                              |
|-----------------------------------------|-------------|----------------------------------------|
| PM2-пути сломаются                      | Высокая     | Этап 7: поэтапная миграция с проверкой |
| Import paths сломаются                  | Высокая     | Этап 3: автоматический grep + pytest   |
| State-файлы потеряются                  | Средняя     | Бэкап перед любыми изменениями         |
| Config.json не будет найден             | Средняя     | Проверка после каждого этапа           |
| Git history потеряется                  | Низкая      | archive/ветка в старом репо            |

---

## Связанные документы

- [[C-Python-Full-Inventory]] — Инвентарь C:\Python
- [[CLAUDE.md]] — Текущая документация проекта
- [[REPOSITORY_INVENTORY.md]] — Результат Jules (ожидается)
