# RESTRUCTURING — Этап 3: Модульная структура

> Дата: 2026-07-14
> Статус: Завершён
> Автор: Hermes Agent

---

## Зачем

1. **Именование модулей.** `from connector import Connector` — непонятно откуда. `from futures_portfolio.core.connector import Connector` — сразу видно: ядро, API-обёртка.

2. **Конфликт имён.** Windows не различает регистр. `supervisor.py` и `supervisor/` — одно и то же. Переименование в `swarm_manager.py` решило это.

3. **Теневой `src/`.** `Prosperous_Bot/.venv/src/prosperous-bot/src/` перехватывал `import src.storage`. Переименование в `core/` устранило.

4. **Подготовка к новому репозиторию.** `futures-portfolio` на GitHub требует чистой структуры. Плоский корень — legacy.

---

## Структура

```
futures_portfolio/
├── core/                        — Ядро системы
│   ├── main.py                  — Entry point, 24h auto-pilot loop
│   ├── connector.py             — Binance Futures API wrapper (aiohttp)
│   ├── calculator.py            — Position sizing & rebalancing math
│   ├── executor.py              — Order execution
│   ├── storage.py               — JSON state persistence
│   └── notifier.py              — Notification system
├── supervisor/                  — Supervision layer
│   ├── swarm_manager.py         — Swarm manager (was supervisor.py)
│   ├── supervisor_service.py    — PM2 service wrapper
│   ├── aggregator.py            — Portfolio aggregation
│   └── swarm_analyzer.py        — Swarm performance analysis
├── scanner/                     — Ticker selection
│   └── rank_tickers.py          — Selection Strategy 3.0
├── optimization/                — Optuna optimizers
│   ├── optimize_threshold.py
│   ├── optimize_trailing_stop.py
│   ├── optimize_trend_guard.py
│   ├── optimize_config.py
│   ├── optimize_stops.py
│   └── optimize_takeprofit_stop.py
├── backtest/                    — Backtesting
│   ├── backtest_rebalance.py
│   ├── run_all_backtests.py
│   └── run_backtest_30d.py
├── monitoring/                  — Monitoring & reporting
│   ├── telegram_sender.py
│   ├── health_check.py
│   ├── send_monitor_report.py
│   └── equity_visualizer.py
├── tools/                       — Utilities
│   ├── debug_scan.py
│   ├── check_market_data.py
│   ├── download_30d.py
│   ├── save_cache.py
│   ├── visualize_bot_log.py
│   └── voice_recognizer.py
├── scripts/                     — BAT scripts
├── tests/                       — 24 test files
├── config.json                  — SSOT конфигурация
├── ecosystem.config.js          — PM2 конфигурация
└── pyproject.toml               — Dependencies & pytest config
```

---

## Переименования

| Было | Стало | Причина |
|------|-------|---------|
| `supervisor.py` | `supervisor/swarm_manager.py` | Конфликт имён на Windows |
| `src/` | `core/` | Теневой `src/` в `.venv` |

---

## Импорты

Все импорты переведены на пакетный стиль:

```python
# Было:
from connector import Connector
from supervisor import manage_swarm
from storage import safe_load_json

# Стало:
from futures_portfolio.core.connector import Connector
from futures_portfolio.supervisor.swarm_manager import manage_swarm
from futures_portfolio.core.storage import safe_load_json
```

Удалены `sys.path.insert` хаки из production-кода.

---

## PM2 конфигурация

### ecosystem.config.js

```javascript
{
  name: "supervisor-service",
  script: "supervisor/supervisor_service.py",  // было supervisor_service.py
  // ...
},
{
  name: "swarm-aggregator",
  script: "supervisor/aggregator.py",          // было aggregator.py
  // ...
},
{
  name: "telegram-sender",
  script: "monitoring/telegram_sender.py",     // было telegram_sender.py
  // ...
}
```

### Запуск ботов

`swarm_manager.py` запускает ботов через:
```bash
pm2 start core/main.py --name "paper-BTCUSDT" --cwd "C:\Python\Prosperous_Bot\futures_portfolio" ...
```

Было: `pm2 start main.py`

---

## Исправления путей

- `BASE_PATH` в `swarm_manager.py`: `Path(__file__).resolve().parent.parent` (было `.parent`)
- `PROJECT_DIR` в `optimization/*.py`: `Path(__file__).parent.parent` (было `.parent`)
- `data_dir` в `backtest/*.py`: `os.path.dirname(os.path.dirname(...))` (было `.dirname(...)`)
- `dashboard/dashboard.py`: `"core/main.py"` в PM2 команде

---

## Тесты

- **327 passed**, 0 failed, 0 errors
- 24 тестовых файла
- Все import paths обновлены на `futures_portfolio.core.*`, `futures_portfolio.supervisor.*` и т.д.
- Mock paths обновлены: `futures_portfolio.core.connector` вместо `connector`

---

## Что НЕ изменилось

- `config.json` — SSOT, не тронут
- Логика бизнес-процессов — та же
- API ключи — через env vars
- State-файлы (`real_state_*.json`, `paper_state_*.json`) — на месте
- Все guards, trailing stops, rebalancing — та же логика

---

## Rollback

Если что-то пошло не так после перезапуска:
```bash
pm2 delete all
cd C:\Python\Prosperous_Bot\futures_portfolio
git checkout HEAD -- *.py supervisor/ core/ scanner/ optimization/ backtest/ monitoring/
pm2 start ecosystem.config.js
```

---

## Баг-фиксы после Этапа 3

### Python 3.13 stderr crash
`supervisor_service.py` и `aggregator.py` перехватывали `sys.stderr/stdout` на module level.
Это ломало сборку pytest — `lost sys.stderr` при импорте модулей.
Fix: перенесено в `if __name__ == "__main__"`.

### Mock path mismatch (test_aggregator.py)
pytest создаёт два отдельных module object: `supervisor.aggregator` и `futures_portfolio.supervisor.aggregator`.
Mock патчил `futures_portfolio.supervisor.aggregator.safe_load_json`, но класс импортировался из `supervisor.aggregator`.
Fix: все patch paths изменены на `supervisor.aggregator.*`.

### B1 тесты: tracking_open не перехватывал READ
`tracking_open` mock перехватывал только WRITE операции на `real_state_*` файлах.
READ падал на `original_builtin_open` → `FileNotFoundError` → B1 ветка недостижима.
Fix: `tracking_open` перехватывает READ и возвращает `StringIO(json.dumps(state_data))`.

### pm2 start path (test_supervisor_new.py)
`start_bot` в `swarm_manager.py` обновлён на `core/main.py`, тест не обновлён.
Fix: assertion обновлён на `"pm2 start core/main.py"`.

### Итог: 327/327 passed
## Связанные файлы

- `.hermes/plans/2026-07-13_multi-repo-split-plan.md` — полный план реорганизации
- `CHANGELOG.md` — история изменений
- `REPOSITORY_INVENTORY.md` — инвентарь репозитория (от Jules)
