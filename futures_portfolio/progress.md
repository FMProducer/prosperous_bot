# Progress Log — Market-Neutral Futures Portfolio Rebalancer

> Формат: **Дата → Что сделали → Почему → Результат**
> Читаемые первые 50 строк = последние 50 строк (свежее сверху).

---

## 2026-07-07 — Supervisor: WinError 2 fix (pm2 .cmd execution on Windows)

### Что сделано
1. **Диагностика ошибки `PM2: [WinError 2] Не удается найти указанный файл`** — повторялась в логах supervisor-service каждые 7 минут при цикле REAPER GUARD
2. **Корневая причина**: На Windows `pm2` — это `.cmd` батник (`C:\Users\svsma\AppData\Roaming\npm\pm2.cmd`), а не `.exe`. Функция `asyncio.create_subprocess_exec` не может запускать `.cmd` файлы напрямую — она работает только с исполняемыми файлами. Поэтому возникала ошибка WinError 2 (файл не найден).
3. **Решение**: Заменил `create_subprocess_exec` на `create_subprocess_shell` в двух функциях `supervisor.py`:
   - `get_pm2_processes()` (стр. 78-94)
   - `get_running_bots_info()` (стр. 54-76)
   - `shell=True` позволяет `cmd.exe` корректно выполнить `.cmd` батник

### Изменённые файлы
| Файл | Изменение |
|------|-----------|
| `supervisor.py` | `get_pm2_processes()`: `create_subprocess_exec` → `create_subprocess_shell` с комментарием |
| `supervisor.py` | `get_running_bots_info()`: добавлен комментарий, уже использовал `shell` |

### Результат
- `get_pm2_processes()` теперь возвращает 28 процессов (было: ошибка → пустой список)
- `get_running_bots_info()` возвращает 25 ботов (было: ошибка → пустой список)
- REAPER GUARD синхронизирует swarm корректно
- Ошибки `WinError 2` в логах исчезли после `pm2 restart supervisor-service`

---

## 2026-07-07 — Supervisor: UnicodeEncodeError fix (Windows cp1251 console)

### Что сделано
1. **Ошибка**: `UnicodeEncodeError: 'charmap' codec can't encode characters` — эмодзи в логах (`🛡️`, `⚔️`, `🚫`) не кодировались в cp1251 (Windows default console encoding). Краш происходит в `StreamHandler(sys.stdout)` при попытке вывести log message.
2. **Решение**: Добавлено принудительное переключение `sys.stdout` на UTF-8 с `errors="replace"`:
   ```python
   if hasattr(sys.stdout, "reconfigure"):
       sys.stdout.reconfigure(encoding="utf-8", errors="replace")
   ```
3. `errors="replace"` — fallback: неподдерживаемые символы заменяются на `?` вместо исключения.

### Изменённые файлы
| Файл | Изменение |
|------|-----------|
| `supervisor.py` | Добавлен `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` перед `setup_logger()` |

### Результат
- Эмодзи в логах (`🛡️`, `⚔️`, `🚫`, `❌`, `⚖️`) выводятся корректно
- Циклы супервайзера проходят без `UnicodeEncodeError`
- Два последовательных цикла (19:04, 19:12) — `Cycle Complete` без ошибок

---

## 2026-07-16 — Dashboard: убийство зомби-процессов + чистка кода

### Что сделано
1. **Диагностика проблемы логина в dashboard** — пароль `Rebalancer0ID` не проходил, хотя bcrypt hash валидный
2. **Обнаружена причина** — 17 зомби-процессов Python от многочисленных `background=true` запусков занимали порт 8080. Новый dashboard не мог занять порт, curl попадал на старый процесс с устаревшим кодом
3. **Убиты зомби** через `wmic process where "ProcessId=..." delete`
4. **Dashboard запущен через PM2** — работает стабильно
5. **Убран debug-мусор** из `dashboard.py` и `login.html` (файловые логи, debug_msg, отладочные flash)

### Урок
- Не использовать `terminal(background=true)` для серверных процессов без `process kill` после тестов
- PM2 предпочтительнее для long-running процессов на Windows
- При отладке Flask на Windows: `taskkill /F /PID` не работает из MSYS (режет `/F`), использовать `wmic`

### Изменённые файлы
| Файл | Изменение |
|------|-----------|
| `dashboard/dashboard.py` | Убран debug-код из login route, возврана чистая версия |
| `dashboard/templates/login.html` | Убран debug-блок |

---

## 2026-06-03 — Масштабная сессия: отладка и подготовка к реальному запуску

### Что сделано

1. **VIRTUAL OFF** — отключена виртуальная нога (share 0.35 → 0.0). Чистый хедж 50/50 LONG/SHORT.

2. **Ротация переписана** — score-driven selection. Прибыльные боты (profit > 0) всегда остаются в рое. Замене подлежат только убыточные.

3. **Net Move Guard** — новый защитный механизм. Если цена >1.5% за 30с — блокировка всех действий.

4. **PnL Guard** — блокировка продажи излишков при отрицательном hedge PnL. Логирование только при смене состояния.

5. **config.json защищён** — supervisor не перезаписывает tickers. Fallback из сканера при пустом списке.

6. **Исправлен бэктестер** — замена available_funds на val_cash. Бэктестер снова работает.

7. **Оптимизация порогов** — surplus 1% / deficit 4% (было 1.5% / 3%).

8. **Изолированная маржа → Кросс-маржа** — в main.py изменено на CROSS. При хедже 50/50 это безопаснее.

9. **Margin ratio пороги** — warning 3.0x, critical 1.5x (было 5.0x / 2.0x).

### Результаты бэктестов

| Тикер | 48ч PnL | MaxDD | Cycles |
|-------|---------|-------|--------|
| HEIUSDT | +0.23% | 0.23% | 10 |
| WLDUSDT | +2.07% | — | — |
| STORJUSDT | +1.95% | — | — |
| FORMUSDT | +1.68% | — | — |

### Результат

**Запущен 1 бот на реальный счёт:**
- HEIUSDT, 180 USDT, x5, кросс-маржа
- Margin ratio: 20.74x
- PnL: +0.23 USDT (11 циклов)

### Известные риски

- Liquidation guard не работает при кросс-марже (Binance не даёт liquidationPrice)
- При экстремальных движениях >50% возможна потеря всего баланса (крайне маловероятно при хедже)

---

## 2026-05-30 — Катастрофа: ликвидация PORTALUSDT SHORT

### Событие
PORTALUSDT SHORT x5 ликвидирована биржей при росте цены +17% за 1ч (0.01034 → 0.01213). Бот не смог восстановить позицию. Supervisor restart убил real-portal PM2 процесс без перезапуска. Средства выведены с аккаунта, реальная торговля отключена.

### Хронология
```
15:00  PORTALUSDT = 0.01034, SHORT маржа ≈ 43.50 USDT
15:00-15:06  Цена растёт до 0.01193, unrealized PnL → -21.93 USDT
15:06  Heartbeat: S:0.0% — ликвидирована биржей
15:11  Rebalance #8 пытается восстановить: BUY 15468.5 LONG + SELL 19202.3 SHORT @ 0.01175
15:11:32  Последняя запись в real_PORTALUSDT.log
15:37  Supervisor service restart → убивает все PM2 процессы (SIGINT)
15:37  real-portal (PID 15616) потерян, НЕ перезапущен
15:37  Cycle Complete. REAL Swarm: [] — оба real-бота мертвы
```

### Анализ: корневые причины

1. **Нет predictive liquidation guard** — только post-factum (margin <= 0). Бот узнаёт о ликвидации постфактум.
2. **Supervisor теряет real-ботов** — при своём рестарте убивает все PM2 процессы через SIGINT, но не перезапускает real-ботов из `live_swarm`.
3. **Рассинхронизация paper/real (КРИТИЧЕСКИЙ БАГ)** — paper mode НЕ симулировал liquidation price. Супервизор отправлял в реальную торговлю тикеры, которые "на бумаге" выглядели безопасно, но в реальных условиях приводили к ликвидации.

### Решение: 3 уровня защиты

**Уровень 1 — connector.py: `get_position_risk()`**
- Для каждой позиции считает: `liq_price`, `distance_pct`, `unrealized_pnl`, `margin`
- Источник: `futures_position_information` + `futures_mark_price` от биржи

**Уровень 2 — main.py: Guard #4 Liquidation Distance Monitor**
- Работает для BOTH paper и real
- Пороги: `liquidation_distance_warn_pct: 15%`, `liquidation_distance_crit_pct: 8%`
- Real: emergency market close позиции через `PortfolioExecutor.execute_market_order()`
- Paper: `_handle_liquidation_guard()` — единая функция, paper симулирует PnL + сброс shadow_state
- При открытии позиции в paper mode рассчитываются `long_liquidation_price` / `short_liquidation_price` по формуле Binance:
  ```
  LONG liq  = entry × (1 - 1/leverage + mmr)   где mmr = 0.004 (0.4%)
  SHORT liq = entry × (1 + 1/leverage - mmr)
  ```
- При закрытии позиции liq price сбрасывается в 0.0

**Уровень 3 — supervisor.py: `_ensure_real_bots_alive()`**
- После каждого swarm цикла проверяет: все ли tickers из `live_swarm` имеют PM2 процессы?
- Если процесс отсутствует → автоперезапуск через `start_bot()`

### Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `connector.py` | `get_position_risk()` — данные о риске ликвидации от биржи |
| `main.py` | Simulated liq price в paper mode, Guard #4 paper+real, `_handle_liquidation_guard()` |
| `supervisor.py` | `_ensure_real_bots_alive()` + liq price в начальный shadow_state |
| `config.json` | `liquidation_distance_warn_pct: 15.0`, `liquidation_distance_crit_pct: 8.0` |
| `backtest_rebalance.py` | Predictive liquidation guard (формула Binance, порог 8%) |
| `CLAUDE.md` | Safety Mechanisms: Guard #3 + Guard #5 |

### Решения по дизайну
- **ISOLATED margin оставлен** — cross margin рискованнее для аккаунта (одна позиция может убить весь аккаунт). Isolated = ликвидация только маржи позиции.
- **Position-level trailing stop НЕ добавляем** (FMProducer: не даём шансов манипуляторам/охотникам за стоплоссами)
- **LIVE TRADING отключена** до полного тестирования новых защит

---

## 2026-05-31 — Оптимизация стоплоссов + запуск Combat

### Что
Optuna-оптимизация параметров стоплосса, 7-дневный бэктест, запуск реальной торговли (Combat).

### Почему
Параметры стоплосса были выбраны эмпирически. Нужна систематическая оптимизация.

### Результат
- **Optuna:** 101 trial, 92 completed, 44с. Оптимальные параметры:
  - `max_drawdown_limit`: 33% → **22%**
  - `equity_trailing_stop_pct`: 10% → **24%**
  - `equity_trailing_stop_activation_pct`: 2% → **7.5%**
  - `equity_trailing_stop_timeout_sec`: 60s (не оптимизировался)
- **7-дневный бэктест (20 тикеров):** 18/20 прибыльных, avg +9.78%, avg DD 7.99%, 0 TS triggers
- **Combat запущен:** live_swarm = [PORTALUSDT, VTHOUSDT], позиции открыты на Binance
- **Incubator (paper):** +14.57 USDT за 8ч (12.7% ROI), 13/30 тикеров прибыльных
- **Проблема:** NFPUSDT -14.56 USDT (28% дамп, TS сработал). Решение: min_cycles_for_rank=10 не пускает в Combat.

---

## 2026-05-31 — Настройка мониторинга

### Что
Создан health_check.py, cron job для автоматического мониторинга.

### Почему
Ручная проверка PM2/логов/state занимает время. Автоматизация = быстрая реакция на проблемы.

### Результат
- `health_check.py` — проверяет PM2, real_state, blacklist, ошибки
- Cron job каждые 15 мин → Telegram отчёт

---

## 2026-05-31 — Рабочие соглашения (OWL ↔ FMProducer)

### Что
Описан workflow взаимодействия, приоритеты, ограничения.

### Результат
- progress.md переписан (дата → что → почему → результат)
- MEMORY обновлён (project-specific контекст вместо общих заметок)
- Skill `prosperous-bot` создан
- CLAUDE.md актуализирован на основе PDF
- experiments.md для структурирования гипотез

---

## 2026-05-30 — Рефакторинг калькулятора

### Что
`calculate_deviations()` теперь возвращает dict вместо list. `calculate_rebalance()` обновлён.

### Почему
Bugs: `'str' object has no attribute 'get'` при итерации по dict keys вместо list.

### Результат
Все backtest движки синхронизированы с calculator.py. `run_all_backtests.py` — 345 тикеров, 0 ошибок.

---

## 2026-05-30 — Voice STT (Hermes Gateway)

### Что
Добавлено распознавание голосовых сообщений в Telegram-боте (Vosk + ffmpeg).

### Результат
STT работает, качество приемлемое для коротких команд.
