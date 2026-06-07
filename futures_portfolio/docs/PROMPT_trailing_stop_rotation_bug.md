# Промпт для анализа проблемы ротации после трейлинг стопа

## Задача

Проанализировать и найти фикс для проблемы: после срабатывания trailing stop на прибыльном боте, тикер НЕ удаляется из `live_swarm` и НЕ попадает в `toxic_blacklist`. Бот продолжает работать бесконечно, блокируя ротацию.

## Контекст системы

**Проект:** `C:\Python\Prosperous_Bot\futures_portfolio\`

**Ключевые файлы:**

| Файл | Назначение |
|------|------------|
| `main.py` | Entry point бота, 24h loop, trailing stop logic |
| `supervisor.py` | Оркестратор: сканер → ротация → PM2 |
| `config.json` | Конфигурация (live_swarm, toxic_blacklist) |
| `rank_tickers.py` | Сканер тикеров |

**Параметры конфига:**
```json
{
  "toxic_cooldown_days": 0.02,
  "equity_trailing_stop_pct": 5.7,
  "equity_trailing_stop_activation_pct": 6.5,
  "equity_trailing_stop_timeout_sec": 60,
  "max_bots": 20,
  "paper_mode_bots": 19,
  "min_cycles_for_rank": 60,
  "probation_period_days": 0.01
}
```

## Описание проблемы

### Хронология событий (на примере INJUSDT):

**Шаг 1:** `main.py` — trailing stop срабатывает (drawdown 5.84% от ATH, timeout 60s)

**Шаг 2:** `main.py:941` — `_update_final_metrics_for_exit()` обновляет state, обнуляет trailing_stop-поля

**Шаг 3:** `main.py:954-983` — проверка `tpv_total < global_initial`:
- ЕСЛИ убыток → `emit_signal("stop", ticker)` → создаёт `stop_INJUSDT.flag`
- ЕСЛИ прибыль → `emit_signal("exit", ticker)` → создаёт `exit_INJUSDT.flag` + пишет в `config.json` напрямую

**Шаг 4:** `supervisor.py:462-484` — чтение signal-флагов:
```python
for flag_file in signals_dir.glob("stop_*.flag"):  # ← exit-флаг НЕ найден!
    toxic_blacklist[ticker] = expiry

# exit-флаг просто удаляется без записи в toxic
for flag_file in signals_dir.glob("exit_*.flag"):
    flag_file.unlink()
```

**Шаг 5:** `supervisor.py:812` — `safe_save_json(CONFIG_PATH, config)` — перезаписывает весь config.json, стирая запись toxic от main.py

**Шаг 6:** `supervisor.py:85-156` — `enforce_swarm_consistency()`:
- Обнаруживает позицию на бирже + валидный state → **HEAL TRIGGERED**
- Перезапускает real-бота из сохранённого state

**Шаг 7:** `supervisor.py:285-310` — `calculate_bot_score()`:
- `is_in_drawdown = True` (last_profit < 0 после HEAL) → `score = INF`
- Profit Guard: `profit > 0` → бот не заменяется
- Drawdown Protection: `score == INF` → LOCKED IN COMBAT

**Результат:** INJUSDT вечно в `live_swarm`, блокирует слот, ротация не работает.

## Root Cause Analysis

### Проблема 1: exit-флаг не создаёт toxic-запись в supervisor

**Файл:** `supervisor.py:464`
```python
for flag_file in signals_dir.glob("stop_*.flag"):  # ← только stop!
```

**Должно быть:** supervisor должен обрабатывать И `stop_*.flag` И `exit_*.flag` как toxic-записи.

### Проблема 2: Race condition config.json

**Файл:** `main.py:963-982` и `supervisor.py:812`

main.py пишет toxic в config.json напрямую, но supervisor перезаписывает config.json своим состоянием.

**Варианты решения:**
1. Убрать запись toxic из main.py, оставить только флаги
2. Синхронизировать запись через lock-файл
3. Читать config.json заново перед записью в supervisor

### Проблема 3: HEAL перезапускает бот после trailing stop

**Файл:** `supervisor.py:85-156`

`enforce_swarm_consistency()` не проверяет, был ли trailing stop. Если state валиден (rebalance_cycles > 0 или virt_qty > 0) — HEAL перезапускает бота.

**Решение:** Проверять `trailing_stop_triggered` в state перед HEAL.

### Проблема 4: Profit Guard + Drawdown Protection = вечный бот

**Файл:** `supervisor.py:710-712`
```python
if profit > 0.0:
    continue  # Profit Guard — не заменять
```

После HEAL бот может иметь profit > 0 (из предыдущего стейта), что блокирует замену.

## Требования к фиксу

1. **Trailing stop → toxic_blacklist:** После trailing stop тикер ДОЛЖЕН попадать в toxic_blacklist на период `toxic_cooldown_days`
2. **HEAL после trailing stop:** Бот НЕ ДОЛЖЕН перезапускаться через HEAL если `trailing_stop_triggered == True`
3. **Exit-флаг:** Supervisor ДОЛЖЕН обрабатывать `exit_*.flag` как toxic-запись
4. **Config race:** Нет race-condition между main.py и supervisor.py при записи config.json

## Ожидаемый результат

После фикса:
1. Trailing stop срабатывает → позиции закрываются
2. Тикер попадает в toxic_blacklist на cooldown
3. Supervisor НЕ перезапускает бот через HEAL
4. Бот удаляется из live_swarm
5. Слот освобождается для нового чемпиона
