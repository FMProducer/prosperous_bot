# Plan: B6 (Race Condition) + B3 (Exit/Stop Propagation)

**Дата**: 2026-07-11
**Статус**: Draft
**Риск**: HIGH — затрагивает live real-боты

---

## Цель

Устранить две архитектурные уязвимости:
1. **B6**: Race condition — `main.py` напрямую пишет в `config.json` (нарушение SSOT)
2. **B3**: `supervisor.py` не различает сигналы `exit` и `stop` → прибыльные тикеры попадают в toxic_blacklist вместо probation

---

## Контекст / Допущения

- SSOT-инвариант уже задокументирован в CLAUDE.md: "Только `supervisor.py` имеет право на запись в `config.json`"
- `emit_signal()` в `main.py` работает корректно — создаёт файлы-флаги `signals/{type}_{mode}_{ticker}.flag`
- `supervisor.py` уже читает сигналы (строки 574-594), но обрабатывает их без различия типа
- `config.json` содержит ключи: `toxic_blacklist_real`, `toxic_blacklist_paper`, `black_list`, `live_swarm`
- Probation-ключей (`probation_real`, `probation_paper`) в текущем config.json НЕТ — нужно добавить

---

## Изменения

### Part 1: B6 — Лишить `main.py` прав на запись в `config.json`

**Файл: `main.py`**

#### 1a. `_handle_liquidation_recovery` (строки 160-179)

Удалить весь блок `try/except` который читает/пишет `config_path`:
- Строки 160-179: блок `import json as _json; cfg_path = Path(config_path); ... cfg_path.write_text(...)` → **УДАЛИТЬ**
- Оставить ТОЛЬКО `emit_signal("stop", base_ticker, is_paper=False)` (строка 181) и Telegram-уведомление

**Итоговый вид блока** (строки ~160-189):
```python
        # [SSOT] Delegate config mutation to supervisor via signal
        emit_signal("stop", base_ticker, is_paper=False)

        try:
            await notifier.send_alert("🚨 LIQUIDATION",
                f"{base_ticker}: Position liquidated! Bot stopped. Remaining positions closed.")
        except:
            pass

        logger.critical(f"🚨 LIQUIDATION RECOVERY complete for {base_ticker}")
```

#### 1b. `_handle_liquidation_guard` (строки 265-297)

Удалить весь блок `try/except` который читает/пишет `_cfg_path`:
- Строки 267-297: блок `import json as _json; _cfg_path = Path("config.json"); ... _cfg_path.write_text(...)` → **УДАЛИТЬ**
- Оставить `emit_signal("stop", base_ticker, is_paper=is_paper)` и лог

**Итоговый вид блока** (строки ~265-298):
```python
        # [SSOT] Delegate config mutation to supervisor via signal
        emit_signal("stop", base_ticker, is_paper=is_paper)
        logger.critical(
            f"🚫 {base_ticker} signal 'stop' emitted. "
            f"Awaiting supervisor to blacklist and remove from live_swarm."
        )
```

#### 1c. Чистка импортов

Убрать неиспользуемые `import json as _json` и `from pathlib import Path` из обоих блоков (если они больше нигде не используются в функциях).

---

### Part 2: B3 — Разделение логики Exit и Stop в `supervisor.py`

**Файл: `supervisor.py`** (строки 565-594)

Текущий код обрабатывает все сигналы одинаково — добавляет в `toxic_blacklist_*`. Нужно:

1. **Добавить чтение `probation_period_days`** из конфига (отдельный cooldown для exit)
2. **Маршрутизировать по типу сигнала**:
   - `stop` → `toxic_blacklist_{mode}` + `black_list` (перманент)
   - `exit` → `probation_{mode}` (мягкий cooldown, без перманентного бана)

**Новые ключи в config.json**:
- `probation_real: {}` — dict `{ticker: expiry_timestamp}`
- `probation_paper: {}` — dict `{ticker: expiry_timestamp}`
- `probation_period_days: 0.041` (~1 час по умолчанию, как в main.py:992)

**Замена блока信号处理** (строки 574-594):

```python
    # Читаем параметры cooldown
    toxic_cooldown_sec = config.get("toxic_cooldown_days", 0.02) * 86400
    probation_cooldown_sec = config.get("probation_period_days", 0.041) * 86400

    if signals_dir.exists():
        for flag_file in signals_dir.glob("*.flag"):
            parts = flag_file.stem.split("_", 2)
            if len(parts) != 3:
                flag_file.unlink(missing_ok=True)
                continue

            signal_type, mode_tag, ticker = parts

            # 1. Удаляем из live_swarm в любом случае
            if "live_swarm" in config and ticker in config["live_swarm"]:
                config["live_swarm"].remove(ticker)
                logger.info(f"🗑️ {ticker} removed from live_swarm (signal: {signal_type})")

            # 2. Маршрутизация по типу сигнала
            if signal_type == "stop":
                # Токсичный исход → toxic_blacklist + permanent black_list
                bl_key = "toxic_blacklist_paper" if mode_tag == "paper" else "toxic_blacklist_real"
                if bl_key not in config:
                    config[bl_key] = {}
                config[bl_key][ticker] = now_ts + toxic_cooldown_sec

                if "black_list" not in config:
                    config["black_list"] = []
                if ticker not in config["black_list"]:
                    config["black_list"].append(ticker)

                logger.critical(
                    f"🚫 STOP signal: {ticker} ({mode_tag}) → {bl_key} + black_list "
                    f"(expires {time.ctime(now_ts + toxic_cooldown_sec)})"
                )

            elif signal_type == "exit":
                # Прибыльный исход → probation (мягкий cooldown)
                prob_key = "probation_paper" if mode_tag == "paper" else "probation_real"
                if prob_key not in config:
                    config[prob_key] = {}
                config[prob_key][ticker] = now_ts + probation_cooldown_sec

                logger.info(
                    f"✅ EXIT signal: {ticker} ({mode_tag}) → probation "
                    f"(expires {time.ctime(now_ts + probation_cooldown_sec)})"
                )
            else:
                logger.warning(f"⚠️ Unknown signal type '{signal_type}' for {ticker}. Ignoring.")

            # 3. Удаляем обработанный флаг (async — не блокируем event loop)
            try:
                await asyncio.to_thread(flag_file.unlink)
            except OSError:
                pass
```

**Важно**: Текущий код supervisor.py читает `bl_paper`/`bl_real` в начале цикла (строка 571-572). Эти переменные используются позже при фильтрации сканера и инкубатора. При добавлении `probation_*` ключей нужно убедиться, что они тоже обновляются в `config` перед `safe_save_json` в конце цикла (строка 966).

---

### Part 3: Интеграция Probation в логику ротации (включая P3)

**Файл: `supervisor.py`**

#### 3a. P1: Конвертация black_list/live_swarm в set при загрузке

В `manage_swarm()`, сразу после `config = await safe_load_json(CONFIG_PATH, {})`:

```python
    # [P1] O(1) lookup: конвертируем list → set для проверок
    if "black_list" in config and isinstance(config["black_list"], list):
        config["black_list"] = set(config["black_list"])
    if "live_swarm" in config and isinstance(config["live_swarm"], list):
        config["live_swarm"] = set(config["live_swarm"])
```

Перед `safe_save_json(CONFIG_PATH, config)` в конце цикла:

```python
    # [P1] Конвертируем set → sorted list для JSON-сериализации
    if "black_list" in config and isinstance(config["black_list"], set):
        config["black_list"] = sorted(config["black_list"])
    if "live_swarm" in config and isinstance(config["live_swarm"], set):
        config["live_swarm"] = sorted(config["live_swarm"])
```

#### 3b. Probation-фильтрация кандидатов

При формировании `old_incubator` (строка 685) и фильтрации кандидатов на REAL (строка 747), проверяем `probation_*` списки:

```python
    # Фильтрация по probation (мягкий бан — тикер не может стать REAL пока на probation)
    bl_prob_paper = config.get("probation_paper", {})
    bl_prob_real = config.get("probation_real", {})

    # При формировании old_incubator: исключаем probation-paper тикеры
    old_incubator = [t for t in old_incubator if t not in bl_prob_paper]

    # При выборе REAL кандидатов: исключаем probation-real тикеры
    current_real_tickers = [t for t in current_real_tickers if t not in bl_prob_real]
```

#### 3c. P3: Замена trailing_stop_paper_timeout_end на probation_*

В блоке оценки кандидатов (строка 762), заменяем чтение из state-файла на проверку probation в config:

```python
        # [P3] Probation check — единый источник кулдаунов в config
        prob_key = "probation_paper" if not is_running_real else "probation_real"
        prob_dict = config.get(prob_key, {})
        if ticker in prob_dict and time.time() < prob_dict[ticker]:
            logger.info(f"⏳ Skipping {ticker}: In {prob_key} until {time.ctime(prob_dict[ticker])}.")
            continue
```

Вместо текущего:
```python
        # БЫЛО (строка 762):
        if not is_running_real and time.time() < p.get('trailing_stop_paper_timeout_end', 0.0):
            logger.info(f"⏳ Skipping {ticker}: Still in trailing stop paper timeout.")
            continue
```

---

## Файлы для изменения

| Файл | Объём изменений | Описание |
|------|----------------|----------|
| `main.py` | ~30 строк удалить | Удаление блоков записи в config.json из `_handle_liquidation_recovery` и `_handle_liquidation_guard` |
| `supervisor.py` | ~60 строк заменить + ~25 добавить | Переписать обработку сигналов (B3), добавить probation-логику, конвертация black_list/live_swarm в set (P1), async unlink (P2), замена trailing_stop_paper_timeout_end на probation_* (P3) |
| `config.json` | +2 ключа | Добавить `probation_real: {}`, `probation_paper: {}` |
| `CLAUDE.md` | ~10 строк | Обновить Change Log, задокументировать P1-P3 |

---

## Тесты

### Новые тесты (tests/test_supervisor.py)

1. **`test_process_signals_stop_toxic_blacklist`**: Сигнал `stop_real_BTCUSDT.flag` → `toxic_blacklist_real["BTCUSDT"]` установлен + `black_list` содержит BTCUSDT + файл удалён
2. **`test_process_signals_exit_probation`**: Сигнал `exit_paper_ZECUSDT.flag` → `probation_paper["ZECUSDT"]` установлен, `toxic_blacklist_paper` пуст, файл удалён
3. **`test_process_signals_removes_from_live_swarm`**: Любой сигнал → тикер удалён из `live_swarm`
4. **`test_process_signals_unknown_type`**: Сигнал `unknown_real_BTC.flag` → файл удалён, конфиг не изменён, warning залогирован
5. **`test_probation_blocks_real_candidate`**: Тикер в `probation_real` → исключён из `current_real_tickers`
6. **`test_blacklist_set_o1_lookup`**: `black_list` конвертирован в set → проверка `in` работает за O(1), при сохранении обратно в list
7. **`test_flag_unlink_is_async`**: `flag_file.unlink()` заменён на `await asyncio.to_thread(...)` — проверяем что event loop не блокируется

### Новые тесты (tests/test_main.py)

8. **`test_handle_liquidation_recovery_no_config_write`**: `_handle_liquidation_recovery` НЕ пишет в config.json (проверяем что `Path.write_text` не вызывается)
9. **`test_handle_liquidation_guard_no_config_write`**: Аналогично для `_handle_liquidation_guard`

### Верификация

```bash
cd C:\Python\Prosperous_Bot
python -m pytest futures_portfolio/tests/test_supervisor.py futures_portfolio/tests/test_main.py -v
```

---

## Риски / Tradeoffs

| Риск | Severité | Митигация |
|------|----------|-----------|
| Пропущенный stop-сигнал → тикер не забанен | HIGH | `_handle_liquidation_recovery` оставляет `emit_signal("stop")` — если файл не создан, supervisor не обновит config. Проверяем что `emit_signal` надёжна (уже tested). |
| Probation-ключи отсутствуют в текущем config.json | LOW | Supervisor создаёт ключи при первом encounter (`if prob_key not in config`). Backward-compatible. |
| main.py может вызвать `_handle_liquidation_guard` для paper-режима без config-доступа | LOW | `emit_signal` не требует config — только создаёт файл. Paper/Real определяется `is_paper` параметром. |
| Race condition между removal из live_swarm (supervisor) и Invariant Gate | MEDIUM | Invariant Gate проверяет биржевые позиции — если позиция ещё открыта, тикер будет принудительно возвращён в live_swarm. Это корректное поведение. |
| black_list как set может сломать JSON-сериализацию | LOW | Конвертация обратно в sorted list перед `safe_save_json`. Проверяется в test_blacklist_set_o1_lookup. |
| asyncio.to_thread для unlink можетIncrease latency при одном файле | NEGLIGIBLE | Overhead ~0.1ms. Выигрыг при каскадных ликвидациях (10+ файлов) — значительный. |

---

## Production-паттерны (ревью FMProducer)

### P1: O(N) → O(1) lookup для black_list

`black_list` хранится как `list` в config.json. Проверка `if ticker not in config["black_list"]` — O(N). При длительном аптайме список разрастается.

**Решение**: При загрузке конфига в `manage_swarm()` конвертировать `black_list` в `set` для проверок. При сохранении (`safe_save_json`) конвертировать обратно в `list` (JSON-совместимый формат).

```python
# В начале manage_swarm(), после загрузки config:
if "black_list" in config and isinstance(config["black_list"], list):
    config["black_list"] = set(config["black_list"])

# Перед safe_save_json в конце цикла:
if "black_list" in config and isinstance(config["black_list"], set):
    config["black_list"] = sorted(config["black_list"])
```

Аналогично для `live_swarm` — проверки `if ticker in config["live_swarm"]` тоже O(N) на list. Конвертация в set при чтении, обратно в sorted list при записи.

### P2: Блокирующий I/O в event loop — flag_file.unlink()

`Path.unlink()` — синхронный syscall. При каскадных ликвидациях (массовый market dump) может заблокировать event loop на несколько миллисекунд.

**Решение**: Заменить на `asyncio.to_thread(flag_file.unlink)`:

```python
await asyncio.to_thread(flag_file.unlink)
```

`asyncio.to_thread` уже доступен в stdlib (Python 3.9+), не требует额外 зависимостей.

### P3: Probation — единый источник контроля кулдаунов

Текущий `trailing_stop_paper_timeout_end` в state-файле воркера дублирует логику, которая должна жить в супервайзере. Антипаттерн: два источника правды для одного кулдауна.

**Решение**: Supervisor — единственный контроллер probation:
- `probation_real: {}` и `probation_paper: {}` в config.json
- При выборе REAL-кандидатов supervisor проверяет `probation_real` (вместо `trailing_stop_paper_timeout_end` из state-файла)
- State-файл воркера НЕ используется для принятия решений о ротации
- При amnesty (истечение toxic_blacklist) супервайзер сбрасывает `trailing_stop_triggered` в state-файле — это уже реализовано (строки 603-614)

**Удалить из supervisor.py**: Проверку `trailing_stop_paper_timeout_end` из state (строка 762):
```python
# БЫЛО (строка 762):
if not is_running_real and time.time() < p.get('trailing_stop_paper_timeout_end', 0.0):

# СТАЛО:
prob_key = "probation_paper" if ... else "probation_real"
if ticker in config.get(prob_key, {}) and time.time() < config[prob_key][ticker]:
```

---

## Открытые вопросы

**Решены в ходе ревью:**

1. **Probation vs Trailing Stop timeout**: Перенести весь контроль кулдаунов в `probation_*` ключи config.json. State-файл воркера — stateless-совместимый, не участвует в ротации.
2. **black_list при exit**: Exit НЕ добавляет в `black_list`. black_list только для stop (токсичных).
