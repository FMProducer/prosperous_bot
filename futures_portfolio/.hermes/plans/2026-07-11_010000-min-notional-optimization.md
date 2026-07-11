# Plan: Per-ticker min_notional из exchange info + снижение global до 5.1

**Дата:** 2026-07-11
**Состояние:** ✅ IMPLEMENTED (11 Jul 2026)
**Приоритет:** Medium — unlocks +22% ребалансов для GRASS/VVV без LOT_SIZE ошибок

---

## Goal

1. Снизить `min_notional_usdt` с 6.1 до 5.1 (config.json)
2. Добавить per-ticker fallback из `exchange_info` → `MIN_NOTIONAL` filter
3. Для каждого тикера использовать `max(config_min, exchange_info_min)`
4. Результат: GRASS/VVV ловят от 3.7%, YFI/UNI корректно используют 7.0 без ошибок

---

## Current Context

### Проблема
Текущий `min_notional_usdt: 6.1` блокирует GRASSUSDT (Binance min=5.03) и VVVUSDT (5.05) — они не могут ребалансировать при 3-4% отклонениях. Config задаёт единый порог для всех тикеров, хотя Binance у каждого свой.

### Ключевой паттерн (УЖЕ ЕСТЬ В КОДЕ)
**main.py:488** — per-ticker step_size из exchange info:
```python
step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
```
Тот же подход для minNotional — 1 строка.

### Действующий поток данных
```
config.json (min_notional_usdt: 6.1)
  → main.py:940 reads it
  → main.py:955 passes to PortfolioCalculator(min_notional=6.1)
  → calculator.py:230 checks: if abs(diff_usdt) < min_notional → skip
  → main.py:1179 second filter: valid_actions = [a for a if abs(diff) >= min_notional]
  → executor.py:74 third check: if abs(qty)*price < min_notional → SKIPPED
```

### Реальные Binance minimums
| Тикер | Binance min | Binance min + 2% | effective_min = max(5.1, buffer) |
|---|---|---|---|
| GRASSUSDT | 5.03 | 5.13 | 5.13 (buffer wins) |
| VVVUSDT | 5.05 | 5.15 | 5.15 (buffer wins) |
| YFIUSDT | 7.00 | 7.14 | 7.14 (buffer wins) |
| UNIUSDT | 7.05 | 7.19 | 7.19 (buffer wins) |

---

## Constraints (от пользователя)
- Атомарные изменения, минимальная инвазивность
- Обязательны unit-тесты ДО деплоя в production
- initial_capital НЕ менять
- SSOT: config.json пишет только supervisor.py

---

## Step-by-step Plan

### Шаг 1: Извлечь per-ticker min_notional из exchange info

**Файл:** `main.py` (строка ~488, после извлечения step_sizes)

Добавить 1 строку:
```python
min_notionals = {s["symbol"]: float(f["minNotional"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"}
```

**Почему безопасно:** Тот же паттерн что `step_sizes` на строке выше. Уже протестирован в production.

### Шаг 2: Вычислить effective_min_notional для текущего тикера

**Файл:** `main.py` (строка ~940, перед созданием PortfolioCalculator)

Было:
```python
active_min_notional = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt", 6.0))
```

Стало:
```python
config_min = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt"))
if config_min is None:
    raise ValueError("min_notional_usdt must be set in config.json")
exchange_min = min_notionals.get(base_ticker, config_min)
active_min_notional = max(config_min, exchange_min * 1.02)  # 2% buffer for dynamic Binance minimums
```

**Логика:** `config_min` — из config.json (`min_notional_usdt: 5.1`). Fallback при отсутствии exchange info = config = 5.1. `max(5.1, exchange_min * 1.02)` — берём больший порог с 2% буфером на динамику Binance minimums.

### Шаг 3: Передать effective_min в executor

**Файл:** `main.py` (места вызова execute_market_order, ~строки 1213+)

В текущем коде `min_notional` передаётся из `active_min_notional` через PortfolioCalculator actions. Проверить что action dict содержит правильный min_notional для данного тикера.

Если action уже содержит `diff_usdt` и проходит фильтр `valid_actions` на строке 1179 — executor получит min_notional из PortfolioCalculator, который уже использует `active_min_notional`. **Дополнительных изменений в executor.py НЕ требуется** — проверка на строке 74 использует тот же `min_notional`, что передан в функцию.

### Шаг 4: Установить min_notional_usdt в config.json

**Файл:** `config.json`

```diff
- "min_notional_usdt": 6.1,
+ "min_notional_usdt": 5.1,
```

5.1 — реальный Binance минимум. Fallback при отсутствии exchange info = config = 5.1.

### Шаг 5: Unit-тесты

**Файл:** `tests/test_main.py` (добавить новые тесты)

#### Тест 1: per-ticker extraction
```python
def test_min_notionals_extracted_from_exchange_info():
    """Проверяем что min_notionals извлекается из exchange_info filters"""
    exchange_info = {
        "symbols": [
            {"symbol": "GRASSUSDT", "filters": [{"filterType": "MIN_NOTIONAL", "minNotional": "5.03"}]},
            {"symbol": "YFIUSDT", "filters": [{"filterType": "MIN_NOTIONAL", "minNotional": "7.00"}]},
        ]
    }
    min_notionals = {s["symbol"]: float(f["minNotional"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"}
    assert min_notionals["GRASSUSDT"] == 5.03
    assert min_notionals["YFIUSDT"] == 7.00
```

#### Тест 2: effective min = max(config, exchange)
```python
def test_effective_min_notional_takes_max():
    """effective_min_notional = max(config_min=5.1, exchange_min * 1.02)"""
    config_min = 5.1
    test_cases = [
        ("GRASSUSDT", 5.03, 5.13),    # 5.03*1.02=5.13 > 5.1 → buffer
        ("VVVUSDT", 5.05, 5.15),      # 5.05*1.02=5.15 > 5.1 → buffer
        ("YFIUSDT", 7.00, 7.14),      # 7.00*1.02=7.14 > 5.1 → buffer
        ("UNKNOWNUSDT", None, 5.1),   # no exchange data → config fallback
    ]
    for ticker, exchange_min, expected in test_cases:
        emin = (exchange_min * 1.02) if exchange_min else config_min
        result = max(config_min, emin)
        assert result == expected, f"{ticker}: expected {expected}, got {result}"
```

#### Тест 3: GRASSUSDT rebalance at 3.7% deviation
```python
def test_grassusdt_rebalances_below_old_threshold():
    """При min_notional=5.13 GRASSUSDT (Binance 5.03*1.02) ребалансирует при 3.7% deviation"""
    notional = 20.0 * 7  # 140 USDT
    config_min = 5.1
    exchange_min = 5.03
    effective_min = max(config_min, exchange_min * 1.02)  # 5.13 (buffer wins)
    
    deviation_3_5pct = notional * 0.035  # 4.90
    deviation_3_7pct = notional * 0.037  # 5.18
    
    assert deviation_3_5pct < effective_min   # 3.5% too small
    assert deviation_3_7pct >= effective_min  # 3.7% passes
```

#### Тест 4: YFIUSDT still blocked at 5% (Binance min = 7.0)
```python
def test_yfiusdt_uses_exchange_min_not_config():
    """YFIUSDT (Binance 7.0*1.02=7.14) не ребалансирует при config_min=5.1"""
    notional = 20.0 * 7  # 140 USDT
    config_min = 5.1
    exchange_min = 7.0
    effective_min = max(config_min, exchange_min * 1.02)  # 7.14
    
    deviation_4pct = notional * 0.04  # 5.6
    deviation_5pct = notional * 0.05  # 7.0
    deviation_5_1pct = notional * 0.051  # 7.14
    
    assert deviation_4pct < effective_min  # 4% blocked by Binance
    assert deviation_5pct < effective_min   # 5% blocked (7.0 < 7.14)
    assert deviation_5_1pct >= effective_min # 5.1% passes (7.14)
```

#### Тест 5: Fallback при отсутствии exchange info = config min_notional_usdt (5.1)
```python
def test_fallback_to_config_when_exchange_info_missing():
    """При отсутствии exchange info используется config min_notional_usdt = 5.1 (без буфера)"""
    config_min = 5.1
    min_notionals = {}  # empty — no exchange data
    base_ticker = "SOMETHINGUSDT"
    
    exchange_min = min_notionals.get(base_ticker, config_min)
    result = max(config_min, exchange_min)  # fallback = config (no buffer when no exchange data)
    assert result == 5.1  # fallback = config value
```

#### Тест 6: Отсутствие min_notional_usdt в config → ошибка
```python
def test_missing_min_notional_in_config_raises():
    """Отсутствие min_notional_usdt в config → ValueError"""
    import pytest
    current_config = {}  # no min_notional_usdt
    portfolio_cfg = {}
    
    config_min = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt"))
    with pytest.raises(ValueError, match="min_notional_usdt must be set"):
        if config_min is None:
            raise ValueError("min_notional_usdt must be set in config.json")
```

### Шаг 6: Запуск тестов

```bash
cd C:\Python\Prosperous_Bot\futures_portfolio
python -m pytest tests/test_main.py -v -k "min_notional"
```

Все тесты должны пройти. Только после этого — деплой.

### Шаг 7: Деплой

1. Применить изменения в `main.py` (строки 488, 940)
2. Изменить `config.json`: `min_notional_usdt: 5.1`
3. Перезапустить supervisor: `pm2 restart supervisor`
4. Мониторить `logs/supervisor.log` — нет ли LOT_SIZE ошибок
5. Мониторить2-3 цикла — GRASS/VVV ребалансируют чаще

### Шаг 8: Документация

- CLAUDE.md: обновить таблицу параметров
- STATUS.md: обновить min_notional
- changelog.md: записать изменение

---

## Files Likely to Change

| Файл | Изменение | Строки | Риск |
|---|---|---|---|
| `main.py` | +1 строка extraction, ~3 строки effective_min | ~488, ~940 | Низкий |
| `config.json` | min_notional_usdt: 6.1→5.1 | 1 строка | Низкий |
| `tests/test_main.py` | +5 unit-тестов | ~50 строк | Нулевой |
| `CLAUDE.md` | Обновить таблицу | Документация | Нулевой |
| `STATUS.md` | Обновить параметры | Документация | Нулевой |
| `Docs/changelog.md` | Записать изменение | Документация | Нулевой |

**executor.py и calculator.py НЕ ТРОГАЕМ** — они уже корректно работают с переданным min_notional.

---

## Risks & Tradeoffs

| Риск | Вероятность | Импакт | Митигация |
|---|---|---|---|
|.exchange info API fail | Низкая | Низкий | fallback на config_min |
| Больше комиссий (+22% rebalances) | Средняя | Низкий | +0.009% к затратам |
| Новые тикеры с unknown min_notional | Низкая | Низкий | fallback на config_min |
| YFI/UNI LOT_SIZE errors | Нет | — | max(config, exchange) предотвращает |

---

## Validation Checklist (post-deploy)

- [ ] `python -m pytest tests/test_main.py -v -k "min_notional"` — all pass
- [ ] `python -m pytest tests/ -v` — full suite pass
- [ ] supervisor.log: нет LOT_SIZE ошибок для GRASS/VVV
- [ ] supervisor.log: YFI/UNI используют min_notional=7.0 (не 5.1)
- [ ] GRASS/VVV: ребалансируют при 3.7-4.5% отклонениях (было только при 5%+)
- [ ] Buffer: effective_min для GRASS = 5.13 (5.03*1.02), не 5.1
- [ ] Мониторинг2-3 часа: нет ложных срабатываний
