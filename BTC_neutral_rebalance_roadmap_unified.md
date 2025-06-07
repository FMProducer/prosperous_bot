# BTC‑Neutral Rebalance Module — Unified Road‑map (v3, 2025‑06‑02)

*(Trading AI Agent | «режим деньги нейросеть» — цель: +3 M USDT ≤ 7 мес, Max DD < 20 %)*

---

## 0 — Текущее состояние

| Метрика | Значение | Источник |
| :---- | :---- | :---- |
| Unit‑tests | **40 / 40 PASS** | pytest log |
| Покрытие | ≥ 90 % (line) | `pytest‑cov` |
| Комиссия / NAV | ≈ 0.28 % сутки | summary.csv |
| Sharpe (30 дней) | 1.34 | summary.csv |
| Max DD | 13.7 % | summary.csv |
| Deprecation warnings | 3 (pytest_asyncio, urllib3, pandas) | CI log |

---

## 1 — Цели следующего цикла

1. **P0 — устранить Deprecation‑warnings** (pytest_asyncio scope, urllib3, pandas `fillna(method)`).
2. **P1 — CI‑покрытие ≥ 90 %, badge в README**.
3. **P2 — добавление отчётов `summary.csv`, `equity.html`, `blocked_trades_log.csv` в CI artefacts**.
4. **P3 — фильтрация повторяющихся INFO‑логов; перевод «шумных» сообщений в DEBUG.**

---

## 2 — Архитектура модулей

| Модуль | Назначение | Ключевые методы |
| :---- | :---- | :---- |
| `exchange_gate.py` | REST/WS Gate (spot + UM‑futures) | `create_futures_order(order_type)`, `get_price()` |
| `portfolio_manager.py` | NAV и ноциональные веса | `get_value_distribution_usdt()` |
| `rebalance_engine.py` | Ребаланс + safe‑mode | `build_orders()`, `execute()` |
| `rebalance_backtester.py` | Backtest на CSV 5 m + signals | `run_backtest()` |
| `policy_layer.py` | Runtime‑политики (секреты, ресурсы) | `enforce()` |
| `tests/*` | Unit + property‑based | pytest + hypothesis |

---

## 3 — Алгоритм Rebalance Engine (актуальная логика)

1. **Цены**: `P_ref` = Binance spot, 5‑мин свеча.
2. **Веса**: `w_cur` = abs(size)·P_contract / NAV.
3. **Порог**: `thr = max(0.5 %, 0.2·ATR_24h)`; минимум 60 мин между ребалансами.
4. **Signal gating** (`apply_signal_logic = true`):
   * `BUY` → запрет SELL для `BTC_SPOT`, `BTC_PERP_LONG`, `BTC_PERP_SHORT`.
   * `SELL` → запрет BUY для `BTC_PERP_SHORT`.
5. **Ордер‑мэппинг**:
   ```
   BTC_PERP_LONG   : OPEN_LONG / CLOSE_LONG
   BTC_PERP_SHORT  : OPEN_SHORT / CLOSE_SHORT
   ```
6. Batch‑ордеры: PostOnly → 5 s timeout → Market.
7. Лог: детали сделки + блокировки сигналами.

---

## 4 — CI / CD Pipeline

| Этап | Действие | Fail‑criteria |
| :--- | :--- | :--- |
| **Build** | `pip install -r requirements.txt` | dependency errors |
| **Test** | `pytest -q --cov=adaptive_agent --cov-fail-under=90` | cover < 90 % |
| **Static** | `ruff check .` | lint errors |
| **Artifacts** | Upload `reports/*/summary.csv`, `equity.html`, `blocked_trades_log.csv` | missing artefacts |
| **Docker** | Build `trading-ai-agent:latest` | non‑zero exit |

---

## 5 — Policy Layer (v2)

* **Secrets**: в env‑vars, чтение через `get_secret("GATE_API_KEY")`.
* **Resources**: CPU ≤ 50 %, RAM ≤ 2 GB (psutil watchdog).
* **Side‑effects**: блок сетевых вызовов вне white‑list доменов в test‑mode.

---

## 6 — Monitoring & Alerting

* **InfluxDB + Grafana**: NAV, δ‑neutral, Sharpe, Basis\_Gate.
* **Telegram bot**: circuit‑breaker, funding > 5 bps/8 h, used‑margin > 70 % NAV.
* **Daily digest**: ROI, turnover, top‑3 costly trades.

---

## 7 — Timeline (ETA)

| Дата | Задача | Ответственный |
| :---- | :---- | :---- |
| 03 июн | P0 warnings fixed | DevOps |
| 04 июн | P1 coverage badge | QA |
| 05 июн | P2 artefacts в CI | DevOps |
| 07 июн | P3 лог‑фильтры | Core dev |
| 10 июн | Docker‑hardening, staging rollout | Core dev + DevOps |
| ⩓ | Производственный запуск | Stakeholders |

---

## 8 — Ежедневный аудит

1. Сравни фактические KPI (Sharpe, DD, комиссии) с этим роад‑мапом.
2. При отклонении > 5 % — автоматически предложить обновление и логировать в `Log.xlsx`.

---

*Документ объединён из `BTC_neutral_rebalance_roadmap.md` и `BTC_neutral_rebalance_roadmap_v2.md`; обновлен 2025‑06‑02.*

---

## 🔍 Аудит Backtester (2025-06-07): выявленные недочёты и критические отклонения

### Ошибки в `rebalance_backtester.py` и `simulate_rebalance()`

1. **Отсутствие моделирования удержания позиции между свечами**:
   - Каждая операция BUY/SELL эмулируется как мгновенная, без кумуляции позиции.
   - Это делает расчёт прибыли некорректным и недостоверным.
2. **Нулевая валовая прибыль (`pnl_gross_quote == 0`)**:
   - Причина: `entry_price == exit_price`, даже если цена на рынке изменилась.
   - Нет использования `price_close` или `price_change` между входом и выходом.
3. **Сделки на SPOT моделируются как маржинальные с плечом**:
   - Ошибка в логике `portfolio_manager.get_value_distribution_usdt`.
4. **Полное отсутствие учёта направления позиции (long/short)**:
   - `simulate_rebalance` не использует накопленные позиции для расчёта `pnl`.
5. **При `apply_signal_logic = false` поведение становится псевдослучайным**:
   - Отсутствует логика удержания текущей позиции в портфеле между ребалансами.
   - Эмуляция строится на пустом портфеле, а не на продолжении удержания.

---

## 🧩 План доработки модуля Backtester (уточнение этапа 4 Roadmap)

**Цель:** обеспечить реалистичную симуляцию PnL, включая удержание позиций, расчёт NAV и корректную фиксацию прибыли/убытков.

### Этапы:

1. **Внедрение модели `PositionTracker` (внутри `simulate_rebalance`)**:
   - Хранит: `entry_price`, `entry_time`, `side`, `size`, `asset_key`.
   - Поддерживает: кумуляцию и закрытие позиций, фиксацию `pnl`.

2. **Переписать `simulate_rebalance()` с логикой удержания**:
   - Новые позиции открываются только при значительном отклонении от цели.
   - Закрытие и фиксация прибыли/убытка происходит при встречной ребалансировке.

3. **Расчёт `pnl_gross_quote`**:
   ```
   pnl = (exit_price - entry_price) * size * direction * leverage
   ```
   где `direction = 1` для LONG и `-1` для SHORT.

4. **SPOT-позиции рассчитываются без плеча и без `margin`**.

5. **Добавить метаданные в `trade_log`**:
   - `position_id`, `side`, `duration`, `cumulative_pnl`, `drawdown`.

6. **Валидировать `gross_pnl_quote` ≠ 0 в CI-тестах**:
   - Добавить юнит/интеграционные тесты для `simulate_rebalance()`.
   - Проверка: `assert abs(pnl_gross_quote) > commission_quote`.

7. **Сохранять json-логи `open_positions.json`, `executed_trades.json` для анализа**.

8. **Обновить `Manual.docx` и Roadmap соответствующим разделом:**
   - «Симуляция удержания позиций и моделирование прибыли».

---