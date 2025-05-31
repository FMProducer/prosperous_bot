# BTC‑Neutral Rebalance Module — Integration Road‑map

*(Trading AI Agent | «режим деньги нейросеть»)*

## 1 Цели

- **Поддерживать 0‑дельту по BTC** при распределении **65 % spot / 11 % long x5 / 24 % short x5**.  
- Минимизировать комиссии и basis‑шум, ориентируясь на **спотовую цену Binance**.  
- Обеспечить автоматический ребаланс каждые 30 с при |δw| \> 0,5 %.  
- Интегрировать все процессы в существующий async‑loop Trading AI Agent без Canvas.

## 2 Изменения в конфигурации (`config.json`)

"asset\_distribution": {

  "BTC\_SPOT"   : 0.65,

  "BTC\_LONG5X" : 0.11,

  "BTC\_SHORT5X": 0.24

},

"rebalance\_threshold": 0.005,

"check\_interval"    : 30,

"price\_source"      : "binance\_spot",

"maker\_mode"        : true,

"max\_leverage"      : 5

## 3 Новые файлы / модули

| Модуль | Назначение | Ключевые методы |
| :---- | :---- | :---- |
| **`exchange_gate.py`** | WS/REST обёртка Gate (spot + UM‑futures) | `get_price()`, `post_only_limit()`, `market_order()` |
| **`portfolio_manager.py`** | Счёт NAV, маржи и **ноциональных** весов | `get_notional_weights(px)` |
| **`rebalance_engine.py`** | Алгоритм ребаланса | `build_orders()`, `execute()` |
| **`analytics.py`** | ROI, Sharpe, fees, basis‑спред | `record_metrics()` |
| **`alert_engine.py`** | Telegram / Slack алерты | `send_alert()` |

## 4 Логика работы `RebalanceEngine`

1. Получить **референс‑цену** `P_ref` из Binance WS.  
2. Запросить веса `w_cur = pm.get_notional_weights(P_ref)`.  
3. Если `max(|w_cur − w_target|) > 0.5 %` → построить дельты  
4. Сложить дельты в **batch‑ордер**: PostOnly → 5 с timeout → Market.  
5. Логировать `fees`, `slippage`, `δ‑neutral`.

### Адаптивные улучшения

| Приём | Когда срабатывает |
| :---- | :---- |
| `thr = max(0.5 %, 0.2·ATR_24h)` | волатильность \>,\< базовый уровень |
| Удвоение порога при turnover \> 4·NAV | контроль комиссий |
| Пауза, если | P\_binance − P\_gate |

## 5 Risk Management

- **Used margin \< 70 % NAV**; при превышении — понизить плечо до 3 ×.  
- Funding‑watchdog: если net‑funding \> 5 bps/8h — уменьшить плечо или скорректировать веса.  
- Circuit‑breaker: стоп ребаланса при 1‑мин свече \> 7 %.

## 6 Monitoring & Analytics

| Метрика | Источник | Алерт |
| :---- | :---- | :---- |
| NAV, ROI, Sharpe | `analytics.py` → InfluxDB | daily digest |
| δ‑neutral ( | w\_long − w\_short | ) |
| Basis\_gate | pricefeed | \>0.5 % → pause |

## 7 Testing

1. **Unit‑tests** (`pytest`) для расчёта весов и ордер‑логики.  
2. **Monte‑Carlo 1 000 путей**: GBM ±basis, funding; сравнить PnL/fee со старой схемой.  
3. **Paper‑trade** 7 дней — verify logs.

## 8 Deployment

docker compose up \-d

\# services: trading\_ai, influxdb, grafana

## 9 Milestone Check‑list

| ✔ | Шаг |
| :---- | :---- |
| ☐ `config.json` расширен |  |
| ☐ `exchange_gate.py` готов |  |
| ☐ `portfolio_manager.py` считает notional |  |
| ☐ `rebalance_engine.py` в loop |  |
| ☐ Metrics + Grafana панель |  |
| ☐ Telegram алерты |  |
| ☐ Монте‑Карло тест \> Sharpe 1.3 |  |
| ☐ Paper‑trade 7 дней OK |  |
| ☐ Production rollout 100 % капитала |  |

---

*Документ создан:* 2025-05-17 23:16 UTC  
