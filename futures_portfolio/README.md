# Market-Neutral Futures Portfolio Rebalancer

## Назначение

Автоматизированная система управления портфелем USD-M фьючерсов на Binance. Извлекает прибыль из волатильности цены, поддерживая рыночно-нейтральную дельту (≈0) через три ноги портфеля.

## Принципы работы

### 1. Рыночная нейтральность (дельта = 0)

```
Long  27% × 5x = +135% дельта
Short 36% × 5x = -180% дельта
Virtual 35% × 1x = +35% дельта
Итого: +135 - 180 + 35 ≈ 0
```

Virtual — математически точная симуляция владения базовым активом на споте. Не дериватив, не маржинальная позиция. Ребалансируется равноправно с Long и Short.

### 2. Логика ребалансировки

**Золотое правило: ни цента из свободной маржи на аккаунте!**

Каждый бот управляет **только своим начальным капиталом** и **собственной прибылью**.

Порядок действий:
1. **Продажа излишков** — только доли, достигшие порога `surplus`
2. **Покупка дефицитов** — только доли, достигшие порога `deficit`
3. **Дефициты покупаются ТОЛЬКО на proceeds от продажи излишков**
4. Если proceeds не хватает — дефицит покупается частично или пропускается
5. `diff_usdt < min_notional` (7 USDT) — действие пропускается без уведомления

**Исключение:** первый запуск бота (`ignore_limits=True`, позиции пустые) — позиции набираются из стартового капитала.

### 3. TPV (Total Portfolio Value)

```
TPV = Real Equity + (Virtual Quantity × Spot Price)
```

Где Real Equity = свободный кэш + маржинальное обеспечение + нереализованный PnL фьючерсных ног.

- **TPV НЕ падает при ребалансировке** — вся прибыль остаётся внутри бота и реинвестируется
- TPV может падать только из-за движения цены (unrealized PnL) или комиссий

### 4. Стоп-лоссы

Применяются к **TPV всего бота** (все доли вместе), НЕ к отдельным позициям:

- **Trailing Stop** — просадка от ATH TPV ≥ `equity_trailing_stop_pct`
- **Max Drawdown** — TPV < `tpv_ath × (1 - max_drawdown_limit/100)` (от пика, а не от старта)

После срабатывания стопа:
- Все позиции закрываются
- Бот завершается (`return`)
- Бот **НЕ перезапускается самопроизвольно** — решение принимает supervisor

**НЕТ:** hard stop от initial_tpv, per-ticker стопов.

### 5. FUSE (Anti-Churn)

Предотвращает частые ребалансировки вблизи порога:
- Long surplus: цена ≥ `last_reb × (1 + threshold_surplus)`
- Short surplus: цена ≤ `last_reb × (1 - threshold_surplus)`
- Long deficit: цена ≤ `last_reb × (1 - threshold_deficit)`
- Short deficit: цена ≥ `last_reb × (1 + threshold_deficit)`

Bypass: первый запуск (`ignore_limits=True`) или `last_reb == 0`.

### 6. Safety Guards

- **Velocity Guard** — блокировка при быстром движении цены (>1% за 60с)
- **Trend Guard** — блокировка при однонаправленном тренде (efficiency > 0.85, move > 0.5%)
- **Spread Guard** — блокировка при широком спреде (>0.15%, только REAL режим)

## Конфигурация

```json
{
  "portfolios": [{
    "targets": {
      "BASE_LONG":  {"share": 0.27, "leverage": 5},
      "BASE_SHORT": {"share": 0.34, "leverage": 5},
      "VIRTUAL":    {"share": 0.35, "leverage": 1}
    },
    "rebalance_threshold_surplus": 0.015,
    "rebalance_threshold_deficit": 0.03,
    "min_notional_usdt": 7.0,
    "safety_guards": {
      "max_spread_pct": 0.15,
      "max_price_velocity_pct": 1.0,
      "velocity_window_sec": 60
    }
  }],
  "equity_trailing_stop_pct": 10.0,
  "equity_trailing_stop_activation_pct": 2.0,
  "equity_trailing_stop_timeout_sec": 0,
  "max_drawdown_limit": 33.0
}
```

## Архитектура

| Модуль | Функция |
|--------|---------|
| `main.py` | Основной цикл: TPV, стоп-условия, вызов calculator/executor |
| `calculator.py` | Математика портфеля: доли, отклонения, действия ребалансировки |
| `executor.py` | Исполнение ордеров, polling, reconciliation |
| `connector.py` | Binance Futures API: цены, позиции, сделки |
| `rank_tickers.py` | Сканер кандидатов, scoring |
| `backtest_rebalance.py` | Бэктест и live-verification |
| `supervisor.py` | Управление роем: PM2, ротация, promotion |

## Skeleton Key Policy

> Каждый бот работает только со своим изолированным капиталом (shadow balance).

Бот может:
- Использовать свой `initial_capital` (при первом запуске)
- Использовать свою прибыль (proceeds от продажи излишков)

Бот **не может**:
- Тратить свободную маржу сверх своего капитала
- Тратить прибыль других ботов

Биржевой маржинал не учитывается при принятии решений — `calculator.py` работает только с shadow balance бота.
