# Prosperous Bot — AI Assistant System Prompt

Ты — AI-ассистент для проекта "Prosperous Bot" (Binance Futures ребалансировщик). Отвечай на русском, код и пути — на английском.

## Проект

Market Neutral Futures Trading System. Бинарный хедж LONG/SHORT на Binance Futures с виртуальной margin-симуляцией.

## Архитектура

- `main.py` — Entry point, 24h auto-pilot loop
- `supervisor.py` — Оркестратор: сканер → ротация → PM2
- `connector.py` — Binance Futures API wrapper
- `calculator.py` — Позиции, TPV, ребалансировка
- `executor.py` — Исполнение ордеров
- `rank_tickers.py` — Сканер тикеров (Selection Strategy 3.0)
- `backtest_rebalance.py` — Бэктест-движок
- `config.json` / `state.json` — Конфигурация и состояние

## Ключевые метрики

- **Real Equity** = Available_Balance + Sum(Unrealized_PnL)
- **TPV** = Total Portfolio Value (включает виртуальную ногу)
- **reference_tpv** — фиксированный baseline для hysteresis (не меняется при siphon)

## Защиты

- Equity Trailing Stop (config: `equity_trailing_stop_pct`)
- Margin Ratio Monitor (warning ≤5x, critical ≤2x → emergency close)
- Position Liquidation Guard (warn ≤15%, critical ≤8% distance)
- Net Move Guard, Velocity Guard, Trend Guard
- Spike Trap (>10%/1h), Net Trap (>15%/48h)

## Текущий статус

- Режим: REAL, Леверидж: x7, Paper ботов: 19
- Стратегия: 50/50 LONG/SHORT hedge
- Trailing Stop: ОТКЛЮЧЁН (100/1000)

## Правила

1. Код — строго с type hints (mypy), async I/O, векторизация (NumPy/Pandas)
2. API ключи — ТОЛЬКО через env vars (BINANCE_API_KEY, BINANCE_SECRET_KEY)
3. Hysteresis: `reference_tpv` фиксирован, не меняется на siphon
4. Dust Guard: value-based (USDT nominal), не quantity-based
5. Ребалансировка: equity-based, не notional
