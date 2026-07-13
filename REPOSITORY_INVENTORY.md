tags: [inventory, architecture, repository-analysis]
created: 2026-07-21

# Repository Inventory — FMProducer/prosperous_bot

## Executive Summary

The `prosperous_bot` repository is a comprehensive, highly automated algorithmic cryptocurrency trading ecosystem primarily built around **Market-Neutral Futures Portfolio Rebalancing** and **Deep Reinforcement Learning (DRL)** strategies for Binance Futures. It consists of exactly **4 distinct systems**: a production PM2-orchestrated Futures Swarm Rebalancer, a Spot Signal-Generator & ML Backtester, a multi-model (2+2) DRL ensemble agent system integrated with Freqtrade, and the Freqtrade execution engine itself. The repository actively manages **4 live trading bots** and **16 paper trading bots** running concurrently, supported by multiple automated monitoring and reporting services.

---

## Repository at a Glance

| Metric              | Count |
|---------------------|-------|
| Total .py files     | 699   |
| Distinct systems    | 4     |
| Active trading bots | 4     |
| Service processes   | 3     |
| Test files          | 42    |
| Config files        | 65    |
| Documentation files | 157   |

---

## Directory Map

```
.
├── .agents/                    # Internal AI agent workflow descriptions and system prompt caches.
├── .hermes/                    # AI companion execution plans, session state files, and logs.
├── config/                     # Shared parameters and unified optimizer/backtester JSON specifications.
├── freqtrade/                  # External cloned modular cryptocurrency trading bot execution framework.
├── futures_portfolio/          # Active System: Multi-bot market-neutral futures portfolio rebalancer (Production).
├── graphs/                     # Visualization outputs, generated plots, and dynamic signals CSV stores.
├── output/                     # Saved machine learning models, Optuna study databases, and evaluation logs.
├── patches/                    # System maintenance patches, hotfixes, and change logs.
├── reports/                    # Backtest summary folders, execution logs, and code coverage outputs.
├── scripts/                    # Dependency installers and system environment setup scripts.
├── signals/                    # State-sharing directory with PM2 process exit and stop flag files.
├── src/                        # Active System: Prosperous Bot ML backtesting and signal generator framework.
├── templates/                  # HTML templates for rendering summary reports and dashboards.
├── test_data/                  # Test input data for validation and verification tasks.
├── tests/                      # Dedicated test suite verifying the ML Signals and Backtesting modules.
├── third_party/                # Active System: DRL agents and environment (rl-trading-binance module).
├── tools/                      # System administration utilities, network helpers, and pytest executors.
└── venv/                       # Cloned Python virtual environment containing external dependencies.
```

---

## Systems Inventory

### System 1: Market-Neutral Futures Portfolio Rebalancer

| Field         | Value                            |
|---------------|----------------------------------|
| Location      | `futures_portfolio/`             |
| Purpose       | Fully-automated multi-bot rebalancer utilizing a synthetic market-neutral hedge across Binance Futures. |
| Entry points  | `main.py`, `supervisor_service.py`, `aggregator.py`, `telegram_sender.py`, `dashboard/dashboard.py` |
| Status        | Active (Production)              |
| External deps | Binance Futures API, Telegram Bot API, PM2 Runtime Engine, Vosk Speech-to-Text |
| Config        | `config.json`, `ecosystem.config.js` |
| State files   | `paper_state_*.json`, `paper_shadow_*.json`, `real_state_*.json` |

#### Components:
- `main.py` — Individual trading bot loop managing isolated capital, fetching mark price, executing guards, and calling calculator/executor.
- `supervisor_service.py` / `supervisor.py` — Orchestrator service managing the swarm, starting/stopping bots in PM2, and running ticker rotation based on ranking.
- `calculator.py` — Precision portfolio calculator operating on Decimals to calculate TPV, allocations, and deviations.
- `connector.py` — Asynchronous wrapper for Binance Futures REST/WebSocket API with auto-retry and RF bypass logic.
- `executor.py` — Order executor utilizing step-size rounding, post-only limits, and market fallbacks.
- `rank_tickers.py` — Scanning and scoring script executing Selection Strategy 3.0 to rank tradeable candidates.
- `backtest_rebalance.py` — Dedicated rebalancing backtester to evaluate historical ticker performances over multiple days.
- `aggregator.py` — Background service aggregating active bot metrics into unified statistics.
- `telegram_sender.py` — Telegram messenger process dispatching alerts and summaries from a shared queue.
- `dashboard/dashboard.py` — Flask-based dashboard offering read-only monitoring and system performance visualizations.
- `health_check.py` — Diagnostic watchdog script verifying PM2 processes and API keys.

#### Architecture:
```
                      +-------------------------+
                      |   supervisor_service    | <----+ (PM2 Process)
                      +-------------------------+
                                   | (Rotates / Spawns)
                                   v
                      +-------------------------+
                      |  main.py (Per Ticker)   | <----+ (PM2 Process)
                      +-------------------------+
                       /           |           \
                      v            v            v
              +-----------+  +-----------+  +-----------+
              | connector |  |calculator |  | executor  |
              +-----------+  +-----------+  +-----------+
                    |                           |
                    v (Rest API calls)          v (Executes Orders)
             +==========================================+
             |            Binance Futures API           |
             +==========================================+
```

---

### System 2: Prosperous Bot — ML Signals & Backtester Framework

| Field         | Value                            |
|---------------|----------------------------------|
| Location      | `src/prosperous_bot/`            |
| Purpose       | Fetches market candles, trains XGBoost classifiers, and runs portfolio backtesting/parameter optimizations. |
| Entry points  | `signal_bot.py`, `rebalance_optimizer.py`, `futures_rebalance_backtester.py` |
| Status        | Active / Experimental            |
| External deps | Binance API, Optuna, Scikit-learn, XGBoost |
| Config        | `config_signal.json`, `config/unified_config.example.json` |
| State files   | `bot_state.json`                 |

#### Components:
- `signal_bot.py` — The core signal generator loop predicting daily market regimes and saving trading flag signals.
- `futures_rebalance_backtester.py` — Portfolio simulator testing spot, long, and short allocations under transaction fees and slippage.
- `rebalance_optimizer.py` — Optuna optimizer searching for the most lucrative rebalancing intervals, thresholds, and weights.
- `ml_model.py` — Feature engineering module and wrapper for XGBoost models.
- `strategy.py` — Defines RuleBased, ML, and Hybrid decision engines to map predictions to signals.
- `portfolio_manager.py` — Simulates asset allocations and evaluates net asset values (NAV) during backtests.
- `data_loader.py` — Handles CSV and Binance data parsing.
- `update_distribution.py` — Writes calculated target asset distributions to state configs.

#### Architecture:
`signal_bot.py` runs periodically to trigger `data_loader.py`. Features are computed and passed to the trained model inside `ml_model.py`. The prediction maps to a signal via `strategy.py`, generating an interactive Plotly chart in `graphs.py`. Optimization scripts (`rebalance_optimizer.py`) loop over different hyperparameter suggestions, calling `futures_rebalance_backtester.py` to evaluate Sharpe Ratios.

---

### System 3: RL-Trading-Binance (DRL Engine)

| Field         | Value                            |
|---------------|----------------------------------|
| Location      | `third_party/rl-trading-binance/` |
| Purpose       | Deep Reinforcement Learning (DRL) agent training and execution using a Double Dueling Q-Network (D3QN) with PER. |
| Entry points  | `main.py`, `train_ohlcv_z.py`, `optimize.py`, `paper_trader_q.py` |
| Status        | Active                           |
| External deps | PyTorch, OpenAI Gym/Gymnasium, PostgreSQL, ONNX Runtime |
| Config        | `user_data/config_rl4z.json`, `config_ws.yaml` |
| State files   | SQLite database, Postgres, `.npz` training datasets |

#### Components:
- `agent.py` — Implements `D3QN_PER_Agent` with Prioritized Experience Replay (PER).
- `model.py` — Neural network architecture mapping state channels to Q-values.
- `trading_environment_z.py` — Gymnasium-compliant custom trading simulator with z-score normalization.
- `replay_buffer.py` — Sum-Tree implementation for priority sampling using optimized NumPy vectorization.
- `paper_trader_q.py` — Live paper trading execution module running on real-time WebSockets.
- `inference_adapter.py` — Handles standard model prediction and ONNX quantization.

#### Architecture:
An ensemble of four neural networks (2 Long-only, 2 Short-only) processes 10-channel z-score normalized market arrays in parallel. Voting occurs if a model's Q-value exceeds an adjusted epsilon threshold. If the opposing side casts a strong veto vote, the trade is blocked. This logic executes within Freqtrade via `CustomD3QNStrategy4z.py`.

---

### System 4: Freqtrade Strategy Execution Engine

| Field         | Value                            |
|---------------|----------------------------------|
| Location      | `freqtrade/`                     |
| Purpose       | Modular open-source cryptocurrency trading bot written in Python, hosting the DRL strategies. |
| Entry points  | `freqtrade/main.py`              |
| Status        | Active (External Cloned Subtree)  |
| External deps | CCXT Library, SQLAlchemy, FastAPI, Telegram API |
| Config        | `config_examples/config_full.example.json` |
| State files   | `tradesv3.dryrun.sqlite`         |

#### Components:
- `freqtrade/commands/` — High-level console entry points (`trade`, `backtesting`, `hyperopt`).
- `freqtrade/exchange/` — API wrappers handling order book, leverage, and margin settings across exchanges via CCXT.
- `freqtrade/strategy/` — Base classes for loading user-defined parameters, indicators, and trade exit protections.

---

## Bot Process Registry

The production swarm currently configures the following real/paper bots and service processes under PM2:

| #   | Process Name       | Type    | Ticker     | Capital  | Module                  | PM2 ID | Status  |
|-----|--------------------|---------|------------|----------|-------------------------|--------|---------|
| 1   | supervisor-service | Service | N/A        | N/A      | `supervisor_service.py` | 0      | online  |
| 2   | swarm-aggregator   | Service | N/A        | N/A      | `aggregator.py`         | 1      | online  |
| 3   | telegram-sender    | Service | N/A        | N/A      | `telegram_sender.py`    | 2      | online  |
| 4   | real-grass         | Real    | GRASSUSDT  | 20 USDT  | `main.py`               | 3      | online  |
| 5   | real-uni           | Real    | UNIUSDT    | 20 USDT  | `main.py`               | 4      | online  |
| 6   | real-vvv           | Real    | VVVUSDT    | 20 USDT  | `main.py`               | 5      | online  |
| 7   | real-yfi           | Real    | YFIUSDT    | 20 USDT  | `main.py`               | 6      | online  |
| 8   | paper-* (16 bots)  | Paper   | 16 Tickers | 20 USDT  | `main.py`               | 7-22   | online  |

---

## Configuration Reference

### Primary Config (`futures_portfolio/config.json`)

Key parameters used in the multi-bot production rebalancer:

- `"max_bots"`: `20` — Limits the concurrent sum of active paper and real bots in the system.
- `"paper_mode_bots"`: `16` — Determines how many paper bots supervisor will spawn.
- `"min_notional_usdt"`: `5.1` — The minimum order value required by Binance; orders under this size are skipped.
- `"targets"`:
  - `"BASE_LONG"`: `{"share": 0.5, "leverage": 7}` — Target allocation (50%) and leverage (x7) for the Long leg.
  - `"BASE_SHORT"`: `{"share": 0.5, "leverage": 7}` — Target allocation (50%) and leverage (x7) for the Short leg.
  - `"VIRTUAL"`: `{"share": 0.0, "leverage": 1}` — Virtual Spot leg target share (currently disabled at 0.0).
- `"rebalance_threshold_surplus"`: `0.01` — Assets exceeding their target weight by 1% are partially sold.
- `"rebalance_threshold_deficit"`: `0.04` — Assets falling under their target weight by 4% are bought using proceeds.
- `"live_swarm"`: `["AGLDUSDT", "TLMUSDT", "TUSDT", "VANRYUSDT"]` — Hardcoded list of tickers for live trading.

---

### Safety Mechanisms

The ecosystem implements multiple safety nets to protect capitals:

1. **Margin Ratio Watchdog**:
   - **Warning**: Margin Ratio $\le$ 3.0x (Triggers Telegram warning).
   - **Critical**: Margin Ratio $\le$ 1.5x (Initiates emergency market liquidation of all positions and shuts down).
2. **Equity Trailing Stop**:
   - Configured with `equity_trailing_stop_pct` (`0.001%` - currently disabled) and an activation baseline of `9.0%`.
3. **Price Velocity Guard**:
   - Blocks rebalancing actions if the spot price fluctuates over `2.0%` within a `60`-second sliding window (`velocity_window_sec`).
4. **Trend Guard**:
   - Suspends orders if price efficiency exceeds `0.55` and the cumulative candle movement exceeds `0.91%` (`trend_min_move_pct`).
5. **Net Move Guard**:
   - Blocks trades if the price trends unilaterally by `2.14%` over `72` seconds.
6. **Toxic Blacklist Cooldown**:
   - Automatically benches volatile tickers for `0.01` days (cooldown) after abnormal spikes.

---

## Test Coverage Map

### Active Tests Inventory

| System Under Test | Test File | Test Count | Coverage Notes |
|-------------------|-----------|------------|----------------|
| **Rebalancer Core** | `futures_portfolio/tests/test_calculator.py` | 12 | Tests precise Decimals portfolio allocation values, TPV, and margins. |
| **Rebalancer Core** | `futures_portfolio/tests/test_executor.py` | 22 | Verifies order quantity step-size floor rounding and limit/market actions. |
| **Rebalancer Core** | `futures_portfolio/tests/test_connector.py` | 44 | Mocks Binance API responses, handling connectivity issues and rate limits. |
| **Rebalancer Core** | `futures_portfolio/tests/test_main.py` | 20 | Tests individual bot loop heartbeats, self-killing signals, and guards. |
| **Rebalancer Core** | `futures_portfolio/tests/test_main_new.py` | 13 | Verifies edge cases for fresh capital allocation and post-only locks. |
| **Rebalancer Core** | `futures_portfolio/tests/test_main_coverage.py` | 21 | Focuses on expanding coverages over trailing stop and max drawdown checks. |
| **Rebalancer Core** | `futures_portfolio/tests/test_rebalance_v37.py` | 1 | Smoke test verifying simple portfolio rebalancing orders build. |
| **Rebalancer Core** | `futures_portfolio/tests/test_rebalance_logic_v378.py` | 1 | Verifies rebalance logic specifically on version 3.7.8 modifications. |
| **Rebalancer Core** | `futures_portfolio/tests/test_get_bot_efficiency.py` | 3 | Checks supervisor calculation of bot profit and execution indices. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor.py` | 46 | Verifies rotation, bot promotion, probation rules, and PM2 commands. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor_new.py` | 19 | Tests additional rotation bounds and shadow state reconciliations. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor_whitelist.py` | 2 | Ensures only authorized tickers bypass rotation into live trading. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor_rotation_whitelist.py` | 1 | Focuses on rotation filters under real whitelisting flags. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor_coverage.py` | 26 | Broadens coverage over state validation, save locks, and error recoveries. |
| **Supervisor Service** | `futures_portfolio/tests/test_storage.py` | 10 | Verifies file locking, safe JSON loads, and corrupted save protections. |
| **Supervisor Service** | `futures_portfolio/tests/test_aggregator.py` | 7 | Mocks PM2 stats fetching and aggregates paper/real shadow balances. |
| **Supervisor Service** | `futures_portfolio/tests/test_send_monitor_report.py` | 6 | Checks generation of diagnostic reports sent via Telegram API. |
| **Supervisor Service** | `futures_portfolio/tests/test_health_check.py` | 9 | Tests PM2 presence checks and API credential validations. |
| **Supervisor Service** | `futures_portfolio/tests/test_rank_tickers.py` | 24 | Mocks candle databases to verify Strategy 3.0 ticker scanning. |
| **Supervisor Service** | `futures_portfolio/tests/test_notifier.py` | 20 | Mocks queueing and rate limit compliance for Telegram messages. |
| **Supervisor Service** | `futures_portfolio/tests/test_telegram_sender.py` | 10 | Checks thread-safe queue consumption by the background sender. |
| **Supervisor Service** | `futures_portfolio/tests/test_swarm_analyzer.py` | 2 | Tests analyzer calculations for total swarm profitability. |
| **Supervisor Service** | `futures_portfolio/tests/test_supervisor_service.py` | 4 | Checks the high-level systemd service loop wrapper. |
| **ML Signal Bot** | `tests/test_utils.py` | 7 | Focuses on directory and CSV file utility handlers. |
| **ML Signal Bot** | `tests/test_ci_probe.py` | 2 | Simple CI heartbeat probe assertions. |
| **ML Signal Bot** | `tests/test_rebalance.py` | 12 | Tests Spot rebalancer limits, debounces, and dynamic thresholds. |
| **ML Signal Bot** | `tests/test_portfolio.py` | 3 | Validates spot portfolio manager distribution returns. |
| **ML Signal Bot** | `tests/test_ensemble_logic.py` | 3 | Verifies ML strategy voting vetoes. |
| **ML Signal Bot** | `tests/test_reaper_guard.py` | 2 | Mocks the signal decay reaper to verify state pruning. |
| **ML Signal Bot** | `tests/test_futures_rebalance_backtester.py` | 26 | Verifies backtester metrics: Sharpe, Sortino, max drawdowns. |
| **ML Signal Bot** | `tests/test_exchange_api.py` | 26 | Verifies basic Gate.io / Binance REST mock exchange responses. |
| **ML Signal Bot** | `tests/test_exchange.py` | 12 | Mocks exchange connection handshakes and orders fetching. |
| **ML Signal Bot** | `tests/test_portfolio_property.py` | 1 | Property-based tests verifying spot allocations always sum to 100%. |
| **ML Signal Bot** | `tests/test_rebalance_engine.py` | 17 | Property-based tests checking rebalancing order outputs. |
| **ML Signal Bot** | `tests/test_exchange_property.py` | 3 | Property-based tests verifying buy/sell balance invariance. |
| **RL Trading System** | `third_party/rl-trading-binance/tests/test_custom_d3qn_strategy4z.py` | 2 | Mocks Freqtrade strategy class to test the PyTorch agent routing. |
| **RL Trading System** | `third_party/rl-trading-binance/tests/test_agent_routing.py` | 4 | Verifies routing predictions across multi-channel model frames. |
| **RL Trading System** | `third_party/rl-trading-binance/tests/test_dynamic_slots.py` | 11 | Tests slot allocations based on long/short profitability ratios. |
| **RL Trading System** | `third_party/rl-trading-binance/tests/test_config_rl4z.json` | 10 | Validates rl4z configurations, thresholds, and regime structures. |
| **RL Trading System** | `third_party/rl-trading-binance/tests/test_trading_environment_z.py` | 2 | Mocks z-score normalizations and Gym observations. |

*Note: In total, there are 42 active Python test files hosting 483 verified test cases.*

---

## Evolution Timeline

The ecosystem has evolved through structured development phases:

- **May 2026 (Phase 0 — Foundation)**:
  - Architecture of `main.py`, `connector.py`, `calculator.py`, and `executor.py` mapped out.
  - Developed the market-neutral hedge ratio (27% LONG / 36% SHORT / 35% VIRTUAL).
  - Designed the initial backtest engine (`backtest_rebalance.py`) and scanner (`rank_tickers.py`).
- **May–June 2026 (Phase 1 — Security and Protections)**:
  - Created Margin Ratio Monitoring protections.
  - Implemented the Equity Trailing Stop and Position-Level Liquidation Guards.
  - Added Net Move, Trend, and Velocity guards to prevent trading during abnormal slippage.
  - Added the Supervisor Auto-Restart Guard (`_ensure_real_bots_alive`).
  - *Incident (May 30)*: PORTALUSDT short liquidation due to missing predictive guards; supervisor killed a real bot during a restart.
  - *Incident (June 5)*: EPICUSDT short liquidation; cross-margin prevented API from returning a precise liquidation price.
- **June 2026 (Phase 2 — Optimization and Tuning)**:
  - Virtual leg disabled (`VIRTUAL OFF`) to eliminate drag during massive trends; migrated to a pure 50/50 LONG/SHORT model.
  - Adjusted rebalancing triggers to 1.1% surplus / 1.1% deficit thresholds.
  - Rewrote rotation algorithm to follow strict score-driven rankings.
  - Created `health_check.py` with cron.
- **July 2026 (Phase 3 — Monitoring Infrastructure)**:
  - Built the Dash/Flask read-only Dashboard.
  - Implemented STT voice-command controls utilizing Vosk.
  - Built `MOC.md` (Map of Content) to serve as the project navigation hub.
- **July 2026 (Phase 4 — Trailing Stop Stabilization)**:
  - Deactivated trailing stop due to noise-induced false exits.
  - Capped maximum drawdown limit to 100%.
- **July-August 2026 (Phase 5 — Scaling)**:
  - Active phase aiming at scaling the swarm size, improving WebSocket connectivity, and closing the remaining race conditions.

---

## Legacy & Archives

- **`VIRTUAL` Spot Leg Trading**: Although code for mathematical virtual spot calculations exists in `calculator.py`, the active configuration has `VIRTUAL` share set to `0.0`, rendering this leg obsolete in the current pure futures hedge strategy.
- **`config.json` Race Conditions**: Due to `main.py` and `supervisor.py` writing directly to `config.json` simultaneously, this logic is scheduled to be superseded by a unified database state or file-locked operations.
- **`setup_env.sh` (Strict Mode)**: Enforces code coverage thresholds $\ge 90\%$, but is mostly bypassed during local manual debugging runs.

---

## Cross-System Dependencies

- **`freqtrade/`** acts as an execution environment which directly invokes **`third_party/rl-trading-binance/`** strategies (`CustomD3QNStrategy4z.py`).
- **`futures_portfolio/`** is a standalone rebalancing client. It shares Binance API keys and logs, but operates on its own isolation mechanisms, distinct from **`src/prosperous_bot`**'s spot signal logic.
- **`src/prosperous_bot/`** generates signal outputs which are consumed during backtesting or model verification loops.

---

## Appendix: Complete File Inventory

### Python Files by Directory

| Directory | Count | Purpose |
|-----------|-------|---------|
| `.` (Root) | 11 | Global deployment tools, database status checks, and run-trial analyzers. |
| `freqtrade/` | 454 | Execution engine, CCXT integrations, commands, databases, and standard strategies. |
| `futures_portfolio/` | 67 | Production rebalancer scripts, calculator, connector, executor, and dashboard. |
| `output/` | 6 | Saved evaluation metrics and neural network weight binaries. |
| `scripts/` | 2 | Dependency installers. |
| `src/` | 19 | Active Signal bot, ML strategies, portfolio manager, and spot rebalancer optimizer. |
| `tests/` | 13 | Tests for the ML Signal Bot, backtesting ratios, and mock exchanges. |
| `third_party/` | 124 | DRL agents, neural network model graphs, custom gym envs, and training logs. |
| `tools/` | 3 | Network scripts and coveraged pytest executors. |

---

### Configuration Files

| File | Format | Used By |
|------|--------|---------|
| `futures_portfolio/config.json` | JSON | Active multi-bot production rebalancer. Defines allocations and safety limits. |
| `config_signal.json` | JSON | Specifies the feature windows and model parameters for the ML Signal Bot. |
| `config/unified_config.example.json` | JSON | Unified template configuring the spot rebalancer backtests and Optuna optimizer. |
| `third_party/rl-trading-binance/user_data/config_rl4z.json` | JSON | Primary config defining PyTorch model paths and Freqtrade limits for the DRL agent. |
| `pyproject.toml` | TOML | Root package definitions and package setup guidelines. |
| `pytest.ini` | INI | Test engine execution and exclusion patterns. |
