# Progress Log — Market-Neutral Futures Portfolio Rebalancer

## 2026-05-31 — Local Memory System & Documentation

### Memory System Created
- **Location**: `D:\\Hermes-USB-Portable-main\\data\\memory_db\\` (on USB flash drive)
- **Two-level architecture**:
  - Level 1: `memories/MEMORY.md` + `memories/USER.md` — auto-loaded at startup (~4KB)
  - Level 2: `memory_db/` — full details on demand (search/read)
- **memory_mgr.py**: Python script for memory management (add/list/search/compact/show)
- **No external dependencies**: all local files, no cloud/API/subscriptions
- **No compression needed anymore**: detailed memory in memory_db/, compact navigation in memories/

### Documentation Updated
- `MEMORY.md` — rewritten as navigation index with links to memory_db/
- `USER.md` — deduplicated, added memory_db reference
- `docs/progress.md` — this file
- `docs/README.md` — system description and rules

### Clarifications from FMProducer
- docs/ is inside `futures_portfolio/`, NOT in project root
- filesystem operations use `execute_code` + Python (not bash with Windows paths)
- `dir /b` doesn't work in MSYS bash — use `ls` or Python `os.listdir()`

---

## 2026-05-30 — Rebalancing Logic Fix: No Free Margin Usage

### Problem Identified
`calculator.py` line 224 used `self.val_cash + total_proceeds` as available funds for deficit purchases. This meant bots could spend **free margin from the Binance account** to buy deficits — violating the core principle that each bot must operate only with its own initial capital and profits.

### Root Cause
- `val_cash` = residual cash in the bot's shadow balance (free margin)
- `total_proceeds` = equity harvested from surplus sales
- Combined, this allowed averaging into losing positions using external funds

### Fix Applied
**File:** `calculator.py`, function `calculate_deviations()`, line 224

**Before:**
```python
available_funds = self.val_cash + total_proceeds
```

**After:**
```python
if ignore_limits and total_proceeds == 0:
    available_funds = self.val_cash  # First startup: build positions from initial capital
else:
    available_funds = total_proceeds  # Only own profit from surplus sales
```

### Rules Established
1. **Each bot manages only its own initial capital** — no spending free margin from the Binance account
2. **Deficit purchases ONLY from surplus proceeds** — "not a single cent from free margin"
3. **First startup exception** — `ignore_limits=True` and `total_proceeds==0` → use `val_cash` to build initial positions
4. **Surplus threshold filter** — if `diff_usdt < min_notional` (7 USDT), skip silently
5. **Deficit threshold filter** — same logic, no forced purchases
6. **TPV must NOT decrease during rebalancing** — all profit stays inside the bot and reinvested
7. **Trailing stop and max drawdown** apply to **total TPV of the bot** (all legs combined), NOT individual positions
8. **NO hard stop from initial_tpv** — breaks market-neutral portfolio logic
9. **NO per-ticker stops in config** — single stop for entire bot portfolio

### Tests Passed
- Test 1: Normal rebalance (price 0.506) — SHORT surplus sold, LONG deficit below threshold → no purchase ✅
- Test 2: First startup — all three legs (VIRTUAL, LONG, SHORT) built from initial capital ✅
- Test 3: Strong price drop (0.480) — SHORT surplus → proceeds → LONG deficit purchased entirely from proceeds ✅

### Related Analysis: GRASSUSDT Loss Post-Mortem
- 8 rebalance cycles, all SHORT expansions on surplus threshold breach
- Price fell 0.514 → 0.447 (-13%), TPV: 115 → 108.74 (-5.5%)
- Old behavior: deficit LONG purchases were funded by free margin (averaging into loss)
- New behavior: LONG deficit won't be bought unless SHORT surplus generates enough proceeds
- Result: bot may do nothing if thresholds aren't met — this is correct behavior

### Trailing Stop Known Issues
- `tpv_ath` initialized to 0 (not `initial_tpv`) — trailing stop won't activate on losing starts
- `activation_pct=2%` too high for paper — TPV must exceed 102% of initial before trailing activates
- **Decision:** defer trailing stop fix to future task — current focus is rebalancing logic correctness

---

## 2026-05-31 — Trailing Stop Fix, Toxic Blacklist, Backtest Sync

### Trailing Stop Bug Fix
**File:** `main.py`

**Problem:** `tpv_ath` was initialized to `0.0` instead of `target_initial_capital`. This caused trailing stop to not activate on losing starts and ghost ATH from first rebalance commission drop.

**Fix:**
- `tpv_ath` initialized to `target_initial_capital` instead of `0.0`
- `timeout_sec` changed from `0` to `60` (interim value, Optuna will optimize later)
- `emergency_stop` updated: state files archived to `history/` with timestamp, then originals deleted

### State Archive on Stop
**File:** `main.py` (emergency_stop / trailing stop handler)

After trailing stop or stop-loss:
1. State files copied to `history/` with timestamp suffix
2. Originals deleted
3. Prevents restart loop with stale `tpv_ah`

### Paper Restart Logic
**File:** `supervisor.py`

- `reset_paper_state()` added — creates clean state files, archives old ones
- `start_bot` for paper always resets state (fresh start each time)
- For real mode: only reset if state files don't exist

### Toxic Blacklist Logic — Formal Rules
**File:** `supervisor.py` + `main.py`

| Signal | Condition | Paper | Real |
|--------|-----------|-------|------|
| `"exit"` | Trailing stop, PnL ≥ 0 | Bot stops, waits for scanner approval | Bot stops → moves to paper (incubator), waits for scanner |
| `"stop"` | Stop-loss, PnL < 0 | Bot stops → added to `toxic_blacklist` → cooldown | Bot stops → added to `black_list` → cooldown |

**Key principle:** Only `stop` (negative PnL) goes to blacklist. `exit` (positive PnL) — bot just stops and waits.

### Supervisor Signal Reading (NEW)
**File:** `supervisor.py`, function `manage_swarm()`, step 0.5

Added signal reading **before** scanner runs:
- Reads `signals/stop_*.flag` files → adds ticker to `toxic_blacklist` with `toxic_cooldown_days` expiry
- Reads `signals/exit_*.flag` files → just deletes (no blacklist)
- Prunes expired entries from `toxic_blacklist`
- If `signals/` doesn't exist → silently skips

This ensures that after stop-loss with negative PnL, the ticker is **guaranteed** to be blacklisted even if the scanner doesn't mark it as `is_toxic`.

### Backtest Synchronization
**File:** `calculator.py` + `backtest_rebalance.py`

**Problem:** `backtest_rebalance.py` used `state.val_cash <= 0` as a guard for expansions. This was incorrect because:
1. `val_cash` could go negative from commissions/slippage
2. It didn't reflect the new `available_funds = total_proceeds` logic

**Fixes:**

1. **`calculate_deviations()` return format changed:**
   - Was: `List[Dict]` (actions list only)
   - Now: `Dict` with keys: `actions`, `available_funds`, `tpv`, `total_tpv`, `share_long_pct`, `share_short_pct`, `share_virt_pct`, `share_cash_pct`

2. **`calculate_rebalance()` updated:**
   - Unwraps dict from `calculate_deviations()`
   - Returns consistent dict with all fields

3. **`backtest_rebalance.py` expansions:**
   - Replaced `state.val_cash <= 0` with `remaining_funds <= 0` (from `calc_res["available_funds"]`)
   - If `needed_usdt > remaining_funds * lev` → qty is reduced to available limit
   - After each expansion: `remaining_funds -= actual_equity_spent`

4. **`min_notional_usdt` enforced for:**
   - Reductions (surplus sales) ✅ — `validate_notional()` at line 315
   - Expansions (deficit purchases) ✅ — `validate_notional()` at line 377
   - Calculator also filters: surplus at line 202, deficits at line 265

### Bug Fixed: `'str' object has no attribute 'get'`
**Root cause:** `calculate_deviations()` was changed to return dict, but `calculate_rebalance()` still treated result as a list. This caused `actions` to be a dict instead of list, and `for a in actions` iterated over string keys.

**Fix:** `calculate_rebalance()` now correctly extracts `actions` from the dict returned by `calculate_deviations()`.

### Verification
- `run_all_backtests.py` — 345 tickers processed, no errors ✅
- `log.md` analysis: 22 bots online, TPV stable at 114.86, Trend Guard working on RIF/ALGO/DYDX ✅
- All syntax checks passed ✅

---

## 2026-05-31 — Optuna Stop-Loss Optimization & Live Trading Launch

### Phase 1: Optuna Optimization of Stop-Loss Parameters

**Goal:** Optimize 4 stop-loss parameters using Optuna with multi-ticker backtesting.

**Optimized parameters:**
- `max_drawdown_limit` (5–50%)
- `equity_trailing_stop_pct` (2–25%)
- `equity_trailing_stop_activation_pct` (0.5–10%)
- `equity_trailing_stop_timeout_sec` — NOT optimized, fixed at 60s from config

**Method:**
- Script: `optimize_stops.py`
- Each trial tests ONE parameter combination on ALL 20 tickers from config
- Metric: average `profit_pct` across all tickers
- TPE sampler, MedianPruner, 100 trials
- Storage: `optuna_stops.db` (SQLite, resumable)
- Period per backtest: `backtest_period_days` = 0.125 (3 hours)

| Ticker | Old (3h) | New (3h) |
|--------|----------|----------|
| PORTALUSDT | +17.77% | +21.84% |
| FORMUSDT | +2.95% | +3.08% |
| STGUSDT | +2.44% | +2.68% |
| DYDXUSDT | +0.64% | +0.52% |
| Overall Top-20 avg | ~1.5% | ~1.6% |

**Optimal parameters found:**

| Parameter | Old | New (Optuna) |
|-----------|-----|--------------|
| `max_drawdown_limit` | 33.0% | **22.0%** |
| `equity_trailing_stop_pct` | 10.0% | **24.0%** |
| `equity_trailing_stop_activation_pct` | 2.0% | **7.5%** |
| `equity_trailing_stop_timeout_sec` | 60s | 60s (fixed) |

**Why it works better:**
- Tighter max_drawdown (22% vs 33%) cuts losing positions earlier
- Wider trailing stop (24% vs 10%) lets profitable positions breathe through volatility
- Higher activation threshold (7.5% vs 2%) prevents premature activation on noise

### Phase 2: 7-Day Backtest Verification

**Goal:** Validate optimized parameters on longer time window (7 days) with 20 tickers.

**Results:**

| Metric | Value |
|--------|-------|
| Total tickers | 20 |
| Profitable | 18/20 |
| Avg profit | +9.78% |
| Avg max DD | 7.99% |
| Worst ticker | SWARMSUSDT -9.62% |
| Best ticker | XLMUSDT +31.72% |
| Trailing Stop triggered | 0/20 |
| Total cycles | 81 |
| Liquidations | 0 |

**Top 5 by profit:**
1. XLMUSDT +31.72% (DD 7.93%)
2. IDUSDT +17.10% (DD 8.73%)
3. INJUSDT +15.78% (DD 5.41%)
4. FETUSDT +15.02% (DD 5.92%)
5. DYDXUSDT +14.68% (DD 6.74%)

**Losers analysis:**
- VVVUSDT -3.06%: asset fell 8.75%, 0 rebalance cycles
- SWARMSUSDT -9.62%: asset fell 27.49%, 0 rebalance cycles (too volatile, filtered by min_cycles)

### Phase 3: Live Paper Trading Results (8 hours)

**Incubator Swarm (paper trading with real Binance prices):**

| Metric | Value |
|--------|-------|
| Period | 8 hours |
| Total PnL | **+14.57 USDT** |
| Incubator ROI | ~12.7% (on ~115 USDT capital) |
| Profitable tickers | 13/30 |
| Active bots | 18 (paper_mode_bots) |

**Top performers (paper):**
- PORTALUSDT: +10.74 USDT (9 cycles)
- VTHOUSDT: +7.00 USDT (11 cycles)
- EPICUSDT: +6.04 USDT (10 cycles)
- XLMUSDT: +3.72 USDT (12 cycles)

**Problematic:**
- NFPUSDT: -14.56 USDT — 28% asset dump in 2.5 hours, trailing stop triggered (-3.04 peak), then continued falling. Post-restart caught 8 more down cycles.
- Root cause: extreme market event, not a strategy bug
- **Mitigation:** `min_cycles_for_rank: 10` — NFPUSDT (8 cycles) would NOT enter Combat (real trading)

### Phase 4: Combat Launch (Real Trading)

**Configuration:**
- `paper_mode_bots:` 18
- `live_swarm`: ["PORTALUSDT", "VTHOUSDT"]
- Positions: ALREADY OPEN on Binance Futures
- **Real capital at risk**

**Reasoning:**
- Optuna-optimized stop-loss parameters validated on 7-day backtest
- Paper trading shows consistent profitability on volatile tickers
- `min_cycles_for_rank: 10` filters out unstable tickers (e.g. NFPUSDT)
- Combat + Incubator running in parallel

**⚠️ RISK NOTES:**
- First time trading real capital
- Only PORTALUSDT and VTHOUSDT in live_swarm initially
- Monitor for slippage, API latency, unexpected market behavior
- NFPUSDT-type events can happen — trailing stop protects but recovery is not guaranteed
