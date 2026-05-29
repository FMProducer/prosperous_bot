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
