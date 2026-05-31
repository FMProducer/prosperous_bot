# CHANGELOG — Prosperous BOT Futures Portfolio

---

## [Unreleased] — 2026-06-03

### Net Move Guard (NMG) — Pump/Dump Protection

**Problem:** During fast unidirectional price movements (pump/dump), the hedge legs move in opposite directions. LONG surplus looks like profit, SHORT deficit looks like loss. Without protection, the bot sells LONG "profit" and buys SHORT "loss" — then the price reverts and the bot locked in a loss.

**Solution:** Added Net Move Guard — if price moves >1.5% in one direction within 30 seconds, ALL rebalance actions are blocked. The bot waits for stabilization before resuming.

**Changes:**

1. **main.py** — New guard in heartbeat loop (after Trend Guard, before Spread Guard):
   - Tracks price history with 300s window (existing deque)
   - Compares current price to price 30s ago
   - If move > 1.5% → `🛡️ Net Move Guard` + `continue` (skip heartbeat)
   - Log throttled: every 5th tick only

2. **config.json** — New parameters in `safety_guards`:
   - `net_move_block_pct`: 1.5 (percent, triggers block)
   - `net_move_window_sec`: 30 (lookback window)

3. **calculator.py** — `force_block` parameter added:
   - `calculate_rebalance()` and `calculate_deviations()` accept `force_block: bool`
   - When True, returns empty actions but preserves TPV/metrics
   - Used by NMG for clean separation (guard in loop, block in calculator)

**Protection layers for pump/dump:**
- Velocity Guard: >1% in 60s → block (existing)
- Trend Guard: >0.5% with >85% efficiency → block (existing)
- Net Move Guard: >1.5% in 30s → block (NEW)
- PnL Guard: hedge PnL < 0 → no surplus selling (previous)
- Combined: covers both fast spikes and sustained trends

---

### PnL Guard — Surplus Sell Block in Drawdown

**Problem:** When hedge portfolio is in drawdown (LONG PnL + SHORT PnL < 0), surplus selling on one side locks in profits while losses accumulate on the other side. Portfolio value melts.

**Solution:** Added `allow_surplus_sell` parameter to calculator. When `hedge_pnl < 0`, all surplus actions are blocked. Only deficit buying (if cash available) continues.

**Changes:**

1. **calculator.py**
   - `calculate_rebalance()`: new param `allow_surplus_sell` (default True)
   - `calculate_deviations()`: new param `allow_surplus_sell` (default True)
   - Phase 1 (surplus processing): if `allow_surplus_sell=False`, skip all surplus actions with `🛡️ PnL GUARD` log

2. **main.py**
   - Before `calculate_rebalance()`: compute `hedge_pnl = pnl_l + pnl_s` from calculator instance attributes
   - Pass `allow_surplus_sell = hedge_pnl >= 0.0` to calculator
   - Log active guard: `🛡️ PnL GUARD active: hedge PnL=X.XX < 0, surplus selling blocked`

3. **main.py** — Anti-spam fix
   - When VIRTUAL share=0 (disabled), skip "Initialized Virtual: 0.0 units" spam on every heartbeat
   - Only log initialization when VIRTUAL share > 0

### VIRTUAL OFF — Switch to Pure Long/Short Hedge
  
**Previous changes (same session):**

**Problem:** VIRTUAL leg (35% of TPV) caused parasitic rebalancing. During strong price movements, VIRTUAL surplus sold while LONG/SHORT deficits accumulated. VIRTUAL proceeds went to cash reserve, not helping hedge balance.

**Solution:** Disabled VIRTUAL entirely. Switched to 50/50 LONG/SHORT pure hedge.

**Changes:**

1. **config.json**
   - `BASE_LONG.share`: 0.29 → 0.50
   - `BASE_SHORT.share`: 0.36 → 0.50  
   - `VIRTUAL.share`: 0.35 → 0.0
   - `supervisor_interval_days`: 0.011 → 0.001 (~90s)
   - `toxic_blacklist`: cleared (PORTALUSDT expired entry removed)
   - `tickers`: expanded from 17 → 24 (added HBAR, MEME, NEAR, ONDO, RENDER, STG, TON, VIRTUAL, WLD, XMR; removed PENDLE)

2. **supervisor.py** — Rotation logic rewrite
   - Old: `to_keep = scored_old[:10]` + `to_add = new[:10]` → final could be <20 when scanner overlap was high
   - New: unified score-driven selection from ALL candidates (old + new), new capped at `max_replace`, sorted by score descending
   - Old bots with positive score always prioritized over new (score 0)
   - Result: always fills to `max_bots` when scanner has candidates

3. **calculator.py** (previous session, already applied)
   - `available_funds = max(Decimal('0'), self.val_cash)` — bot cash reserve only, NOT exchange margin
   - FUSE for deficit purchases DISABLED — no wrong-side-of-threshold blocking

**New parameters:**
   - `supervisor_interval_days`: 0.001 → 0.02 (~30 min, was ~90s)
   - `min_cycles_for_rank`: 10 → 3 (reduced from default, allows rotation after 3 cycles)

**Prevented:**
   - `config["tickers"]` is no longer overwritten by supervisor.py on each cycle
   - `config["base_ticker"]` is no longer overwritten by supervisor.py
   - These values are now set manually in config.json only

---

## [18 Apr 2026]

### Security & Safety Update
- API keys moved to env vars (BINANCE_API_KEY, BINANCE_SECRET_KEY)
- Margin ratio monitoring added (warning ≤5x, critical ≤2x)
- Hysteresis fix: `reference_tpd` as fixed baseline
- Added .env.example and .gitignore

## [30 May 2026]

### Liquidation Guard & Supervisor Auto-Restart
- Per-position liquidation distance monitoring (warn ≤15%, critical ≤8%)
- Supervisor auto-relaunch for missing real bots
