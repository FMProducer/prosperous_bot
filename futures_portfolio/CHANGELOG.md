CHANGELOG — Prosperous BOT Futures Portfolio

[14 Jul 2026 — 17:05] — State File Path Deduplication (SSOT for core/)
Problem: State-файлы (paper_state_*, paper_shadow_*, real_state_*, shadow_state_*) дублировались в корне futures_portfolio/ и в core/. Swarm_manager.py использовал BASE_PATH (корень) для чтения/записи, а main.py — os.path.dirname(__file__) (core/). Итого: 41 stale-файл в корне при 17 live-ботах.

Root cause: CORE_PATH не существовал. Все 10 ссылок в swarm_manager.py на state-файлы шли через BASE_PATH.

Fix:
- Добавлен CORE_PATH = BASE_PATH / "core" (swarm_manager.py:24)
- Переключены 10 ссылок: enforce_swarm_consistency, reset_bot_state_files, start_bot, get_bot_efficiency, amnesty-блок, get_running_bots_info
- 41 stale-файл перемещён из корня в history/
- Тесты: patched CORE_PATH в test_supervisor.py (1) + test_supervisor_coverage.py (6)

Result: 327/327 passed. State-файлы существуют ТОЛЬКО в core/. Источник дублирования устранён.

[14 Jul 2026 — 16:42] — Supervisor PM2 Log Spam Elimination
Problem: manage_swarm() запускает ~19 ботов за цикл. Каждый pm2 start/delete/save без stdout-редиректа выводил PM2 box-drawing таблицу в лог супервайзера. Итого ~400 строк мусора за один цикл.

Root cause: 6 точек в swarm_manager.py использовали create_subprocess_shell() без stdout=DEVNULL:
1. start_bot() — pm2 start (главный источник: 19 таблиц)
2. stop_bot() — pm2 delete
3. enforce_swarm_consistency → cmd_stop — закрытие stray-позиций
4. manage_swarm → cmd_stop #1 — Authoritative Cleanup
5. manage_swarm → cmd_stop #2 — Stopping Combat REAL
6. pm2 save — сохранение дампа

Fix: Все 6 вызовов дополнены stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL.
get_running_bots_info() и get_pm2_processes() оставлены с PIPE — они читают данные из pm2 jlist.

Result: 327/327 tests passed. Лог супервайзера теперь чистый — только бизнес-события.

[14 Jul 2026] — Этап 3: Модульная структура (Multi-Repo Split)
Problem: Плоская структура ~30 .py файлов в корне futures_portfolio/. Непрозрачная навигация, конфликт имён supervisor.py/supervisor/ на Windows, теневой src/ в .venv.

Solution: Модульная структура с поддиректориями core/, supervisor/, scanner/, optimization/, backtest/, monitoring/, tools/, scripts/.

Changes:
- supervisor.py → supervisor/swarm_manager.py (разрешение конфликта имён)
- src/ → core/ (устранение теневого src/ в .venv)
- Все import paths переведены на пакетный стиль (futures_portfolio.core.*, futures_portfolio.supervisor.*)
- Удалены sys.path.insert хаки из production-кода
- ecosystem.config.js: script paths обновлены (supervisor/supervisor_service.py, supervisor/aggregator.py, monitoring/telegram_sender.py)
- swarm_manager.py: start_bot запускает core/main.py (было main.py)
- dashboard/dashboard.py: PM2 команда обновлена на core/main.py
- BASE_PATH в swarm_manager.py: parent.parent (было parent)
- PROJECT_DIR в optimization/*.py: parent.parent (было parent)
- data_dir в backtest/*.py: dirname(dirname(...)) (было dirname(...))
- Все mock paths в тестах обновлены
- 24 тестовых файла, 318/320 passed

Status: Этап 3 завершён. Система готова к перезапуску с новой структурой. 327/327 passed.

[14 Jul 2026] — Баг-фиксы Этапа 3: stderr crash + 3 failing tests
Problem 1: Python 3.13 + pytest крашится на "lost sys.stderr" при сборке всех тестов.
Root cause: supervisor_service.py и aggregator.py перехватывали sys.stderr на module level.
Fix: sys.stderr/stdout перехват перенесён в if __name__ == "__main__".

Problem 2: test_generate_swarm_section — assert 0.0 == 10.0.
Root cause: pytest создаёт два отдельных module object (supervisor.aggregator и futures_portfolio.supervisor.aggregator). Mock патчит не тот.
Fix: patch paths изменены с futures_portfolio.supervisor.aggregator → supervisor.aggregator.

Problem 3: B1 тесты (test_b1_stop_bot_called, test_b1_ts_flag_reset) — stop_bot не вызывается.
Root cause: tracking_open mock перехватывал только WRITE операции. READ падал на original_builtin_open → FileNotFoundError → B1 ветка недостижима.
Fix: tracking_open перехватывает READ для real_state_* файлов, возвращает StringIO с mock данными.

Problem 4: test_supervisor_new.py — assert "pm2 start main.py" не найден.
Root cause: start_bot в swarm_manager.py обновлён на core/main.py, тест не обновлён.
Fix: assertion обновлён на "pm2 start core/main.py".

Result: 327 passed, 0 failed, 0 errors.

[14 Jul 2026] — PYTHONPATH для subprocess в swarm_manager.py
Problem: enforce_swarm_consistency вызывает core/main.py через subprocess.
Из temp-директории pytest модуль core/ не найден (ModuleNotFoundError).
В production PM2 --cwd обеспечивает правильный CWD, но subprocess脆弱ен.
Fix: все 3 subprocess-вызова теперь передают env с PYTHONPATH=BASE_PATH.
Result: 327/327, ModuleNotFoundError исчез.

[11 Jul 2026] — B6/B3 SSOT Refactor: Race Condition + Exit/Stop Isolation
Problem: main.py directly writes to config.json in _handle_liquidation_recovery and _handle_liquidation_guard (non-atomic write_text). Parallel processes (main.py + supervisor.py) cause race condition → data loss (toxic records, live_swarm changes). Additionally, supervisor.py treats exit and stop signals identically — profitable tickers get permanently blacklisted instead of probation.

Root cause: Violation of SSOT invariant (only supervisor.py should write config.json). No signal type differentiation in supervisor signal processing loop.

Solution: Three-layer refactoring:
- B6: Removed all config.json write blocks from main.py. Worker → Read-Only + Signal (emit_signal). Supervisor = single mutator.
- B3: Signal routing — stop → toxic_blacklist + black_list (toxic outcome), exit → probation_* (profitable outcome, soft cooldown).
- P1: black_list/live_swarm converted to set (O(1) lookup) on load, back to sorted list on save.
- P2: flag_file.unlink() → asyncio.to_thread (non-blocking I/O in event loop).
- P3: trailing_stop_paper_timeout_end from worker state replaced by probation_* in config.json (supervisor as single cooldown controller).

Changes:
main.py:160-179 — removed config.json write block from _handle_liquidation_recovery
main.py:267-297 — removed config.json write block from _handle_liquidation_guard
supervisor.py:565-640 — rewritten signal processing (stop→toxic+blacklist, exit→probation, async unlink)
supervisor.py:555-559 — P1: list→set conversion on config load
supervisor.py:1022-1025 — P1: set→sorted list on config save
supervisor.py:814-818 — P3: trailing_stop_paper_timeout_end → probation_* check
supervisor.py:147-157 — stop_bot: bare except:pass → specific exception handling (FileNotFoundError, OSError)
supervisor.py:232-234 — enforce_swarm_consistency: append → add (set compatibility)
supervisor.py:656-660 — probation pruning (expired entries cleanup)
config.json:110-111 — probation_paper and probation_real keys
tests/test_supervisor.py — 8 new tests + 1 updated (B3 routing, P1 set, P2 async, P3 probation)
tests/test_main.py — 2 new tests (B6 config write verification)

Test results: 30/30 passed (27 supervisor + 3 main). pre-existing timeout in test_rebalance_loop_trailing_stop (unrelated).
Status: SSOT pattern enforced. Race condition eliminated. Exit/Stop isolation restored.

[21 Jul 2026] — B1 Fix: TS Flag Persistence (zombie process kill)
Problem: After trailing stop triggers and supervisor restarts, the flag trailing_stop_triggered=True persists in real_state JSON. In enforce_swarm_consistency(), HEAL REJECTED path resets TS flags BEFORE Reaper Guard runs. Reaper Guard checks state file — flags already False — doesn't kill the zombie. Result: zombie PM2 process sleeps forever (while True: asyncio.sleep(86400)), never cleaned up.

Root cause: Call order in supervisor.py — enforce_swarm_consistency (line 559) runs BEFORE Reaper Guard (line 622+). HEAL REJECTED resets flags at line 200, Reaper Guard at line 635 can't see them anymore.

Solution: Added await stop_bot(ticker, is_paper=False) in HEAL REJECTED path (supervisor.py:197-199) — kills zombie PM2 process via pm2 delete BEFORE resetting TS flags. Atomic 1-line fix.

Changes:
supervisor.py:197-199 — +await stop_bot(ticker, is_paper=False) in HEAL REJECTED block
tests/test_supervisor.py — +2 unit tests (test_b1_stop_bot_called_before_ts_flag_reset, test_b1_ts_flag_reset_after_stop_bot)

Test results: 19/19 passed (17 existing + 2 new B1 tests)
Status: B1 Score 8.3 → ✅ ИСПРАВЛЕН. Активируется при включении TS.

[21 Jul 2026] — Documentation sync (ROADMAP, STATUS, PROJECT_INDEX)
Problem: ROADMAP.md described "1 REAL bot (INJUSDT)" — actual: 4 bots (GRASS, UNI, VVV, YFI). STATUS.md had stale tickers, PROJECT_INDEX.md had wrong config values.

Solution: Updated all three documents to match actual config.json state (SSOT).

Changes:
ROADMAP.md — Текущее состояние: 4 REAL bots, 20 USDT/bot, ~7 min interval, TS OFF
STATUS.md — Live Swarm: GRASS/UNI/VVV/YFI, TS=0.001% (OFF), guards=6 active
PROJECT_INDEX.md — Config params, tickers, changelog entries

[13 Jul 2026] — Test Coverage Phase 1: Prod Scripts ≥89%
Problem: Test coverage at 34.2% overall. 3 FAIL tests blocking progress. rank_tickers.py at 22%, telegram_sender.py at 81%, send_monitor_report.py at 0%.

Solution: Fix 3 FAIL tests, then expand coverage for all prod-critical scripts.

Phase 0 (prior session): 15 dead test files removed, capsys bugs fixed, bounded_sleep→async (6 instances), pyproject.toml coverage omit updated.

Phase 1 (this session):
- Fixed test_coverage_paper_cross_margin_check (infinite loop via bounded_sleep continue@L827)
- Fixed test_coverage_liquidation_guard_missing_long: get_positions mock returned qty as string "-1.0" instead of float -1.0 (TypeError → exception → guard never reached)
- Fixed test_coverage_emergency_stop_full: saves filter "state.json" in s[0] matched "paper_state.json" (substring) → state_save was paper_state dict without virt_qty
- telegram_sender.py: 81% → 100% (+5 tests: rate limit invalid JSON, timeout, connection error, missing credentials, empty queue, send failure, success path, API error)
- rank_tickers.py: 22% → 100% (+17 tests: retry decorator exhaustion, TickerRanker init/metrics/momentum, run_ranker_task, fetch non-200/exception, fetch_klines too few/empty, get_top_tickers blacklist/whitelist/volume/non-ascii/empty_dfs, main quiet/not-quiet/with-config)
- send_monitor_report.py: 0% → 100% (+6 tests: no state files, paper+real, guard active, old heartbeat, sends request, no alerts)

Test results: 333 collected, 331 passed, 2 pre-existing FAIL (test_supervisor B1 TS flag — unrelated).
Coverage: Prod scripts 93% (2823 stmts, 195 miss). All 14 prod scripts ≥89%.

[21 Jul 2026] — Hermes Agent upgrade 0.17.0 → 0.18.2
Portable USB venv (D:\Hermes-USB-Portable-main\data\hermes-agent\venv) upgraded.
Config migrated: v24 → v33.
Key changes: stream-stale circuit breaker, PTY session management, approval gate fixes.

[11 Jul 2026] — Per-ticker min_notional + 2% buffer
Problem: Global min_notional_usdt: 6.1 blocked GRASSUSDT (Binance min=5.03) and VVVUSDT (5.05) — they couldn't rebalance at 3-4% deviations. Config set a single threshold for all tickers, though Binance minimums differ per ticker.

Solution: Added per-ticker min_notional lookup from exchange_info with 2% buffer for dynamic Binance minimum changes.

Changes:
main.py:489 — min_notionals extraction (Binance Futures: f.get("notional"), SPOT: f.get("minNotional"))
main.py:941-945 — active_min_notional = max(config_min, exchange_min * 1.02)
config.json — min_notional_usdt: 6.1 → 5.1
tests/test_main.py — 6 unit tests (TestMinNotionalExtraction, TestEffectiveMinNotional)

Effective minimums:
GRASSUSDT: max(5.1, 5.03*1.02) = 5.13
VVVUSDT: max(5.1, 5.05*1.02) = 5.15
YFIUSDT: max(5.1, 7.00*1.02) = 7.14
UNIUSDT: max(5.1, 7.05*1.02) = 7.19

Fallback (no exchange info): config = 5.1 (no buffer)

Bug fix: KeyError 'minNotional' — Binance Futures API uses 'notional' field, not 'minNotional'. Fixed with f.get("minNotional") or f.get("notional").

[Unreleased]
Net Move Guard (NMG) — Pump/Dump Protection
Problem: During fast unidirectional price movements (pump/dump), the hedge legs move in opposite directions. LONG surplus looks like profit, SHORT deficit looks like loss. Without protection, the bot sells LONG "profit" and buys SHORT "loss" — then the price reverts and the bot locked in a loss.

Solution: Added Net Move Guard — if price moves >1.5% in one direction within 30 seconds, ALL rebalance actions are blocked. The bot waits for stabilization before resuming.

Changes:

main.py — New guard in heartbeat loop (after Trend Guard, before Spread Guard):

Tracks price history with 300s window (existing deque)
Compares current price to price 30s ago
If move > 1.5% → 🛡️ Net Move Guard + continue (skip heartbeat)
Log throttled: every 5th tick only
config.json — New parameters in safety_guards:

net_move_block_pct: 1.5 (percent, triggers block)
net_move_window_sec: 30 (lookback window)
calculator.py — force_block parameter added:

calculate_rebalance() and calculate_deviations() accept force_block: bool
When True, returns empty actions but preserves TPV/metrics
Used by NMG for clean separation (guard in loop, block in calculator)
Protection layers for pump/dump:

Velocity Guard: >1% in 60s → block (existing)
Trend Guard: >0.5% with >85% efficiency → block (existing)
Net Move Guard: >1.5% in 30s → block (NEW)
PnL Guard: hedge PnL < 0 → no surplus selling (previous)
Combined: covers both fast spikes and sustained trends
PnL Guard — Surplus Sell Block in Drawdown
Problem: When hedge portfolio is in drawdown (LONG PnL + SHORT PnL < 0), surplus selling on one side locks in profits while losses accumulate on the other side. Portfolio value melts.

Solution: Added allow_surplus_sell parameter to calculator. When hedge_pnl < 0, all surplus actions are blocked. Only deficit buying (if cash available) continues.

Changes:

calculator.py

calculate_rebalance(): new param allow_surplus_sell (default True)
calculate_deviations(): new param allow_surplus_sell (default True)
Phase 1 (surplus processing): if allow_surplus_sell=False, skip all surplus actions with 🛡️ PnL GUARD log
main.py

Before calculate_rebalance(): compute hedge_pnl = pnl_l + pnl_s from calculator instance attributes
Pass allow_surplus_sell = hedge_pnl >= 0.0 to calculator
Log active guard: 🛡️ PnL GUARD active: hedge PnL=X.XX < 0, surplus selling blocked
main.py — Anti-spam fix

When VIRTUAL share=0 (disabled), skip "Initialized Virtual: 0.0 units" spam on every heartbeat
Only log initialization when VIRTUAL share > 0
VIRTUAL OFF — Switch to Pure Long/Short Hedge
Previous changes (same session):

Problem: VIRTUAL leg (35% of TPV) caused parasitic rebalancing. During strong price movements, VIRTUAL surplus sold while LONG/SHORT deficits accumulated. VIRTUAL proceeds went to cash reserve, not helping hedge balance.

Solution: Disabled VIRTUAL entirely. Switched to 50/50 LONG/SHORT pure hedge.

Changes:

config.json

BASE_LONG.share: 0.29 → 0.50
BASE_SHORT.share: 0.36 → 0.50
VIRTUAL.share: 0.35 → 0.0
supervisor_interval_days: 0.011 → 0.001 (~90s)
toxic_blacklist: cleared (PORTALUSDT expired entry removed)
tickers: expanded from 17 → 24 (added HBAR, MEME, NEAR, ONDO, RENDER, STG, TON, VIRTUAL, WLD, XMR; removed PENDLE)
supervisor.py — Rotation logic rewrite

Old: to_keep = scored_old[:10] + to_add = new[:10] → final could be <20 when scanner overlap was high
New: unified score-driven selection from ALL candidates (old + new), new capped at max_replace, sorted by score descending
Old bots with positive score always prioritized over new (score 0)
Result: always fills to max_bots when scanner has candidates
calculator.py (previous session, already applied)

available_funds = max(Decimal('0'), self.val_cash) — bot cash reserve only, NOT exchange margin
FUSE for deficit purchases DISABLED — no wrong-side-of-threshold blocking
New parameters:

supervisor_interval_days: 0.001 → 0.02 (~30 min, was ~90s)
min_cycles_for_rank: 10 → 3 (reduced from default, allows rotation after 3 cycles)
Prevented:

config["tickers"] is no longer overwritten by supervisor.py on each cycle
config["base_ticker"] is no longer overwritten by supervisor.py
These values are now set manually in config.json only
[18 Apr 2026]
Security & Safety Update
API keys moved to env vars (BINANCE_API_KEY, BINANCE_SECRET_KEY)
Margin ratio monitoring added (warning ≤5x, critical ≤2x)
Hysteresis fix: reference_tpd as fixed baseline
Added .env.example and .gitignore
[13 Jul 2026] — Coverage Push: supervisor.py 66%→95%, main.py 65%→77%
Problem: Test coverage below 85% fail_under threshold. supervisor.py at 66%, main.py at 65%.

Root cause: integration tests for manage_swarm() not covering signal processing, rotation, scoring, and reaper logic. main.py tests not covering trailing stop closure, emergency stop function, liquidation guard, and blacklist rebase paths.

Solution: Two-phase coverage push targeting highest-impact uncovered blocks.

Phase 1 — supervisor.py (66% → 95%):
- Rewrote all manage_swarm() integration tests with proper CONFIG_PATH monkeypatch (module-level constant at line 52 loads config from disk, not from test locals)
- Fixed 5 failing tests: replaced_unprofitable (drawdown protection via inf score), real_stop_drops_bot (whitelist filter), authoritative_cleanup_stray (early return on empty scanner), safety_trim (whitelist guard)
- Added tests for: signal processing (stop→toxic, exit→probation), scoring pipeline, rotation with candidates, reaper guard, amnesty logic

Phase 2 — main.py (65% → 77%):
- Created test_main_coverage.py with 21 tests targeting: trailing stop trigger + closure (lines 978-1067), emergency stop / max drawdown (931-957), liquidation guard (1421-1460), VIRTUAL_ORDER accounting (1191-1233), stop/close logic (1525-1601), paper cross-margin check (814-827), blacklist rebase (598-625), emergency_stop function (1561-1601), _handle_liquidation_recovery (130-148)
- Key pattern: rebalance_loop() creates its own BinanceConnector internally but accepts connector param; PortfolioCalculator must be mocked to control tpv_total for trailing stop / drawdown activation

Changes:
tests/test_supervisor.py — 18 integration tests added (manage_swarm signal processing, scoring, rotation, reaper, amnesty)
tests/test_main_coverage.py — 21 tests added (trailing stop, emergency stop, liquidation guard, VIRTUAL_ORDER, emergency_stop function)
pyproject.toml — unchanged (fail_under=85, addopts with -p no:capture)
tests/test_supervisor_coverage.py — 377 lines (unit-level tests for get_pm2_processes, reconcile_swarm_state, stop_bot, enforce_invariant_gate)

Test results: 72/72 supervisor + main tests pass. Coverage: supervisor.py 95.08%, main.py 77.14%.

Remaining main.py gaps (need +8%): liquidation guard get_position_risk flow (1421-1460), trailing stop position closure real-mode path (994-1009), stop/close logic in emergency_stop function (1532-1601), blacklist rebase profit path (616-625).
[30 May 2026]
Liquidation Guard & Supervisor Auto-Restart
Per-position liquidation distance monitoring (warn ≤15%, critical ≤8%)
