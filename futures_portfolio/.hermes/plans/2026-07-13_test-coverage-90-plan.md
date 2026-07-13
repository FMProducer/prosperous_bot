# Plan: Test Coverage >90% — Prioritized Implementation

**Date**: 2026-07-13
**Goal**: Achieve >90% test coverage across all production scripts
**Current**: 34.2% (1643/4802 stmts covered)
**Target**: ≥90% (need 4322/4802 stmts covered, i.e. cover 2679 of 3159 missing lines)

---

## Current State

### Coverage by File (from `coverage report --show-missing`)

| File | Stmts | Miss | Cov% | Status |
|------|-------|------|------|--------|
| calculator.py | 124 | 0 | 100% | ✅ |
| connector.py | 193 | 0 | 100% | ✅ |
| storage.py | 50 | 2 | 96% | ✅ |
| health_check.py | 59 | 3 | 95% | ✅ |
| aggregator.py | 126 | 9 | 93% | ✅ |
| notifier.py | 135 | 10 | 93% | ✅ |
| executor.py | 273 | 29 | 89% | ✅ |
| telegram_sender.py | 62 | 12 | 81% | ⚠️ |
| main.py | 901 | 311 | 65% | ❌ |
| rank_tickers.py | 155 | 121 | 22% | ❌ |
| supervisor.py | 590 | 533 | 10% | ❌ |
| swarm_analyzer.py | 58 | 53 | 9% | ❌ |
| supervisor_service.py | 46 | 46 | 0% | ❌ |
| send_monitor_report.py | 51 | 51 | 0% | ❌ |
| download_30d.py | 108 | 108 | 0% | ❌ |

### Scenario Analysis

- Cover Tier1+Tier2 (critical+support production) → **59.9%**
- Cover Tier1+Tier2+Tier3 (add offline tools) → **86.8%**
- Cover ALL → **100%**

**Conclusion**: To hit 85% we MUST cover Tier1+Tier2+Tier3. Tier4 (utilities) can be excluded from coverage via `pyproject.toml` omit list.

---

## Strategy

### Phase 0: Coverage Config Cleanup (1 session)
**Why first**: Without proper omit/exclude rules, utility scripts drag down the total.

1. Update `pyproject.toml` `[tool.coverage.run]` omit list:
   ```toml
   omit = [
       "tests/*",
       "backtest_rebalance.py",
       "__pycache__/*",
       "test_dd.py",
       "*.py_time.py",
       # Tier 4: utilities (one-off, not production)
       "debug_scan.py",
       "visualize_bot_log.py",
       "equity_visualizer.py",
       "voice_recognizer.py",
       "full_reset.py",
       "save_cache.py",
       "run_debug.py",
       "check_json_encoding.py",
       "check_market_data.py",
       "update_white_list.py",
   ]
   ```
2. Remove stale/orphan test files (0 tests, dead code):
   - `tests/test_exact.py`, `tests/test_fetch.py`, `tests/test_fetch2.py`
   - `tests/test_fix.py`, `tests/test_pattern.py`, `tests/test_risk_engine.py`
   - `tests/test_scan_v2.py`, `tests/test_scanner.py`, `tests/test_scoring_logic.py`
   - `tests/test_siphoning.py`, `tests/test_import.py`, `tests/test_surplus_first.py`
3. Fix 2 ERROR tests in test_swarm_analyzer.py
4. Fix 1 ERROR test in test_health_check.py (test_main)
5. Fix hanging test `test_coverage_paper_cross_margin_check` in test_main_coverage.py
6. Verify all tests pass: `pytest -q`

**Estimated coverage after Phase 0**: ~55-60% (same absolute, but denominator shrinks)

### Phase 1: main.py (65% → 90%) — HIGHEST IMPACT
**Why**: 901 stmts, 311 missing. Single biggest gap. Core trading loop.

**Missing lines analysis** (311 lines, grouped by function):
- Lines 122-148: `sync_read_json`, `emit_signal` edge cases → ~27 lines
- Lines 201-218: `_apply_clean_slate` paths → ~18 lines
- Lines 253-278: Hysteresis logic paths → ~10 lines
- Lines 343-370: Config validation, portfolio setup → ~37 lines
- Lines 463-466: `self_kill_pm2` error paths → ~4 lines
- Lines 515-533: Paper margin check, initial setup → ~19 lines
- Lines 550-665: Rebalance loop inner paths (position loading, PnL calc, virtual leg) → ~115 lines
- Lines 680-700: Rebalance threshold, anti-churn → ~21 lines
- Lines 781-848: Paper mode rebalance, cross-margin, virtual order processing → ~68 lines
- Lines 883-974: Trailing stop, emergency stop, liquidation guard → ~92 lines
- Lines 1067-1079: Config reload detection → ~13 lines
- Lines 1191-1233: `emergency_stop` function → ~43 lines
- Lines 1240-1601: Main block, CLI parsing, loop orchestration → ~162 lines

**Test approach**:
- Fix existing hanging tests first (test_main_coverage.py has 1 hang + 1 fail)
- Add targeted tests for uncovered branches
- Mock-heavy approach: mock BinanceConnector, PortfolioCalculator, asyncio.sleep
- Key: each test covers a specific uncovered code block

**Estimated tests needed**: ~25-30 new test functions

### Phase 2: supervisor.py (10% → 90%) — SECOND HIGHEST
**Why**: 590 stmts, 533 missing. Critical for bot management.

**Missing lines analysis** (533 lines, grouped by function):
- Lines 32-73: `setup_logger`, `get_running_bots_info` → ~42 lines
- Lines 82-94: `get_pm2_processes` → ~13 lines
- Lines 103-145: `reconcile_swarm_state` → ~43 lines
- Lines 148-155: `stop_bot` → ~8 lines
- Lines 165-251: `enforce_swarm_consistency` (partially tested) → ~87 lines
- Lines 258-282: `enforce_invariant_gate` → ~25 lines
- Lines 291-361: `reset_bot_state_files` → ~71 lines
- Lines 364-405: `start_bot` → ~42 lines
- Lines 424-501: `selective_merge_incubator` (partially tested) → ~78 lines
- Lines 509-523: `_calc_rotation_score` edge cases → ~15 lines
- Lines 531-553: `get_bot_efficiency` (partially tested) → ~23 lines
- Lines 555-1040: `manage_swarm` (THE BIG ONE — 486 lines, only ~57 tested) → ~430 lines → reduce to ~100 after fixing existing tests
- Lines 1050-1064: `_ensure_real_bots_alive` → ~15 lines

**Test approach**:
- Existing test_supervisor.py has 46 tests but many are shallow
- Need deep mocking: PM2 subprocess calls, file I/O, connector
- `manage_swarm()` is 486 lines — needs ~15-20 test cases for different paths
- Focus on: rotation logic, incubator merge, invariant gate, real bot alive check

**Estimated tests needed**: ~35-40 new test functions

### Phase 3: rank_tickers.py (22% → 90%)
**Why**: 155 stmts, 121 missing. Scanner selection.

**Missing lines**: Lines 25-231 (almost everything except basic imports and helpers)
**Test approach**: Mock HTTP requests (aiohttp), test filter logic, SPIKE/NET trap detection

### Phase 4: Support Files (Tier 2)
- **telegram_sender.py**: 81% → 90% (12 lines: error paths, retry logic)
- **download_30d.py**: 0% → 90% (108 lines: data download, file saving)
- **supervisor_service.py**: 0% → 90% (46 lines: service loop, sleep interval)
- **send_monitor_report.py**: 0% → 90% (51 lines: report generation, send)
- **swarm_analyzer.py**: 9% → 90% (53 lines: swarm analysis logic)

### Phase 5: Offline Tools (Tier 3)
These are Optuna optimizers and backtest runners. Less critical but needed for >90%.

- **optimize_trend_guard.py**: 0% → 85% (229 lines)
- **optimize_takeprofit_stop.py**: 0% → 85% (219 lines)
- **optimize_trailing_stop.py**: 0% → 85% (189 lines)
- **optimize_stops.py**: 0% → 85% (176 lines)
- **optimize_threshold.py**: 0% → 85% (118 lines)
- **optimize_config.py**: 0% → 85% (109 lines)
- **phase1_validate.py**: 0% → 85% (129 lines)
- **run_all_backtests.py**: 0% → 85% (53 lines)
- **run_backtest_30d.py**: 0% → 85% (69 lines)

---

## Critical Safety Invariants

### For ALL test development:
1. **NO API KEYS in tests** — all Binance calls mocked
2. **NO real state.json writes** — always use temp files or mocks
3. **NO real PM2 calls** — always mock subprocess
4. **Each atomic change = unit test first** (memory rule from 2026-06-21 incident)
5. **Hanging tests are BLOCKERS** — fix before proceeding

### For main.py tests:
- Mock `BinanceConnector` for all API calls
- Mock `asyncio.sleep` to prevent real waits
- Use `bounded_sleep` pattern (existing tests show this)
- Test config validation separately from rebalance loop
- `rebalance_loop` tests need careful mock orchestration

### For supervisor.py tests:
- Mock `asyncio.create_subprocess_exec` for PM2
- Mock `safe_load_json_sync` / `safe_save_json_sync` for state
- Mock `BinanceConnector` for exchange calls
- Test `manage_swarm` in isolation — it's 486 lines

---

## Execution Order

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼
Config   main.py   super-   rank_     support   offline
cleanup  (901)     visor    tickers   files     tools
(1 sess) (3 sess)  (590)    (155)     (2 sess)  (3 sess)
                    (4 sess) (1 sess)
```

**Total estimated sessions**: ~12-15 sessions
**Each session**: Write tests → Run tests → Verify coverage increment → Obsidian update

---

## Verification Protocol

After each phase:
1. `python -m pytest tests/ -q --tb=short` — all tests pass
2. `python -m coverage run ... && python -m coverage report --show-missing` — coverage check
3. Update this plan with actual coverage numbers
4. Obsidian: log progress in project docs

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hanging tests block CI | Fix root cause immediately, don't skip |
| Mock complexity in manage_swarm | Break into smaller testable functions if needed |
| coverage.py config conflicts with pyproject.toml | Use `coverage run` directly, not pytest-cov |
| Test isolation (state files) | Use tmp_path fixture, never touch real state.json |
| Real account safety | All tests run in isolation, no env vars loaded |

---

## Open Questions

1. Should Tier3 (offline tools) be excluded from coverage instead of tested?
   - **Pro**: Saves ~4 sessions of work
   - **Con**: Won't reach 95%+
   - **Recommendation**: Test them — they're small and mostly pure functions

2. Should we refactor `manage_swarm` (486 lines) into smaller functions?
   - **Pro**: Easier to test, better code quality
   - **Con**: Touches production code, risk of regression
   - **Recommendation**: Only if testing proves impractical
