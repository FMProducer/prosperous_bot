"""Прогоняем все комбинации гвардов на ВСЕХ тикерах из tickers.txt.
Velocity всегда OFF (главный убытогенер).
Комбинации: T×N×P = 8 вариантов."""
import json, os, sys, logging, warnings
from collections import deque
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from dataclasses import field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculator import PortfolioCalculator
from backtest_rebalance import MarketOrderSlippageSimulator, quantize_qty, validate_notional

logging.basicConfig(level=logging.WARNING, format="%(message)s")
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PORTFOLIO = CONFIG["portfolios"][0]
TARGETS = PORTFOLIO["targets"]
INITIAL_CAP = float(PORTFOLIO.get("initial_capital", 180.0))
GUARDS = PORTFOLIO.get("safety_guards", {})
MAX_VEL = float(GUARDS.get("max_price_velocity_pct", 1.0)) / 100
VEL_WIN = int(GUARDS.get("velocity_window_sec", 60))
NMG_PCT = float(GUARDS.get("net_move_block_pct", 1.5)) / 100
NMG_WIN = int(GUARDS.get("net_move_window_sec", 30))
TS_PCT = CONFIG.get("equity_trailing_stop_pct", 0.0)
TS_ACT = CONFIG.get("equity_trailing_stop_activation_pct", 0.0)
TS_TOUT = int(CONFIG.get("equity_trailing_stop_timeout_sec", 60))
MIN_NOTIONAL = Decimal(str(CONFIG.get("min_notional_usdt", 6.0)))
LIQ_CRIT = PORTFOLIO.get("liquidation_distance_crit_pct", 8.0)
SURPLUS_T = float(PORTFOLIO.get("rebalance_threshold_surplus", 0.01))
L_LEV = Decimal(str(TARGETS["BASE_LONG"]["leverage"]))
S_LEV = Decimal(str(TARGETS["BASE_SHORT"]["leverage"]))
MMR = Decimal('0.004')
FUND_STEP = Decimal('0.0002') / Decimal('1440')
BARS_DAY = 1440
DAYS = 2.0
MAX_BARS = int(DAYS * BARS_DAY)


def run_ticker(ticker, enable_trend, enable_nmg, enable_pnl):
    """Бэктест одного тикера с Velocity=OFF и комбинацией T/N/P."""
    fp = os.path.join(DATA_DIR, f"{ticker}_live.feather")
    if not os.path.exists(fp):
        return None

    df = pd.read_feather(fp)[['open','high','low','close','volume']]
    if len(df) > MAX_BARS:
        df = df.iloc[-MAX_BARS:]
    prices = df['close'].values.astype(np.float64)
    if len(prices) < 10:
        return None

    # Per-ticker thresholds
    t_cfg = PORTFOLIO.get("ticker_thresholds", {}).get(ticker)
    ts = SURPLUS_T
    if isinstance(t_cfg, dict):
        ts = float(t_cfg.get("surplus", SURPLUS_T))

    step = 0.001
    sim = MarketOrderSlippageSimulator(0.0004, 0.0002)

    val_cash = Decimal(str(INITIAL_CAP))
    pos_l = Decimal('0')
    pos_s = Decimal('0')
    l_entry = Decimal('0')
    s_entry = Decimal('0')
    reb_l_entry = Decimal(str(prices[0]))
    l_lev = L_LEV
    s_lev = S_LEV

    tpv_ath = 0.0
    ts_start = 0.0
    ts_triggered = False
    liq_count = 0
    cycles = 0
    blocks_t = 0
    blocks_n = 0
    blocks_p = 0
    short_entry_price = Decimal('0')

    equity = [INITIAL_CAP]
    ph = deque()

    for i in range(len(prices)):
        mp = Decimal(str(prices[i]))
        now = float(i)
        ph.append((now, float(mp)))
        while ph and (now - ph[0][0]) > 300:
            ph.popleft()

        blocked = False

        # TREND GUARD
        if enable_trend and len(ph) > 10:
            pl = [p[1] for p in ph]
            nm = abs(pl[-1] - pl[0])
            tp = sum(abs(pl[j] - pl[j-1]) for j in range(1, len(pl)))
            eff = (nm / tp) if tp > 0 else 0
            if (nm / pl[0] > 0.005) and eff > 0.85:
                blocked = True
                blocks_t += 1

        # NMG
        if enable_nmg and not blocked and len(ph) > 1:
            nm = [(t,p) for t,p in ph if now - t <= NMG_WIN]
            if nm:
                _, np_ = nm[0]
                if np_ > 0 and abs(float(mp) - np_) / np_ > NMG_PCT:
                    blocked = True
                    blocks_n += 1

        if not blocked:
            allow_surplus = True
            if enable_pnl:
                ml = (pos_l * l_entry) / l_lev if pos_l > 0 else Decimal('0')
                ms = (pos_s * s_entry) / s_lev if pos_s > 0 else Decimal('0')
                wb = val_cash + ml + ms
                eq = float(wb)  # no virtual debt for 50/50
                c = PortfolioCalculator(
                    positions={f"{ticker}_LONG": float(pos_l), f"{ticker}_SHORT": float(pos_s)},
                    spot_price=float(mp), real_equity=eq,
                    virt_qty=0.0, virt_debt=0.0,
                    long_entry_price=float(l_entry), short_entry_price=float(s_entry),
                    base_ticker=ticker, targets=TARGETS,
                    initial_capital=INITIAL_CAP,
                    last_rebalance_price=float(reb_l_entry))
                if float(c.pnl_l) + float(c.pnl_s) < 0:
                    allow_surplus = False
                    blocks_p += 1
                cr = c.calculate_rebalance(TARGETS, ts, ts, allow_surplus_sell=allow_surplus)
            else:
                ml = (pos_l * l_entry) / l_lev if pos_l > 0 else Decimal('0')
                ms = (pos_s * s_entry) / s_lev if pos_s > 0 else Decimal('0')
                wb = val_cash + ml + ms
                eq = float(wb)
                c = PortfolioCalculator(
                    positions={f"{ticker}_LONG": float(pos_l), f"{ticker}_SHORT": float(pos_s)},
                    spot_price=float(mp), real_equity=eq,
                    virt_qty=0.0, virt_debt=0.0,
                    long_entry_price=float(l_entry), short_entry_price=float(s_entry),
                    base_ticker=ticker, targets=TARGETS,
                    initial_capital=INITIAL_CAP,
                    last_rebalance_price=float(reb_l_entry))
                cr = c.calculate_rebalance(TARGETS, ts, ts, allow_surplus_sell=True)

            actions = cr.get("actions", [])
            if actions:
                reds = [a for a in actions if a.get("is_reduction")]
                exps = [a for a in actions if not a.get("is_reduction")]

                for a in reds:
                    lev = Decimal(str(a["leverage"]))
                    qty = quantize_qty(abs(Decimal(str(a["diff_usdt"]))) / mp, Decimal(str(step)))
                    if qty <= 0:
                        continue
                    k = a["key"]
                    if k == "BASE_LONG" and pos_l > 0:
                        ep, cm = sim.simulate_market_execution("SELL", qty, mp)
                        rpnl = qty * (ep - l_entry)
                        rm = (qty * l_entry) / lev
                        pos_l -= qty
                        val_cash += rm + rpnl - cm
                        if pos_l == 0: l_entry = Decimal('0')
                    elif k == "BASE_SHORT" and pos_s > 0:
                        ep, cm = sim.simulate_market_execution("BUY", qty, mp)
                        rpnl = qty * (s_entry - ep)
                        rm = (qty * s_entry) / lev
                        pos_s -= qty
                        val_cash += rm + rpnl - cm
                        if pos_s == 0: s_entry = Decimal('0')

                exps.sort(key=lambda x: 0 if x["key"]=="VIRTUAL" else 1)
                rem = Decimal(str(cr.get("val_cash", 0.0)))
                for a in exps:
                    lev = Decimal(str(a["leverage"]))
                    need = abs(Decimal(str(a["diff_usdt"])))
                    if rem <= 0: continue
                    if need > rem * lev: need = rem * lev
                    qty = quantize_qty(need / mp, Decimal(str(step)))
                    if qty <= 0: continue
                    eq_sp = need / lev
                    k = a["key"]
                    if k == "BASE_LONG":
                        ep, cm = sim.simulate_market_execution("BUY", qty, mp)
                        mr = (qty * ep) / lev
                        val_cash -= mr + cm
                        l_entry = ((pos_l * l_entry) + (qty * ep)) / (pos_l + qty) if pos_l + qty > 0 else ep
                        pos_l += qty
                    elif k == "BASE_SHORT":
                        ep, cm = sim.simulate_market_execution("SELL", qty, mp)
                        mr = (qty * ep) / lev
                        val_cash -= mr + cm
                        s_entry = ((pos_s * s_entry) + (qty * ep)) / (pos_s + qty) if pos_s + qty > 0 else ep
                        pos_s += qty
                    rem -= eq_sp

                cycles += 1
                reb_l_entry = mp

        # FUNDING
        drag = (pos_l * mp + pos_s * mp) * FUND_STEP
        val_cash -= drag

        # LIQ LONG
        if pos_l > 0 and l_entry > 0:
            lp = l_entry * (Decimal('1') - Decimal('1')/l_lev + MMR)
            d = float((mp - lp) / mp * 100)
            if d <= LIQ_CRIT:
                m = (pos_l * l_entry) / l_lev
                up = pos_l * (mp - l_entry)
                val_cash += m + up
                pos_l = Decimal('0'); l_entry = Decimal('0'); liq_count += 1
        # LIQ SHORT
        if pos_s > 0 and s_entry > 0:
            lp = s_entry * (Decimal('1') + Decimal('1')/s_lev - MMR)
            d = float((lp - mp) / mp * 100)
            if d <= LIQ_CRIT:
                m = (pos_s * s_entry) / s_lev
                up = pos_s * (s_entry - mp)
                val_cash += m + up
                pos_s = Decimal('0'); s_entry = Decimal('0'); liq_count += 1

        # TRAILING STOP
        ml = (pos_l * l_entry) / l_lev if pos_l > 0 else Decimal('0')
        ms = (pos_s * s_entry) / s_lev if pos_s > 0 else Decimal('0')
        tpv = float(val_cash + ml + ms)

        if tpv > tpv_ath:
            tpv_ath = tpv
            ts_start = 0.0
        if TS_PCT > 0 and tpv_ath > INITIAL_CAP * (1 + TS_ACT / 100):
            dd = (1 - tpv / tpv_ath) * 100
            if dd >= TS_PCT:
                if ts_start == 0: ts_start = float(i)
                elif i - int(ts_start) >= max(1, TS_TOUT // 60):
                    ts_triggered = True
                    equity.append(tpv)
                    break
            else:
                ts_start = 0.0

        equity.append(tpv)

    eq = np.array(equity)
    profit = (eq[-1] / (INITIAL_CAP + 1e-9) - 1) * 100
    meq = np.maximum.accumulate(eq)
    dd = np.max((meq - eq) / (meq + 1e-9)) * 100 if len(eq) > 0 else 0
    return {"profit": profit, "dd": dd, "cycles": cycles, "liq": liq_count,
            "ts": ts_triggered, "T": blocks_t, "N": blocks_n, "P": blocks_p}


def main():
    tf = os.path.join(PROJECT_DIR, "tickers.txt")
    with open(tf, "r") as f:
        tickers = [l.strip() for l in f if l.strip()]

    combos = [
        ("V ON  T ON  N ON  P ON ", True,  True,  True ),
        ("V OFF T ON  N ON  P ON ", False, True,  True ),
        ("V OFF T OFF N ON  P ON ", False, False, True ),
        ("V OFF T ON  N OFF P ON ", False, True,  False),
        ("V OFF T OFF N OFF P ON ", False, False, False),
        ("V OFF T ON  N ON  P OFF", False, True,  True ),  # pnl=off -> False
        ("V OFF T OFF N ON  P OFF", False, False, True ),
        ("V OFF T ON  N OFF P OFF", False, True,  False),
        ("V OFF T OFF N OFF P OFF", False, False, False),
    ]
    # Fix: enable_pnl is separate
    combos = [
        ("ALL ON (baseline)",    True,  True,  True,  True),
        ("V OFF, rest ON",       False, True,  True,  True),
        ("V T OFF, N P ON",      False, False, True,  True),
        ("V N OFF, T P ON",      False, True,  False, True),
        ("V T N OFF, P ON",      False, False, False, True),
        ("V P OFF, T N ON",      False, True,  True,  False),
        ("V T P OFF, N ON",      False, False, True,  False),
        ("V N P OFF, T ON",      False, True,  False, False),
        ("V T N P ALL OFF",      False, False, False, False),
    ]

    print(f"Tickers: {len(tickers)}, Combos: {len(combos)}, Total: {len(tickers)*len(combos)} backtests")
    print(f"{'Combo':<25s}", end="")
    for t in tickers[:5]:
        print(f" {t:>10s}", end="")
    print(" ...")

    # Results: {combo_name: {ticker: profit}}
    all_results = {}
    for cname, vel, trend, nmg, pnl in combos:
        res = {}
        for t in tickers:
            r = run_ticker(t, trend, nmg, pnl)
            if r:
                res[t] = r["profit"]
        all_results[cname] = res

    # Print table
    header = f"{'Combo':<25s} {'AvgProfit':>10s} {'Profitable':>11s} {'Best':>12s} {'Worst':>12s}"
    print(f"\n{header}")
    print("-" * 75)

    baseline = all_results.get("ALL ON (baseline)", {})
    for cname, res in all_results.items():
        if not res: continue
        vals = list(res.values())
        avg = sum(vals) / len(vals)
        prof = sum(1 for v in vals if v > 0)
        best_t = max(res, key=res.get)
        worst_t = min(res, key=res.get)
        base_avg = sum(baseline.values())/len(baseline) if baseline else 0
        diff = avg - base_avg
        print(f"  {cname:<23s} {avg:>+9.2f}% {prof:>5d}/{len(vals):<5d} "
              f"{best_t:>8s}:{res[best_t]:>+6.2f}% {worst_t:>8s}:{res[worst_t]:>+6.2f}%  ({diff:+.2f}%)")

    # Per-ticker best combo
    print(f"\n{'Per-ticker best combo:':}")
    print("-" * 75)
    for t in tickers:
        best_c = max(all_results, key=lambda c: all_results[c].get(t, -999))
        best_p = all_results[best_c].get(t, 0)
        base_p = baseline.get(t, 0)
        print(f"  {t:<14s} best={best_c:<22s} profit={best_p:>+7.2f}%  (base={base_p:>+7.2f}%, diff={best_p-base_p:+.2f}%)")


if __name__ == "__main__":
    main()
