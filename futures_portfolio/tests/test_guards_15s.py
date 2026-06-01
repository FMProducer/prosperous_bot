"""Бэктест с 15-секундным интервалом (интерполяция из 1-минутных свей).
Все 9 комбинаций гвардов (Velocity всегда OFF, варируем T/N/P)."""
import json, os, sys, logging, warnings
from collections import deque
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from typing import List, Optional
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
VIRT_SHARE = Decimal(str(TARGETS.get("VIRTUAL", {}).get("share", 0.0)))
SEC_PER_BAR = 15  # 15-секундный интервал
BARS_PER_MIN = 60 // SEC_PER_BAR  # 4 бара на минуту (но мы интерполируем 9)


def interpolate_1min_to_15s(prices_1min):
    """Линейная интерполяция: 1 минута → 9 точек по 15с (включая первую)."""
    result = []
    for i in range(len(prices_1min) - 1):
        p0 = prices_1min[i]
        p1 = prices_1min[i + 1]
        for j in range(6):  # 0,1,2,3,4,5 → 6 точек на интервал (каждые 10с для простоты)
            frac = j / 6.0
            result.append(p0 + (p1 - p0) * frac)
    result.append(prices_1min[-1])
    return np.array(result)


def run_ticker(ticker, enable_velocity, enable_trend, enable_nmg, enable_pnl):
    """Бэктест одного тикера с 15-секундным интервалом."""
    fp = os.path.join(DATA_DIR, f"{ticker}_live.feather")
    if not os.path.exists(fp):
        return None

    df = pd.read_feather(fp)[['open','high','low','close','volume']]
    raw_prices = df['close'].values.astype(np.float64)

    # Интерполируем до ~10-секундных точек
    prices = interpolate_15s_from_1m(raw_prices)
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
    reb_price = Decimal(str(prices[0]))

    # Virtual leg
    if VIRT_SHARE > 0:
        vq = (VIRT_SHARE * INITIAL_CAP) / float(prices[0])
        virt_qty = vq
        virt_debt = float(Decimal(str(VIRT_SHARE)) * Decimal(str(INITIAL_CAP)))
    else:
        virt_qty = 0.0
        virt_debt = 0.0

    tpv_ath = 0.0
    ts_start = 0.0
    ts_triggered = False
    liq_count = 0
    cycles = 0
    blocks_v = 0
    blocks_t = 0
    blocks_n = 0
    blocks_p = 0
    total_bars = len(prices)

    equity = [INITIAL_CAP]
    ph = deque()  # (sec_offset, price)

    for i in range(total_bars):
        mp = Decimal(str(prices[i]))
        now_sec = float(i * 10)  # каждая точка = 10 секунд

        ph.append((now_sec, float(mp)))
        while ph and (now_sec - ph[0][0]) > 300:
            ph.popleft()

        blocked = False

        # 1. VELOCITY GUARD (цена за VEL_WIN секунд назад)
        if enable_velocity and len(ph) > 1:
            cutoff = now_sec - VEL_WIN
            old_pts = [p for t, p in ph if t <= cutoff]
            if not old_pts:
                old_pts = [ph[0][1]]
            old_p = old_pts[0]
            if old_p > 0:
                vel = abs(float(mp) - old_p) / old_p
                if vel > MAX_VEL:
                    blocked = True
                    blocks_v += 1

        # 2. TREND GUARD
        if enable_trend and not blocked and len(ph) > 10:
            pl = [p[1] for p in ph]
            nm = abs(pl[-1] - pl[0])
            tp = sum(abs(pl[j] - pl[j-1]) for j in range(1, len(pl)))
            eff = (nm / tp) if tp > 0 else 0
            if (nm / pl[0] > 0.005) and eff > 0.85:
                blocked = True
                blocks_t += 1

        # 3. NMG
        if enable_nmg and not blocked and len(ph) > 1:
            cutoff = now_sec - NMG_WIN
            nm_pts = [p for t, p in ph if t <= cutoff]
            if not nm_pts:
                nm_pts = [ph[0][1]]
            np_ = nm_pts[0]
            if np_ > 0 and abs(float(mp) - np_) / np_ > NMG_PCT:
                blocked = True
                blocks_n += 1

        if not blocked:
            # Вычисляем PnL для guard
            ml = (pos_l * l_entry) / L_LEV if pos_l > 0 else Decimal('0')
            ms = (pos_s * s_entry) / S_LEV if pos_s > 0 else Decimal('0')
            upl_l = pos_l * (mp - l_entry) if pos_l > 0 else Decimal('0')
            upl_s = pos_s * (s_entry - mp) if pos_s > 0 else Decimal('0')
            wb = val_cash + ml + ms

            allow_surplus = True
            if enable_pnl:
                c = PortfolioCalculator(
                    positions={f"{ticker}_LONG": float(pos_l), f"{ticker}_SHORT": float(pos_s)},
                    spot_price=float(mp), real_equity=float(wb),
                    virt_qty=virt_qty, virt_debt=virt_debt,
                    long_entry_price=float(l_entry), short_entry_price=float(s_entry),
                    base_ticker=ticker, targets=TARGETS,
                    initial_capital=INITIAL_CAP,
                    last_rebalance_price=float(reb_price))
                if float(c.pnl_l) + float(c.pnl_s) < 0:
                    allow_surplus = False
                    blocks_p += 1
                cr = c.calculate_rebalance(TARGETS, ts, ts, allow_surplus_sell=allow_surplus)
            else:
                c = PortfolioCalculator(
                    positions={f"{ticker}_LONG": float(pos_l), f"{ticker}_SHORT": float(pos_s)},
                    spot_price=float(mp), real_equity=float(wb),
                    virt_qty=virt_qty, virt_debt=virt_debt,
                    long_entry_price=float(l_entry), short_entry_price=float(s_entry),
                    base_ticker=ticker, targets=TARGETS,
                    initial_capital=INITIAL_CAP,
                    last_rebalance_price=float(reb_price))
                cr = c.calculate_rebalance(TARGETS, ts, ts, allow_surplus_sell=True)

            actions = cr.get("actions", [])
            if actions:
                reds = [a for a in actions if a.get("is_reduction")]
                exps = [a for a in actions if not a.get("is_reduction")]

                for a in reds:
                    lev = Decimal(str(a["leverage"]))
                    raw_q = abs(Decimal(str(a["diff_usdt"]))) / mp
                    qty = quantize_qty(raw_q, Decimal(str(step)))
                    if qty <= 0: continue
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
                    elif k == "VIRTUAL" and virt_qty > 0:
                        ep, cm = sim.simulate_market_execution("SELL", qty, mp)
                        v_cost = virt_debt / Decimal(str(virt_qty)) if virt_qty > 0 else ep
                        alloc_debt = qty * v_cost
                        rpnl = qty * (ep - v_cost)
                        virt_qty -= float(qty)
                        virt_debt -= float(alloc_debt)
                        val_cash += float(alloc_debt) + float(rpnl) - float(cm)

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
                    elif k == "VIRTUAL":
                        ep, cm = sim.simulate_market_execution("BUY", qty, mp)
                        cs = float(qty * ep) + float(cm)
                        val_cash -= cs
                        virt_qty += float(qty)
                        virt_debt += cs
                    rem -= eq_sp

                cycles += 1
                reb_price = mp

        # FUNDING (каждый 10-секундный бар → пропорционально)
        fund_per_bar = FUND_STEP / Decimal('6.0')
        drag = (pos_l * mp + pos_s * mp) * fund_per_bar
        val_cash -= drag

        # LIQ
        if pos_l > 0 and l_entry > 0:
            lp = l_entry * (Decimal('1') - Decimal('1')/L_LEV + MMR)
            d = float((mp - lp) / mp * 100)
            if d <= LIQ_CRIT:
                m = (pos_l * l_entry) / L_LEV
                up = pos_l * (mp - l_entry)
                val_cash += m + up
                pos_l = Decimal('0'); l_entry = Decimal('0'); liq_count += 1

        if pos_s > 0 and s_entry > 0:
            lp = s_entry * (Decimal('1') + Decimal('1')/S_LEV - MMR)
            d = float((lp - mp) / mp * 100)
            if d <= LIQ_CRIT:
                m = (pos_s * s_entry) / S_LEV
                up = pos_s * (s_entry - mp)
                val_cash += m + up
                pos_s = Decimal('0'); s_entry = Decimal('0'); liq_count += 1

        # TRAILING STOP
        ml = (pos_l * l_entry) / L_LEV if pos_l > 0 else 0
        ms = (pos_s * s_entry) / S_LEV if pos_s > 0 else 0
        tpv = float(val_cash + ml + ms)

        if tpv > tpv_ath:
            tpv_ath = tpv
            ts_start = 0.0
        if TS_PCT > 0 and float(tpv_ath) > INITIAL_CAP * (1 + TS_ACT / 100):
            dd = (1 - tpv / tpv_ath) * 100
            if dd >= TS_PCT:
                if ts_start == 0: ts_start = now_sec
                elif now_sec - ts_start >= TS_TOUT:
                    ts_triggered = True
                    equity.append(tpv)
                    break
            else:
                ts_start = 0.0

        equity.append(tpv)

    eq = np.array(equity)
    profit = (eq[-1] / (INITIAL_CAP + 1e-9) - 1) * 100
    meq = np.maximum.accumulate(eq)
    dd = float(np.max((meq - eq) / (meq + 1e-9))) * 100 if len(eq) > 0 else 0
    return {"profit": profit, "dd": dd, "cycles": cycles, "liq": liq_count,
            "ts": ts_triggered, "V": blocks_v, "T": blocks_t, "N": blocks_n, "P": blocks_p}


def interpolate_15s_from_1m(prices_1min):
    """Интерполяция 1-минутных свечей в ~10-секундные точки."""
    result = []
    for i in range(len(prices_1min)):
        result.append(prices_1min[i])
        if i < len(prices_1min) - 1:
            # Добавляем 3 промежуточные точки (каждые ~15с)
            for k in range(1, 4):
                frac = k / 4.0
                interp = prices_1min[i] + (prices_1min[i+1] - prices_1min[i]) * frac
                result.append(interp)
    return np.array(result)


def main():
    tf = os.path.join(PROJECT_DIR, "tickers.txt")
    with open(tf, "r") as f:
        tickers = [l.strip() for l in f if l.strip()]

    combos = [
        ("ALL ON",              True,  True,  True,  True),
        ("V OFF",               False, True,  True,  True),
        ("V T OFF",             False, False, True,  True),
        ("V N OFF",             False, True,  False, True),
        ("V T N OFF",           False, False, False, True),
        ("V P OFF",             False, True,  True,  False),
        ("V T P OFF",           False, False, True,  False),
        ("V N P OFF",           False, True,  False, False),
        ("V T N P ALL OFF",     False, False, False, False),
    ]

    print(f"Tickers: {len(tickers)}, Combos: {len(combos)}, Total: {len(tickers)*len(combos)}")
    print(f"Interval: ~15s (interpolated from 1m)")

    all_results = {}
    for cname, vel, trend, nmg, pnl in combos:
        res = {}
        blocks = {"V": 0, "T": 0, "N": 0, "P": 0}
        for t in tickers:
            r = run_ticker(t, vel, trend, nmg, pnl)
            if r:
                res[t] = r["profit"]
                blocks["V"] += r["V"]
                blocks["T"] += r["T"]
                blocks["N"] += r["N"]
                blocks["P"] += r["P"]
        all_results[cname] = (res, blocks)

    # Summary table
    baseline_res, _ = all_results.get("ALL ON", ({}, {}))
    base_avg = sum(baseline_res.values())/len(baseline_res) if baseline_res else 0

    header = f"{'Combo':<20s} {'Avg%':>8s} {'Prof':>6s} {'Best':>18s} {'Worst':>18s}  Blocks[V/T/N/P]"
    print(f"\n{header}")
    print("-" * 110)

    for cname, (res, blk) in all_results.items():
        if not res: continue
        vals = list(res.values())
        avg = sum(vals) / len(vals)
        prof = sum(1 for v in vals if v > 0)
        best_t = max(res, key=res.get)
        worst_t = min(res, key=res.get)
        diff = avg - base_avg
        print(f"  {cname:<18s} {avg:>+7.2f}% {prof:>3d}/{len(vals):<3d} "
              f"{best_t:>8s}:{res[best_t]:>+7.2f}% {worst_t:>8s}:{res[worst_t]:>+7.2f}%  "
              f"[{blk['V']:>5d}/{blk['T']:>4d}/{blk['N']:>4d}/{blk['P']:>5d}]  ({diff:+.2f}%)")

    # Per-ticker best
    print(f"\n{'Per-ticker breakdown:':}")
    print("-" * 110)
    for t in tickers:
        best_c = max(all_results, key=lambda c: all_results[c][0].get(t, -999))
        best_p = all_results[best_c][0].get(t, 0)
        base_p = baseline_res.get(t, 0)
        # Find worst combo
        worst_c = min(all_results, key=lambda c: all_results[c][0].get(t, 999))
        worst_p = all_results[worst_c][0].get(t, 0)
        print(f"  {t:<14s} best={best_c:<18s} {best_p:>+7.2f}%  worst={worst_c:<18s} {worst_p:>+7.2f}%  (base={base_p:>+7.2f}%, Δbest={best_p-base_p:+.2f}%)")


if __name__ == "__main__":
    main()
