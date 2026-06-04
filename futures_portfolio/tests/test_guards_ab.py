"""
A/B тест: бэктест HEIUSDT с отключением гвардов.
Базовая линия: все гварды ON. Все тесты: Velocity OFF + комбинации остальных.
Запуск: python test_guards_ab.py
"""
import json
import os
import sys
import logging
import warnings
from collections import deque
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculator import PortfolioCalculator
from backtest_rebalance import MarketOrderSlippageSimulator, quantize_qty, validate_notional

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("GuardsAB")
warnings.filterwarnings("ignore")
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


@dataclass
class BacktestState:
    initial_capital: Decimal
    base_ticker: str
    val_cash: Decimal
    pos_long: Decimal = Decimal('0')
    pos_short: Decimal = Decimal('0')
    virt_qty: Decimal = Decimal('0')
    virt_debt: Decimal = Decimal('0')
    long_entry_price: Decimal = Decimal('0')
    short_entry_price: Decimal = Decimal('0')
    siphoning_reserve: Decimal = Decimal('0')
    cycles: int = 0
    skipped_expansions_counter: int = 0
    liquidations_counter: int = 0
    tpv_ath: float = 0.0
    trailing_stop_violation_start: float = 0.0
    trailing_stop_triggered: bool = False
    last_rebalance_price: Decimal = Decimal('0')
    history: List[float] = field(default_factory=list)

    def get_tpv(self, price):
        upnl_l = self.pos_long * (price - self.long_entry_price) if self.pos_long > 0 else Decimal('0')
        upnl_s = self.pos_short * (self.short_entry_price - price) if self.pos_short > 0 else Decimal('0')
        m_l = (self.pos_long * self.long_entry_price) / Decimal('5') if self.pos_long > 0 else Decimal('0')
        m_s = (self.pos_short * self.short_entry_price) / Decimal('5') if self.pos_short > 0 else Decimal('0')
        v_eq = (self.virt_qty * price) - self.virt_debt
        return self.val_cash + m_l + upnl_l + m_s + upnl_s + v_eq


def run_one(enable_velocity, enable_trend, enable_nmg, enable_pnl_guard):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    t_surplus = portfolio_cfg.get("rebalance_threshold_surplus", 0.02)
    t_deficit = portfolio_cfg.get("rebalance_threshold_deficit", t_surplus)
    t_cfg = portfolio_cfg.get("ticker_thresholds", {}).get("HEIUSDT")
    if isinstance(t_cfg, dict):
        threshold_surplus = float(t_cfg.get("surplus", t_surplus))
        threshold_deficit = float(t_cfg.get("deficit", t_deficit))
    else:
        threshold_surplus = threshold_deficit = float(t_surplus)

    guards_cfg = portfolio_cfg.get("safety_guards", {})
    max_velocity = float(guards_cfg.get("max_price_velocity_pct", 1.0)) / 100
    velocity_window = int(guards_cfg.get("velocity_window_sec", 60))
    net_move_block_pct = float(guards_cfg.get("net_move_block_pct", 1.5)) / 100
    net_move_window = int(guards_cfg.get("net_move_window_sec", 30))

    file_path = os.path.join(DATA_DIR, "HEIUSDT_live.feather")
    if not os.path.exists(file_path):
        return None

    df = pd.read_feather(file_path)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    bars_per_day = 1440
    max_bars = int(2.0 * bars_per_day)
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]

    initial_capital = float(portfolio_cfg.get("initial_capital", 180.0))
    trailing_stop_pct = config.get("equity_trailing_stop_pct", 0.0)
    trailing_stop_activation_pct = config.get("equity_trailing_stop_activation_pct", 0.0)
    trailing_stop_timeout_sec = int(config.get("equity_trailing_stop_timeout_sec", 60))
    min_notional_usdt = Decimal(str(config.get("min_notional_usdt", 6.0)))
    step_sizes = {"HEIUSDT": 0.001}
    close_prices = df['close'].values.astype(np.float64)

    sim = MarketOrderSlippageSimulator(commission_pct=0.0004, slippage_pct=0.0002)
    state = BacktestState(
        initial_capital=Decimal(str(initial_capital)),
        base_ticker="HEIUSDT",
        val_cash=Decimal(str(initial_capital))
    )

    l_lev = Decimal(str(targets["BASE_LONG"]["leverage"]))
    s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"]))
    start_price = Decimal(str(close_prices[0]))
    state.last_rebalance_price = start_price
    target_v_share = Decimal(str(targets["VIRTUAL"]["share"]))
    virt_cost = target_v_share * state.initial_capital
    state.virt_qty = virt_cost / start_price
    state.virt_debt = virt_cost

    funding_drag_step = Decimal('0.0002') / Decimal('1440')
    mmr = Decimal('0.004')
    liq_crit = portfolio_cfg.get("liquidation_distance_crit_pct", 8.0)

    velocity_blocks = 0
    trend_blocks = 0
    nmg_blocks = 0
    pnl_blocks = 0
    price_history = deque()

    for i in range(len(df)):
        mid_price = Decimal(str(close_prices[i]))
        now = float(i)

        price_history.append((now, float(mid_price)))
        while price_history and (now - price_history[0][0]) > 300:
            price_history.popleft()

        guards_block = False

        # 1. VELOCITY GUARD
        if enable_velocity and len(price_history) > 1:
            v_pts = [(t, p) for t, p in price_history if now - t <= velocity_window]
            if v_pts:
                _, old_p = v_pts[0]
                if old_p > 0:
                    vel = abs(float(mid_price) - old_p) / old_p
                    if vel > max_velocity:
                        guards_block = True
                        velocity_blocks += 1

        # 2. TREND GUARD
        if enable_trend and not guards_block and len(price_history) > 10:
            p_list = [p[1] for p in price_history]
            net_move = abs(p_list[-1] - p_list[0])
            total_path = sum(abs(p_list[j] - p_list[j-1]) for j in range(1, len(p_list)))
            trend_eff = (net_move / total_path) if total_path > 0 else 0
            if (net_move / p_list[0] > 0.005) and trend_eff > 0.85:
                guards_block = True
                trend_blocks += 1

        # 3. NET MOVE GUARD
        if enable_nmg and not guards_block and len(price_history) > 1:
            nmg_pts = [(t, p) for t, p in price_history if now - t <= net_move_window]
            if nmg_pts:
                _, nmg_p = nmg_pts[0]
                if nmg_p > 0:
                    nmg_move = abs(float(mid_price) - nmg_p) / nmg_p
                    if nmg_move > net_move_block_pct:
                        guards_block = True
                        nmg_blocks += 1

        if not guards_block:
            # PNL GUARD
            allow_surplus_sell = True
            margin_l = (state.pos_long * state.long_entry_price) / l_lev if state.pos_long > 0 else Decimal('0')
            margin_s = (state.pos_short * state.short_entry_price) / s_lev if state.pos_short > 0 else Decimal('0')
            wallet_bal = state.val_cash + margin_l + margin_s
            sim_eq = float(wallet_bal - state.virt_debt)

            calc = PortfolioCalculator(
                positions={"HEIUSDT_LONG": float(state.pos_long), "HEIUSDT_SHORT": float(state.pos_short)},
                spot_price=float(mid_price), real_equity=sim_eq,
                virt_qty=float(state.virt_qty), virt_debt=float(state.virt_debt),
                long_entry_price=float(state.long_entry_price),
                short_entry_price=float(state.short_entry_price),
                base_ticker="HEIUSDT", targets=targets,
                initial_capital=float(state.initial_capital),
                last_rebalance_price=float(state.last_rebalance_price)
            )

            if enable_pnl_guard:
                hedge_pnl = float(calc.pnl_l) + float(calc.pnl_s)
                if hedge_pnl < 0:
                    allow_surplus_sell = False
                    pnl_blocks += 1

            calc_res = calc.calculate_rebalance(targets, threshold_surplus, threshold_surplus,
                                                current_equity=calc.tpv,
                                                allow_surplus_sell=allow_surplus_sell)
            actions = calc_res["actions"]

            if actions:
                reductions = [a for a in actions if a.get("is_reduction", False)]
                expansions = [a for a in actions if not a.get("is_reduction", False)]

                for act in reductions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    qty = quantize_qty(abs(Decimal(str(act["diff_usdt"]))) / mid_price,
                                       Decimal(str(step_sizes["HEIUSDT"])))
                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue
                    if key == "BASE_LONG":
                        ep, comm = sim.simulate_market_execution("SELL", qty, mid_price)
                        rpnl = qty * (ep - state.long_entry_price)
                        rm = (qty * state.long_entry_price) / lev
                        state.pos_long -= qty
                        state.val_cash += rm + rpnl - comm
                        if state.pos_long == Decimal('0'):
                            state.long_entry_price = Decimal('0')
                    elif key == "BASE_SHORT":
                        ep, comm = sim.simulate_market_execution("BUY", qty, mid_price)
                        rpnl = qty * (state.short_entry_price - ep)
                        rm = (qty * state.short_entry_price) / lev
                        state.pos_short -= qty
                        state.val_cash += rm + rpnl - comm
                        if state.pos_short == Decimal('0'):
                            state.short_entry_price = Decimal('0')

                expansions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)
                rem = Decimal(str(calc_res.get("val_cash", 0.0)))

                for act in expansions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    need = abs(Decimal(str(act["diff_usdt"])))
                    if rem <= Decimal('0'):
                        state.skipped_expansions_counter += 1
                        continue
                    if need > rem * lev:
                        need = rem * lev
                    qty = quantize_qty(need / mid_price, Decimal(str(step_sizes["HEIUSDT"])))
                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue
                    eq_spent = need / lev
                    if key == "BASE_LONG":
                        ep, comm = sim.simulate_market_execution("BUY", qty, mid_price)
                        mr = (qty * ep) / lev
                        state.val_cash -= mr + comm
                        state.long_entry_price = ((state.pos_long * state.long_entry_price) + (qty * ep)) / (state.pos_long + qty)
                        state.pos_long += qty
                    elif key == "BASE_SHORT":
                        ep, comm = sim.simulate_market_execution("SELL", qty, mid_price)
                        mr = (qty * ep) / lev
                        state.val_cash -= mr + comm
                        state.short_entry_price = ((state.pos_short * state.short_entry_price) + (qty * ep)) / (state.pos_short + qty)
                        state.pos_short += qty
                    rem -= eq_spent

                state.cycles += 1
                state.last_rebalance_price = mid_price

        # FUNDING
        drag = (state.pos_long * mid_price + state.pos_short * mid_price) * funding_drag_step
        state.val_cash -= drag

        # LIQUIDATION
        if state.pos_long > 0 and state.long_entry_price > 0:
            liq_p = state.long_entry_price * (Decimal('1') - Decimal('1')/l_lev + mmr)
            dist = (mid_price - liq_p) / mid_price * 100
            if float(dist) <= liq_crit:
                m = (state.pos_long * state.long_entry_price) / l_lev
                upnl = state.pos_long * (mid_price - state.long_entry_price)
                state.val_cash += m + upnl
                state.pos_long = Decimal('0')
                state.long_entry_price = Decimal('0')
                state.liquidations_counter += 1

        if state.pos_short > 0 and state.short_entry_price > 0:
            liq_p = state.short_entry_price * (Decimal('1') + Decimal('1')/s_lev - mmr)
            dist = (liq_p - mid_price) / mid_price * 100
            if float(dist) <= liq_crit:
                m = (state.pos_short * state.short_entry_price) / s_lev
                upnl = state.pos_short * (state.short_entry_price - mid_price)
                state.val_cash += m + upnl
                state.pos_short = Decimal('0')
                state.short_entry_price = Decimal('0')
                state.liquidations_counter += 1

        # TRAILING STOP
        tpv_val = float(state.get_tpv(mid_price)) + float(state.siphoning_reserve)
        if tpv_val > state.tpv_ath:
            state.tpv_ath = tpv_val
            state.trailing_stop_violation_start = 0.0
        if trailing_stop_pct > 0 and state.tpv_ath > initial_capital * (1 + trailing_stop_activation_pct / 100):
            dd = (1 - tpv_val / state.tpv_ath) * 100
            if dd >= trailing_stop_pct:
                if state.trailing_stop_violation_start == 0:
                    state.trailing_stop_violation_start = float(i)
                elif i - int(state.trailing_stop_violation_start) >= max(1, trailing_stop_timeout_sec // 60):
                    state.trailing_stop_triggered = True
                    state.history.append(tpv_val)
                    break
            else:
                state.trailing_stop_violation_start = 0.0

        state.history.append(tpv_val)

    eq = np.array(state.history)
    profit_pct = (eq[-1] / (initial_capital + 1e-9) - 1) * 100
    max_eq = np.maximum.accumulate(eq)
    dd = np.max((max_eq - eq) / (max_eq + 1e-9)) * 100 if len(eq) > 0 else 0
    asset_chg = (close_prices[-1] / (close_prices[0] + 1e-9) - 1) * 100

    return {
        "profit_pct": profit_pct, "max_dd_pct": dd, "cycles": state.cycles,
        "asset_chg_pct": asset_chg, "liquidations": state.liquidations_counter,
        "ts": state.trailing_stop_triggered,
        "V": velocity_blocks, "T": trend_blocks, "N": nmg_blocks, "P": pnl_blocks,
    }


def main():
    tests = [
        # (name, velocity, trend, nmg, pnl_guard)
        ("BASELINE all ON",       True,  True,  True,  True),
        ("V OFF, T N P ON",       False, True,  True,  True),
        ("V OFF, T OFF, N P ON",  False, False, True,  True),
        ("V OFF, T ON, N OFF P ON",False, True,  False, True),
        ("V OFF, T N OFF, P ON",  False, False, False, True),
        ("V OFF, T N P OFF",      False, False, False, False),
        ("V OFF, T ON, N ON P OFF",False, True,  True, False),
        ("V OFF, T OFF, N ON P ON",False, False, True, True),
    ]

    results = []
    for name, v, t, n, p in tests:
        r = run_one(v, t, n, p)
        if r:
            results.append((name, r))

    base = results[0][1]["profit_pct"]
    print(f"\n{'Config':<38s} {'Profit':>8s} {'DD':>7s} {'Cyc':>5s} {'Liq':>4s} {'TS':>4s}  Blks [V/T/N/P]")
    print("-" * 95)
    for name, r in results:
        diff = r["profit_pct"] - base
        ts_s = "YES" if r["ts"] else "no"
        print(f"  {name:<36s} {r['profit_pct']:>+7.2f}% {r['max_dd_pct']:>5.1f}% {r['cycles']:>4d}  "
              f"{r['liquidations']:>3d}  {ts_s:>3s}  [{r['V']:>4d}/{r['T']:>3d}/{r['N']:>3d}/{r['P']:>4d}]"
              f"  ({diff:+.2f}%)")


if __name__ == "__main__":
    main()
