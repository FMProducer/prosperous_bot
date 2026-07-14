import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import asyncio
import aiohttp
import time
import json
import os
import math
import logging
import argparse
import traceback
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

try:
    from core.calculator import PortfolioCalculator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.calculator import PortfolioCalculator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class MarketOrderSlippageSimulator:
    """Симулятор рыночных ордеров с учетом проскальзывания и Taker-комиссии."""
    def __init__(self, commission_pct: float = 0.0004, slippage_pct: float = 0.0002):
        self.commission_pct = Decimal(str(commission_pct))
        self.slippage_pct = Decimal(str(slippage_pct))
        self.stats: Dict[str, int] = {"attempted": 0, "filled": 0}

    def simulate_market_execution(self, side: str, qty: Decimal, mid_price: Decimal) -> Tuple[Decimal, Decimal]:
        self.stats["attempted"] += 1
        dec_qty = abs(Decimal(str(qty)))
        dec_mid = Decimal(str(mid_price))

        if side.upper() == "BUY":
            exec_price = dec_mid * (Decimal('1') + self.slippage_pct)
        else:
            exec_price = dec_mid * (Decimal('1') - self.slippage_pct)

        commission = dec_qty * exec_price * self.commission_pct
        self.stats["filled"] += 1
        return exec_price, commission

    def get_summary(self) -> Dict[str, Any]:
        return self.stats

def quantize_qty(qty: Decimal, step_size: Decimal) -> Decimal:
    if step_size <= Decimal('0'):
        return qty
    remainder = qty % step_size
    return qty - remainder

def validate_notional(qty: Decimal, price: Decimal, min_notional: Decimal) -> bool:
    return abs(qty * price) >= min_notional

@dataclass
class BacktestState:
    """Управление состоянием портфеля и кросс-маржинального аккаунта."""
    initial_capital: Decimal
    base_ticker: str
    val_cash: Decimal
    account_free_margin: Decimal = Decimal('0')  # Внешняя кросс-маржинальная подушка
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
    stops_counter: int = 0
    cooldown_until_bar: int = 0
    dormant_capital: Decimal = Decimal('0')
    
    tpv_ath: float = 0.0
    trailing_stop_violation_start: float = 0.0
    trailing_stop_triggered: bool = False
    last_rebalance_price: Decimal = Decimal('0')
    history: List[float] = field(default_factory=list)
    rebalance_log: List[Dict[str, Any]] = field(default_factory=list)

    # Leverage values from config (added to fix hardcoded 5x in TPV calc)
    l_lev: Decimal = Decimal('5')
    s_lev: Decimal = Decimal('5')

    def get_tpv(self, price: Decimal) -> Decimal:
        unrealized_pnl_long = self.pos_long * (price - self.long_entry_price) if self.pos_long > 0 else Decimal('0')
        unrealized_pnl_short = self.pos_short * (self.short_entry_price - price) if self.pos_short > 0 else Decimal('0')
        margin_long = (self.pos_long * self.long_entry_price) / self.l_lev if self.pos_long > 0 else Decimal('0')
        margin_short = (self.pos_short * self.short_entry_price) / self.s_lev if self.pos_short > 0 else Decimal('0')
        virtual_equity = (self.virt_qty * price) - self.virt_debt
        return self.val_cash + margin_long + unrealized_pnl_long + margin_short + unrealized_pnl_short + virtual_equity

    def get_tpv_fast(self, price: float) -> float:
        pos_l = float(self.pos_long)
        pos_s = float(self.pos_short)
        p = float(price)
        l_entry = float(self.long_entry_price)
        s_entry = float(self.short_entry_price)
        v_qty = float(self.virt_qty)
        v_debt = float(self.virt_debt)
        cash = float(self.val_cash)
        l_lev = float(self.l_lev)
        s_lev = float(self.s_lev)

        upnl_l = pos_l * (p - l_entry) if pos_l > 0 else 0.0
        upnl_s = pos_s * (s_entry - p) if pos_s > 0 else 0.0
        m_l = (pos_l * l_entry) / l_lev if pos_l > 0 else 0.0
        m_s = (pos_s * s_entry) / s_lev if pos_s > 0 else 0.0
        v_eq = (v_qty * p) - v_debt
        return cash + m_l + upnl_l + m_s + upnl_s + v_eq

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0) -> str:
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Downloading live data for {symbol} ({days} days)...")
    total_minutes = int(days * 24 * 60)
    limit_per_request = 1500
    num_requests = math.ceil(total_minutes / limit_per_request)
    all_data = []
    end_time_ms = int(time.time() * 1000)

    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            batch = min(limit_per_request, total_minutes - len(all_data))
            if batch <= 0:
                break
            params = {"symbol": symbol, "interval": "1m", "limit": batch}
            if i > 0:
                params["endTime"] = end_time_ms

            async with session.get(endpoint, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data:
                        break
                    all_data = data + all_data
                    end_time_ms = data[0][0] - 1
                else:
                    raise Exception(f"Binance API error: {resp.status}")
            if i < num_requests - 1:
                await asyncio.sleep(0.25)

    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in df.columns:
        df[col] = df[col].astype(float)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, f"{symbol}_live.feather")
    df.to_feather(file_path)
    return file_path

async def run_backtest(config_path: str, data_dir: str, live_mode: bool = False, ticker_override: Optional[str] = None,
                        days: float = 2.0, commission: float = 0.0004, slippage: float = 0.0005,
                        quiet: bool = False,
                        threshold_surplus_override: Optional[float] = None,
                        threshold_deficit_override: Optional[float] = None,
                        capital_override: Optional[float] = None) -> Optional[Dict[str, Any]]:
    try:
        if quiet:
            logger.setLevel(logging.WARNING)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        base_ticker: str = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

        if live_mode:
            file_path = await download_live_data(base_ticker, data_dir, days)
        else:
            possible_files: List[str] = [
                os.path.join(data_dir, f"{base_ticker}_live.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_10d.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_48h.feather")
            ]
            file_path = next((p for p in possible_files if os.path.exists(p)), None)

        if not file_path or not os.path.exists(file_path):
            file_path = await download_live_data(base_ticker, data_dir, days)

        df: pd.DataFrame = pd.read_feather(file_path)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        portfolio_cfg: Dict[str, Any] = config["portfolios"][0]
        
        bars_per_day = 1440
        max_bars = int(days * bars_per_day)
        if len(df) > max_bars:
            df = df.iloc[-max_bars:]
        
        initial_capital: float = capital_override if capital_override is not None else portfolio_cfg.get("initial_capital", 1000.0)
        
        # Инъекция параметров Кросс-Маржи из конфигурации
        paper_initial_capital: float = portfolio_cfg.get("paper_initial_capital", initial_capital)
        paper_account_free_margin: float = portfolio_cfg.get("paper_account_free_margin", 0.0)
        
        targets: Dict[str, Any] = portfolio_cfg["targets"]
        ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
        
        if threshold_surplus_override is not None:
            t_surplus = threshold_surplus_override
            t_deficit = threshold_deficit_override if threshold_deficit_override is not None else threshold_surplus_override
        else:
            t_surplus = portfolio_cfg.get("rebalance_threshold_surplus", portfolio_cfg.get("rebalance_threshold", 0.02))
            t_deficit = portfolio_cfg.get("rebalance_threshold_deficit", t_surplus)
        
        t_cfg = ticker_thresholds.get(base_ticker)
        if isinstance(t_cfg, dict):
            threshold_surplus = float(t_cfg.get("surplus", t_surplus))
            threshold_deficit = float(t_cfg.get("deficit", t_deficit))
        elif isinstance(t_cfg, (float, int)):
            threshold_surplus = threshold_deficit = float(t_cfg)
        else:
            threshold_surplus = float(t_surplus)
            threshold_deficit = float(t_deficit)
            
        siphoning_threshold_pct: float = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio: float = portfolio_cfg.get("reinvestment_ratio", 0.0)
        trailing_stop_pct: float = config.get("equity_trailing_stop_pct", 0.0)
        trailing_stop_activation_pct: float = config.get("equity_trailing_stop_activation_pct", 0.0)
        trailing_stop_timeout_sec: int = int(config.get("equity_trailing_stop_timeout_sec", 0))
        min_notional_usdt: Decimal = Decimal(str(config.get("min_notional_usdt", 6.0)))
        toxic_cooldown_days: float = config.get("toxic_cooldown_days", 0.02)
        cooldown_bars_duration: int = int(toxic_cooldown_days * 1440)
        max_drawdown_limit_pct: float = config.get("max_drawdown_limit", 100.0)  # по умолчанию 100% = отключён

        close_prices: npt.NDArray[np.float64] = df['close'].values.astype(np.float64)
        step_sizes = {base_ticker: 0.001}

        sim = MarketOrderSlippageSimulator(commission_pct=commission, slippage_pct=slippage)
        
        # Инициализация состояния с учетом внешней кросс-маржи
        state = BacktestState(
            initial_capital=Decimal(str(initial_capital)), 
            val_cash=Decimal(str(initial_capital)),
            account_free_margin=Decimal(str(paper_account_free_margin)),
            base_ticker=base_ticker
        )

        daily_funding_rate = Decimal('0.0002')
        bars_per_day = Decimal('1440')
        funding_drag_step = daily_funding_rate / bars_per_day
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"]))
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"]))

        start_price = Decimal(str(close_prices[0]))
        state.last_rebalance_price = start_price
        target_v_share = Decimal(str(targets["VIRTUAL"]["share"]))
        virt_cost = target_v_share * state.initial_capital
        state.virt_qty = virt_cost / start_price
        state.virt_debt = virt_cost

        trend_guard_cfg = config.get("trend_guard", {})
        guards_cfg = portfolio_cfg.get("safety_guards", {})
        tg_min_move_pct = trend_guard_cfg.get("min_move_pct", guards_cfg.get("trend_min_move_pct", 0.5)) / 100.0
        tg_eff_threshold = trend_guard_cfg.get("eff_threshold", guards_cfg.get("trend_eff_threshold", 0.85))
        tg_nmg_pct = trend_guard_cfg.get("net_move_block_pct", guards_cfg.get("net_move_block_pct", 1.5)) / 100.0
        tg_nmg_window = trend_guard_cfg.get("net_move_window_sec", guards_cfg.get("net_move_window_sec", 30))
        tg_nmg_bars = max(1, tg_nmg_window // 60)
        tg_trend_bars = 30
        tg_blocked_count = 0

        # --- Velocity Guard / Net Move Guard (из safety_guards) ---
        vg_velocity_pct = guards_cfg.get("max_price_velocity_pct", 2.0) / 100.0
        vg_window_sec = guards_cfg.get("velocity_window_sec", 60)
        vg_window_bars = max(1, vg_window_sec // 60)
        vg_blocked_count = 0
        vg_cooldown_bars = 0  # оставшиеся бары блокировки от Velocity Guard

        for i in range(len(df)):
            mid_price = Decimal(str(close_prices[i]))

            # --- 1.5 Cooldown & Rebase Logic ---
            if i < state.cooldown_until_bar:
                state.history.append(float(state.dormant_capital))
                continue

            if i == state.cooldown_until_bar and state.cooldown_until_bar > 0:
                if state.dormant_capital < Decimal('5.0'):
                    logger.critical(f"Bar {i}: Capital too low to restart ({state.dormant_capital:.2f}). Backtest effectively dead.")
                    state.cooldown_until_bar = len(df) + 1  # Dead forever
                    state.history.append(float(state.dormant_capital))
                    continue

                logger.info(f"Bar {i}: Cooldown finished. Reviving bot. Rebased capital: {state.dormant_capital:.2f}")
                state.initial_capital = state.dormant_capital
                state.val_cash = state.dormant_capital
                state.pos_long = Decimal('0')
                state.pos_short = Decimal('0')
                state.virt_qty = Decimal('0')
                state.virt_debt = Decimal('0')
                state.long_entry_price = Decimal('0')
                state.short_entry_price = Decimal('0')
                state.siphoning_reserve = Decimal('0')
                state.tpv_ath = max(float(state.dormant_capital), initial_capital_f)
                state.trailing_stop_violation_start = 0.0
                state.trailing_stop_triggered = False
                state.last_rebalance_price = mid_price

                # Re-init virtual leg
                target_v_share = Decimal(str(targets["VIRTUAL"]["share"]))
                virt_cost = target_v_share * state.initial_capital
                state.virt_qty = virt_cost / mid_price
                state.virt_debt = virt_cost
                state.cooldown_until_bar = 0

            # --- 2. Portfolio Calculation ---
            margin_long_current = (state.pos_long * state.long_entry_price) / l_lev if state.pos_long > 0 else Decimal('0')
            margin_short_current = (state.pos_short * state.short_entry_price) / s_lev if state.pos_short > 0 else Decimal('0')
            wallet_balance = state.val_cash + margin_long_current + margin_short_current
            simulated_real_equity = wallet_balance - state.virt_debt

            calculator = PortfolioCalculator(
                positions={f"{base_ticker}_LONG": float(state.pos_long), f"{base_ticker}_SHORT": float(state.pos_short)},
                spot_price=float(mid_price),
                real_equity=float(simulated_real_equity),
                virt_qty=float(state.virt_qty),
                virt_debt=float(state.virt_debt),
                long_entry_price=float(state.long_entry_price),
                short_entry_price=float(state.short_entry_price),
                base_ticker=base_ticker,
                targets=targets,
                initial_capital=float(state.initial_capital),
                last_rebalance_price=float(state.last_rebalance_price)
            )

            calc_res = calculator.calculate_rebalance(
                targets,
                threshold_surplus=threshold_surplus,
                threshold_deficit=threshold_deficit,
                current_equity=state.get_tpv(mid_price)
            )
            actions = calc_res["actions"]

            if actions and i >= tg_trend_bars:
                lookback_prices = close_prices[max(0, i - tg_trend_bars):i + 1]
                if len(lookback_prices) >= 2:
                    p_start = lookback_prices[0]
                    p_end = lookback_prices[-1]
                    net_move = abs(p_end - p_start)
                    total_path = sum(abs(lookback_prices[j] - lookback_prices[j-1]) for j in range(1, len(lookback_prices)))
                    trend_eff = (net_move / total_path) if total_path > 0 else 0
                    net_move_pct = net_move / p_start if p_start > 0 else 0
                    if net_move_pct > tg_min_move_pct and trend_eff > tg_eff_threshold:
                        actions = []
                        tg_blocked_count += 1
                
                if actions and i >= tg_nmg_bars:
                    nmg_prices = close_prices[max(0, i - tg_nmg_bars):i + 1]
                    if len(nmg_prices) >= 2:
                        nmg_old = nmg_prices[0]
                        nmg_move = abs(close_prices[i] - nmg_old) / nmg_old if nmg_old > 0 else 0
                        if nmg_move > tg_nmg_pct:
                            actions = []
                            tg_blocked_count += 1

            # --- Velocity Guard: блокировка при быстром движении цены ---
            if actions and vg_cooldown_bars > 0:
                vg_cooldown_bars -= 1
                actions = []
                vg_blocked_count += 1
            elif actions and i >= vg_window_bars:
                vg_prices = close_prices[max(0, i - vg_window_bars):i + 1]
                if len(vg_prices) >= 2:
                    vg_old = vg_prices[0]
                    vg_move = abs(close_prices[i] - vg_old) / vg_old if vg_old > 0 else 0
                    if vg_move > vg_velocity_pct:
                        actions = []
                        vg_blocked_count += 1
                        vg_cooldown_bars = vg_window_bars  # блокируем на окно

            if actions:
                reductions = [a for a in actions if a.get("is_reduction", False)]
                expansions = [a for a in actions if not a.get("is_reduction", False)]

                for act in reductions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    raw_qty = abs(Decimal(str(act["diff_usdt"]))) / mid_price
                    qty = quantize_qty(raw_qty, Decimal(str(step_sizes.get(base_ticker, 0.001))))

                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue

                    if key == "BASE_LONG":
                        exec_price, commission = sim.simulate_market_execution("SELL", qty, mid_price)
                        realized_pnl = qty * (exec_price - state.long_entry_price)
                        released_margin = (qty * state.long_entry_price) / lev
                        state.pos_long -= qty
                        state.val_cash += released_margin + realized_pnl - commission
                        if state.pos_long == Decimal('0'):
                            state.long_entry_price = Decimal('0')
                    elif key == "BASE_SHORT":
                        exec_price, commission = sim.simulate_market_execution("BUY", qty, mid_price)
                        realized_pnl = qty * (state.short_entry_price - exec_price)
                        released_margin = (qty * state.short_entry_price) / lev
                        state.pos_short -= qty
                        state.val_cash += released_margin + realized_pnl - commission
                        if state.pos_short == Decimal('0'):
                            state.short_entry_price = Decimal('0')
                    elif key == "VIRTUAL":
                        exec_price, commission = sim.simulate_market_execution("SELL", qty, mid_price)
                        v_entry_price = (state.virt_debt / state.virt_qty) if state.virt_qty > 0 else exec_price
                        allocated_debt_reduction = qty * v_entry_price
                        realized_pnl = qty * (exec_price - v_entry_price)
                        state.virt_qty -= qty
                        state.virt_debt -= allocated_debt_reduction
                        state.val_cash += allocated_debt_reduction + realized_pnl - commission

                expansions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)
                remaining_funds = Decimal(str(calc_res.get("val_cash", calc_res.get("available_funds", 0.0))))

                for act in expansions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    needed_usdt = abs(Decimal(str(act["diff_usdt"])))

                    if remaining_funds <= Decimal('0'):
                        state.skipped_expansions_counter += 1
                        continue

                    max_usdt = remaining_funds * lev
                    if needed_usdt > max_usdt:
                        needed_usdt = max_usdt

                    raw_qty = needed_usdt / mid_price
                    qty = quantize_qty(raw_qty, Decimal(str(step_sizes.get(base_ticker, 0.001))))

                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue

                    actual_equity_spent = needed_usdt / lev

                    if key == "BASE_LONG":
                        exec_price, commission = sim.simulate_market_execution("BUY", qty, mid_price)
                        margin_required = (qty * exec_price) / lev
                        state.val_cash -= margin_required + commission
                        state.long_entry_price = ((state.pos_long * state.long_entry_price) + (qty * exec_price)) / (state.pos_long + qty)
                        state.pos_long += qty
                    elif key == "BASE_SHORT":
                        exec_price, commission = sim.simulate_market_execution("SELL", qty, mid_price)
                        margin_required = (qty * exec_price) / lev
                        state.val_cash -= margin_required + commission
                        state.short_entry_price = ((state.pos_short * state.short_entry_price) + (qty * exec_price)) / (state.pos_short + qty)
                        state.pos_short += qty
                    elif key == "VIRTUAL":
                        exec_price, commission = sim.simulate_market_execution("BUY", qty, mid_price)
                        cash_spent = (qty * exec_price) + commission
                        state.val_cash -= cash_spent
                        state.virt_qty += qty
                        state.virt_debt += cash_spent
                        actual_equity_spent = cash_spent

                    remaining_funds -= actual_equity_spent

                state.cycles += 1
                state.last_rebalance_price = mid_price
                state.rebalance_log.append({
                    "step": i, "price": float(mid_price), "tpv": float(state.get_tpv(mid_price)),
                    "shares": {"L": calc_res["share_long_pct"], "S": calc_res["share_short_pct"], "V": calc_res["share_virt_pct"], "C": calc_res["share_cash_pct"]},
                    "actions": [f"{a['key']} {'SELL' if a['is_reduction'] else 'BUY'}" for a in actions]
                })

            long_notional = state.pos_long * mid_price
            short_notional = state.pos_short * mid_price
            total_drag = (long_notional + short_notional) * funding_drag_step
            state.val_cash -= total_drag

            # --- 2.5 Unified Predictive Cross-Margin Liquidation Guard ---
            # Расчет метрик Кросс-Маржи согласно спецификации Binance Futures
            mmr = Decimal('0.004')  # Maintenance Margin Rate для плеча <= 5x
            
            upnl_l = state.pos_long * (mid_price - state.long_entry_price) if state.pos_long > 0 else Decimal('0')
            upnl_s = state.pos_short * (state.short_entry_price - mid_price) if state.pos_short > 0 else Decimal('0')
            m_long = (state.pos_long * state.long_entry_price) / l_lev if state.pos_long > 0 else Decimal('0')
            m_short = (state.pos_short * state.short_entry_price) / s_lev if state.pos_short > 0 else Decimal('0')
            
            # Совокупный Margin Balance суб-счета + внешняя подушка обеспечения аккаунта
            futures_margin_balance = state.val_cash + m_long + upnl_l + m_short + upnl_s
            total_margin_balance = futures_margin_balance + state.account_free_margin
            
            # Общий Maintenance Margin Requirement для обеих фьючерсных позиций
            total_mm = (long_notional + short_notional) * mmr

            if state.pos_long > 0 or state.pos_short > 0:
                if total_margin_balance > 0:
                    cross_liq_dist_pct = ((total_margin_balance - total_mm) / total_margin_balance) * Decimal('100')
                else:
                    cross_liq_dist_pct = Decimal('-100')

                liq_distance_warn_pct = Decimal(str(portfolio_cfg.get("liquidation_distance_warn_pct", 15.0)))
                liq_distance_crit_pct = Decimal(str(portfolio_cfg.get("liquidation_distance_crit_pct", 8.0)))

                if cross_liq_dist_pct <= liq_distance_crit_pct:
                    logger.warning(f"Bar {i}: Predictive CROSS Liquidation Guard triggered! Distance={float(cross_liq_dist_pct):.1f}%. Экстренное закрытие фьючерсных позиций.")
                    
                    # Закрытие Long
                    if state.pos_long > 0:
                        exec_p, comm = sim.simulate_market_execution("SELL", state.pos_long, mid_price)
                        realized_pnl = state.pos_long * (exec_p - state.long_entry_price)
                        state.val_cash += m_long + realized_pnl - comm
                        state.pos_long = Decimal('0')
                        state.long_entry_price = Decimal('0')

                    # Закрытие Short
                    if state.pos_short > 0:
                        exec_p, comm = sim.simulate_market_execution("BUY", state.pos_short, mid_price)
                        realized_pnl = state.pos_short * (state.short_entry_price - exec_p)
                        state.val_cash += m_short + realized_pnl - comm
                        state.pos_short = Decimal('0')
                        state.short_entry_price = Decimal('0')

                    state.liquidations_counter += 1
                    state.dormant_capital = Decimal(str(state.get_tpv_fast(float(mid_price)) + float(state.siphoning_reserve)))
                    state.cooldown_until_bar = i + cooldown_bars_duration
                    logger.warning(f"Bar {i}: Entering {toxic_cooldown_days}d cooldown after Predictive Liquidation.")
                    state.history.append(float(state.dormant_capital))
                    continue
                elif cross_liq_dist_pct <= liq_distance_warn_pct:
                    logger.debug(f"Bar {i}: CROSS Liquidation Warning! Distance={float(cross_liq_dist_pct):.1f}%")

            # --- 3. Post-Factum Cross-Margin Liquidation Check (Абсолютный Фоллбэк) ---
            if state.pos_long > 0 or state.pos_short > 0:
                upnl_l = state.pos_long * (mid_price - state.long_entry_price) if state.pos_long > 0 else Decimal('0')
                upnl_s = state.pos_short * (state.short_entry_price - mid_price) if state.pos_short > 0 else Decimal('0')
                m_long = (state.pos_long * state.long_entry_price) / l_lev if state.pos_long > 0 else Decimal('0')
                m_short = (state.pos_short * state.short_entry_price) / s_lev if state.pos_short > 0 else Decimal('0')
                
                futures_margin_balance = state.val_cash + m_long + upnl_l + m_short + upnl_s
                total_margin_balance = futures_margin_balance + state.account_free_margin
                total_mm = (state.pos_long * mid_price + state.pos_short * mid_price) * mmr

                if total_margin_balance <= total_mm:
                    logger.warning(f"Bar {i}: HARD CROSS LIQUIDATION! Весь баланс суб-аккаунта уничтожен.")
                    # Если баланс суб-аккаунта отрицательный, списываем убыток из внешней кросс-маржи аккаунта
                    if futures_margin_balance < 0:
                        state.account_free_margin += futures_margin_balance
                        if state.account_free_margin < 0:
                            state.account_free_margin = Decimal('0')
                    
                    state.val_cash = Decimal('0')
                    state.pos_long = Decimal('0')
                    state.long_entry_price = Decimal('0')
                    state.pos_short = Decimal('0')
                    state.short_entry_price = Decimal('0')
                    state.liquidations_counter += 1
                    state.dormant_capital = Decimal(str(state.account_free_margin + float(state.siphoning_reserve)))
                    state.cooldown_until_bar = i + cooldown_bars_duration
                    logger.warning(f"Bar {i}: Entering {toxic_cooldown_days}d cooldown after HARD Liquidation.")
                    state.history.append(float(state.dormant_capital))
                    continue

            # --- 4. SAFE Siphoning ---
            tpv_f = state.get_tpv_fast(float(mid_price))
            total_tpv_with_reserve = tpv_f + float(state.siphoning_reserve)
            initial_capital_f = float(state.initial_capital)
            total_surplus = total_tpv_with_reserve - initial_capital_f
            siphoning_threshold_abs = initial_capital_f * (siphoning_threshold_pct / 100.0)

            if total_surplus > float(state.siphoning_reserve) + max(0.1, siphoning_threshold_abs):
                new_profit = total_surplus - float(state.siphoning_reserve)
                siphon_amount = new_profit * (1 - reinvestment_ratio)
                if siphon_amount > 0.1:
                    state.siphoning_reserve += Decimal(str(siphon_amount))
                    state.val_cash -= Decimal(str(siphon_amount))
                    total_tpv_with_reserve -= siphon_amount

            # --- 4b. Max Drawdown Limit (Emergency Stop) ---
            if max_drawdown_limit_pct < 100.0:
                _dd_base = state.tpv_ath if state.tpv_ath > 0 else initial_capital_f
                drawdown_threshold = _dd_base * (1.0 - max_drawdown_limit_pct / 100.0)
                if total_tpv_with_reserve < drawdown_threshold:
                    logger.warning(f"Bar {i}: MAX DRAWDOWN LIMIT reached! TPV={total_tpv_with_reserve:.2f} < threshold={drawdown_threshold:.2f} ({max_drawdown_limit_pct}% DD from ATH={_dd_base:.2f}). Stopping.")
                    state.trailing_stop_triggered = True
                    state.stops_counter += 1
                    # Закрыть все позиции по текущей цене
                    if state.pos_long > 0:
                        exec_p, comm = sim.simulate_market_execution("SELL", state.pos_long, mid_price)
                        realized_pnl = state.pos_long * (exec_p - state.long_entry_price)
                        m_l = (state.pos_long * state.long_entry_price) / l_lev
                        state.val_cash += m_l + realized_pnl - comm
                        state.pos_long = Decimal('0')
                        state.long_entry_price = Decimal('0')
                    if state.pos_short > 0:
                        exec_p, comm = sim.simulate_market_execution("BUY", state.pos_short, mid_price)
                        realized_pnl = state.pos_short * (state.short_entry_price - exec_p)
                        m_s = (state.pos_short * state.short_entry_price) / s_lev
                        state.val_cash += m_s + realized_pnl - comm
                        state.pos_short = Decimal('0')
                        state.short_entry_price = Decimal('0')
                    state.dormant_capital = Decimal(str(total_tpv_with_reserve))
                    state.cooldown_until_bar = len(df) + 1  # больше не стартуем
                    state.history.append(float(state.dormant_capital))
                    continue

            # --- 4c. Trailing Stop ---
            if total_tpv_with_reserve > state.tpv_ath:
                state.tpv_ath = total_tpv_with_reserve
                # Floor: tpv_ath cannot be below initial_capital
                if state.tpv_ath < initial_capital_f:
                    state.tpv_ath = initial_capital_f
                state.trailing_stop_violation_start = 0.0

            if (trailing_stop_pct > 0
                    and state.tpv_ath > initial_capital_f * (1 + trailing_stop_activation_pct / 100.0)):
                drawdown_from_ath = (1 - total_tpv_with_reserve / state.tpv_ath) * 100.0
                if drawdown_from_ath >= trailing_stop_pct:
                    if state.trailing_stop_violation_start == 0:
                        state.trailing_stop_violation_start = float(i)
                        logger.warning(f"Bar {i}: Trailing Stop threshold breached (DD {drawdown_from_ath:.2f}% from ATH).")
                    else:
                        elapsed_bars = i - int(state.trailing_stop_violation_start)
                        timeout_bars = max(1, trailing_stop_timeout_sec // 60)
                        if elapsed_bars >= timeout_bars:
                            logger.warning(f"Bar {i}: Trailing Stop triggered. Entering {toxic_cooldown_days}d cooldown.")
                            state.trailing_stop_triggered = True
                            state.stops_counter += 1
                            state.dormant_capital = Decimal(str(total_tpv_with_reserve))
                            state.cooldown_until_bar = i + cooldown_bars_duration
                            state.history.append(float(state.dormant_capital))
                            continue

            # --- 5. Проверка математического инварианта ---
            virtual_equity = (state.virt_qty * mid_price) - state.virt_debt
            current_tpv = state.val_cash + m_long + upnl_l + m_short + upnl_s + virtual_equity
            assert current_tpv > Decimal('0'), "Критический дефолт портфеля: TPV <= 0"

            state.history.append(float(total_tpv_with_reserve))

        equity_curve: npt.NDArray[np.float64] = np.array(state.history)
        profit_pct: float = (equity_curve[-1] / (initial_capital + 1e-9) - 1) * 100
        max_eq: npt.NDArray[np.float64] = np.maximum.accumulate(equity_curve)
        dd: npt.NDArray[np.float64] = (max_eq - equity_curve) / (max_eq + 1e-9)
        max_dd_pct: float = np.max(dd) * 100 if len(dd) > 0 else 0.0
        asset_chg_pct: float = (close_prices[-1] / (close_prices[0] + 1e-9) - 1) * 100

        # --- Sortino Ratio ---
        # Средняя доходность за бар / downside deviation
        if len(equity_curve) > 1:
            bar_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
            mean_return = np.mean(bar_returns)
            # Downside deviation: только отрицательные доходности
            negative_returns = bar_returns[bar_returns < 0]
            if len(negative_returns) > 0:
                downside_dev = np.sqrt(np.mean(negative_returns ** 2))
                sortino_ratio = mean_return / (downside_dev + 1e-9)
            else:
                sortino_ratio = mean_return / 1e-9  # нет отрицательных — идеально
        else:
            sortino_ratio = 0.0

        if not quiet:
            logger.info(f"Refactored Cross-Margin Backtest for {base_ticker}:")
            logger.info(f"Profit: {profit_pct:+.2f}% | MaxDD: {max_dd_pct:.2f}% | Cycles: {state.cycles}")
            logger.info(f"Sortino: {sortino_ratio:.4f} | Asset Change: {asset_chg_pct:+.2f}% | SAFE Reserve: {state.siphoning_reserve:.2f} USDT")
            logger.info(f"Remained Account Free Margin: {state.account_free_margin:.2f} USDT")
            logger.info(f"Skipped Expansions: {state.skipped_expansions_counter} | Stops: {state.stops_counter} | Liqs: {state.liquidations_counter}")
            logger.info(f"Trend Guard blocks: {tg_blocked_count} | Velocity Guard blocks: {vg_blocked_count}")
            if sim:
                s = sim.get_summary()
                logger.info(f"Market Orders: {s['filled']}/{s['attempted']} filled")

        return {
            "profit_pct": float(profit_pct),
            "max_dd_pct": float(max_dd_pct),
            "cycles": int(state.cycles),
            "asset_chg_pct": float(asset_chg_pct),
            "siphoning_reserve": float(state.siphoning_reserve),
            "skipped_expansions": int(state.skipped_expansions_counter),
            "liquidations": int(state.liquidations_counter),
            "trailing_stops": int(state.stops_counter),
            "trailing_stop_triggered": state.stops_counter > 0,
            "trend_guard_blocks": int(tg_blocked_count),
            "velocity_guard_blocks": int(vg_blocked_count),
            "sortino_ratio": float(sortino_ratio)
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--threshold-surplus", type=float, default=None)
    parser.add_argument("--threshold-deficit", type=float, default=None)
    args = parser.parse_args()
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days,
                             capital_override=args.capital,
                             threshold_surplus_override=args.threshold_surplus,
                             threshold_deficit_override=args.threshold_deficit))