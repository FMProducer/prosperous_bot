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

# Import the live calculator to ensure perfect logic synchronization
# Ensure PYTHONPATH is set or we are in the correct directory
try:
    from calculator import PortfolioCalculator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calculator import PortfolioCalculator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class MarketOrderSlippageSimulator:
    """Симулятор рыночных ордеров с учетом проскальзывания и Taker-комиссии."""
    def __init__(self, commission_pct: float = 0.0004, slippage_pct: float = 0.0002):
        self.commission_pct = Decimal(str(commission_pct))
        self.slippage_pct = Decimal(str(slippage_pct))
        self.stats: Dict[str, int] = {"attempted": 0, "filled": 0}

    def simulate_market_execution(self, side: str, qty: Decimal, mid_price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Возвращает (exec_price, commission).
        Применяет проскальзывание в зависимости от направления сделки.
        """
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
    """Округление объема ордера строго вниз до параметров stepSize биржи."""
    if step_size <= Decimal('0'):
        return qty
    remainder = qty % step_size
    return qty - remainder

def validate_notional(qty: Decimal, price: Decimal, min_notional: Decimal) -> bool:
    """
    Проверка Dust Guard: ордер отбрасывается, если его номинальная стоимость
    ниже установленного биржевого лимита min_notional_usdt.
    """
    notional = abs(qty * price)
    return notional >= min_notional

@dataclass
class BacktestState:
    """Управление состоянием портфеля во время бэктеста."""
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
    # Trailing stop state
    tpv_ath: float = 0.0
    trailing_stop_violation_start: float = 0.0
    trailing_stop_triggered: bool = False
    last_rebalance_price: Decimal = Decimal('0')
    history: List[float] = field(default_factory=list)
    rebalance_log: List[Dict[str, Any]] = field(default_factory=list)

    def get_tpv(self, price: Decimal) -> Decimal:
        """Calculate TPV using precise Decimal arithmetic."""
        unrealized_pnl_long = self.pos_long * (price - self.long_entry_price) if self.pos_long > 0 else Decimal('0')
        unrealized_pnl_short = self.pos_short * (self.short_entry_price - price) if self.pos_short > 0 else Decimal('0')
        margin_long = (self.pos_long * self.long_entry_price) / Decimal('5') if self.pos_long > 0 else Decimal('0')
        margin_short = (self.pos_short * self.short_entry_price) / Decimal('5') if self.pos_short > 0 else Decimal('0')
        virtual_equity = (self.virt_qty * price) - self.virt_debt
        return self.val_cash + margin_long + unrealized_pnl_long + margin_short + unrealized_pnl_short + virtual_equity

    def get_tpv_fast(self, price: float) -> float:
        """Calculate TPV using fast float arithmetic for performance."""
        pos_l = float(self.pos_long)
        pos_s = float(self.pos_short)
        p = float(price)
        l_entry = float(self.long_entry_price)
        s_entry = float(self.short_entry_price)
        v_qty = float(self.virt_qty)
        v_debt = float(self.virt_debt)
        cash = float(self.val_cash)

        upnl_l = pos_l * (p - l_entry) if pos_l > 0 else 0.0
        upnl_s = pos_s * (s_entry - p) if pos_s > 0 else 0.0
        m_l = (pos_l * l_entry) / 5.0 if pos_l > 0 else 0.0
        m_s = (pos_s * s_entry) / 5.0 if pos_s > 0 else 0.0
        v_eq = (v_qty * p) - v_debt
        
        return cash + m_l + upnl_l + m_s + upnl_s + v_eq

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0) -> str:
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Downloading live data for {symbol} ({days} days)...")

    # Calculate how many 1m candles we need (Binance limit: 1500 per request)
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
                    all_data = data + all_data  # prepend earlier data
                    # Next batch ends where this one starts
                    end_time_ms = data[0][0] - 1
                else:
                    raise Exception(f"Binance API error: {resp.status}")

            # Rate limit courtesy between requests
            if i < num_requests - 1:
                await asyncio.sleep(0.25)

    if not all_data:
        raise Exception(f"No data returned for {symbol}")

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
                        days: float = 2.0, commission: float = 0.0004, slippage: float = 0.0002,
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
            logger.error(f"Data file not found for {base_ticker}")
            return None

        df: pd.DataFrame = pd.read_feather(file_path)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        portfolio_cfg: Dict[str, Any] = config["portfolios"][0]
        
        # Trim data to the specified lookback window (days -> minutes -> bars)
        bars_per_day = 1440  # 1-minute bars
        max_bars = int(days * bars_per_day)
        if len(df) > max_bars:
            df = df.iloc[-max_bars:]
        
        initial_capital: float = capital_override if capital_override is not None else portfolio_cfg.get("initial_capital", 1000.0)
        targets: Dict[str, Any] = portfolio_cfg["targets"]
        ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
        
        # Resolve asymmetric thresholds: overrides > config > defaults
        if threshold_surplus_override is not None:
            t_surplus = threshold_surplus_override
            t_deficit = threshold_deficit_override if threshold_deficit_override is not None else threshold_surplus_override
        else:
            t_surplus = portfolio_cfg.get("rebalance_threshold_surplus", portfolio_cfg.get("rebalance_threshold", 0.02))
            t_deficit = portfolio_cfg.get("rebalance_threshold_deficit", t_surplus)
        
        # Per-ticker override (supports both dict {"surplus": x, "deficit": y} and legacy float)
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
        # Trailing stop parameters (config-based)
        trailing_stop_pct: float = config.get("equity_trailing_stop_pct", 0.0)
        trailing_stop_activation_pct: float = config.get("equity_trailing_stop_activation_pct", 0.0)
        trailing_stop_timeout_sec: int = int(config.get("equity_trailing_stop_timeout_sec", 0))
        min_notional_usdt: Decimal = Decimal(str(config.get("min_notional_usdt", 6.0)))
        
        close_prices: npt.NDArray[np.float64] = df['close'].values.astype(np.float64)

        # Step sizes (mocked for backtest, usually from exchange_info)
        step_sizes = {base_ticker: 0.001}

        sim = MarketOrderSlippageSimulator(commission_pct=commission, slippage_pct=slippage)
        state = BacktestState(initial_capital=Decimal(str(initial_capital)), base_ticker=base_ticker, val_cash=Decimal(str(initial_capital)))

        # Pre-calculate constants
        daily_funding_rate = Decimal('0.0002')
        bars_per_day = Decimal('1440') # Assuming 1-minute bars
        funding_drag_step = daily_funding_rate / bars_per_day
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"]))
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"]))

        # First bar initialization (align with main.py)
        start_price = Decimal(str(close_prices[0]))
        state.last_rebalance_price = start_price
        target_v_share = Decimal(str(targets["VIRTUAL"]["share"]))
        virt_cost = target_v_share * state.initial_capital
        state.virt_qty = virt_cost / start_price
        state.virt_debt = virt_cost
        # ВАЖНО: В бэктесте согласно Патчу №2 покупка Virtual не уменьшает val_cash напрямую
        # val_cash остается равным initial_capital, а Virtual живет в своем "долге".

        for i in range(len(df)):
            mid_price = Decimal(str(close_prices[i]))

            # --- 2. Portfolio Calculation ---
            # Согласно логике main.py, мы передаем в калькулятор real_equity = WalletBalance - virt_debt
            # Wallet Balance = val_cash + margin_long + margin_short
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

            # Pass asymmetric thresholds
            calc_res = calculator.calculate_rebalance(targets, threshold_surplus=threshold_surplus, threshold_deficit=threshold_deficit)
            actions = calc_res["actions"]

            if actions:
                reductions = [a for a in actions if a.get("is_reduction", False)]
                expansions = [a for a in actions if not a.get("is_reduction", False)]

                # --- Фаза 1: Выполнение Reductions (SELL для Лонга, BUY для Шорта) ---
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
                        
                        # Вычисляем среднюю историческую цену входа виртуальной ноги перед продажей
                        # Если virt_qty равен 0 (защита от ZeroDivision), берем текущую цену
                        v_entry_price = (state.virt_debt / state.virt_qty) if state.virt_qty > 0 else exec_price
                        
                        # Историческая себестоимость продаваемых монет (то, на сколько реально уменьшается долг)
                        allocated_debt_reduction = qty * v_entry_price
                        
                        # Физическая прибыль от фиксации профицита на споте (MTM Realized PnL)
                        realized_pnl = qty * (exec_price - v_entry_price)
                        
                        # Обновляем состояние виртуальной ноги
                        state.virt_qty -= qty
                        state.virt_debt -= allocated_debt_reduction
                        
                        # Деньги физически возвращаются в кэш: себестоимость + прибыль - комиссия
                        state.val_cash += allocated_debt_reduction + realized_pnl - commission

                # --- Фаза 2: Выполнение Expansions (BUY для Лонга, SELL для Шорт) ---
                expansions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)

                # Use val_cash (initial cash) for backtest, not available_funds (remaining after internal calc)
                remaining_funds = Decimal(str(calc_res.get("val_cash", calc_res.get("available_funds", 0.0))))

                for act in expansions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    needed_usdt = abs(Decimal(str(act["diff_usdt"])))

                    if remaining_funds <= Decimal('0'):
                        state.skipped_expansions_counter += 1
                        continue

                    # Урезаем qty если не хватает средств
                    max_usdt = remaining_funds * lev
                    if needed_usdt > max_usdt:
                        needed_usdt = max_usdt

                    raw_qty = needed_usdt / mid_price
                    qty = quantize_qty(raw_qty, Decimal(str(step_sizes.get(base_ticker, 0.001))))

                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue

                    # Списываем equity из remaining_funds (до расчёта exec_price)
                    # Equity = notional / leverage (VIRTUAL: lev=1, equity = cash_spent)
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

            # --- Расчет Funding Drag (Удержание за фьючерсные плечи) ---
            long_notional = state.pos_long * mid_price
            short_notional = state.pos_short * mid_price
            total_drag = (long_notional + short_notional) * funding_drag_step
            state.val_cash -= total_drag

            # --- 2.5 Predictive Liquidation Guard ---
            # Mirror live Guard #4: close position BEFORE exchange liquidation.
            # Uses Binance isolated margin formula for liquidation price estimation.
            liq_distance_warn_pct = portfolio_cfg.get("liquidation_distance_warn_pct", 15.0)
            liq_distance_crit_pct = portfolio_cfg.get("liquidation_distance_crit_pct", 8.0)
            # Binance maintenance margin rate for 5x leverage ≈ 0.4%
            mmr = Decimal('0.004')

            if state.pos_long > 0 and state.long_entry_price > 0:
                liq_long = state.long_entry_price * (Decimal('1') - Decimal(str(1/l_lev)) + mmr)
                dist_long = (mid_price - liq_long) / mid_price * Decimal('100')
                if dist_long <= Decimal(str(liq_distance_crit_pct)):
                    logger.warning(f"Bar {i}: Predictive LONG liq guard! dist={float(dist_long):.1f}%, closing.")
                    # Close at market: recover remaining margin after unrealized loss
                    notional = state.pos_long * state.long_entry_price
                    unrealized_pnl = state.pos_long * (mid_price - state.long_entry_price)
                    margin = notional / l_lev
                    state.val_cash += margin + unrealized_pnl
                    state.pos_long = Decimal('0')
                    state.long_entry_price = Decimal('0')
                    state.liquidations_counter += 1
                elif dist_long <= Decimal(str(liq_distance_warn_pct)):
                    logger.debug(f"Bar {i}: LONG liq warning, dist={float(dist_long):.1f}%")

            if state.pos_short > 0 and state.short_entry_price > 0:
                liq_short = state.short_entry_price * (Decimal('1') + Decimal(str(1/s_lev)) - mmr)
                dist_short = (liq_short - mid_price) / mid_price * Decimal('100')
                if dist_short <= Decimal(str(liq_distance_crit_pct)):
                    logger.warning(f"Bar {i}: Predictive SHORT liq guard! dist={float(dist_short):.1f}%, closing.")
                    notional = state.pos_short * state.short_entry_price
                    unrealized_pnl = state.pos_short * (state.short_entry_price - mid_price)
                    margin = notional / s_lev
                    state.val_cash += margin + unrealized_pnl
                    state.pos_short = Decimal('0')
                    state.short_entry_price = Decimal('0')
                    state.liquidations_counter += 1
                elif dist_short <= Decimal(str(liq_distance_warn_pct)):
                    logger.debug(f"Bar {i}: SHORT liq warning, dist={float(dist_short):.1f}%")

            # --- 3. Post-Factum Liquidation Check (Per Leg) ---
            unrealized_pnl_long = state.pos_long * (mid_price - state.long_entry_price) if state.pos_long > 0 else Decimal('0')
            unrealized_pnl_short = state.pos_short * (state.short_entry_price - mid_price) if state.pos_short > 0 else Decimal('0')
            margin_long = (state.pos_long * state.long_entry_price) / l_lev if state.pos_long > 0 else Decimal('0')
            margin_short = (state.pos_short * state.short_entry_price) / s_lev if state.pos_short > 0 else Decimal('0')

            val_l = margin_long + unrealized_pnl_long
            val_s = margin_short + unrealized_pnl_short

            if val_l <= 0 and state.pos_long != 0:
                state.pos_long = Decimal('0')
                state.long_entry_price = Decimal('0')
                state.liquidations_counter += 1
                logger.warning(f"Bar {i}: LONG leg liquidated!")

            if val_s <= 0 and state.pos_short != 0:
                state.pos_short = Decimal('0')
                state.short_entry_price = Decimal('0')
                state.liquidations_counter += 1
                logger.warning(f"Bar {i}: SHORT leg liquidated!")

            # --- 4. SAFE Siphoning (Calculated on TPV) ---
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

            # --- 4b. Trailing Stop (Equity-based) ---
            # Update ATH
            if total_tpv_with_reserve > state.tpv_ath:
                state.tpv_ath = total_tpv_with_reserve
                state.trailing_stop_violation_start = 0.0

            # Check trailing stop only if ATH is above activation threshold
            if (trailing_stop_pct > 0
                    and state.tpv_ath > initial_capital_f * (1 + trailing_stop_activation_pct / 100.0)):
                drawdown_from_ath = (1 - total_tpv_with_reserve / state.tpv_ath) * 100.0
                if drawdown_from_ath >= trailing_stop_pct:
                    if state.trailing_stop_violation_start == 0:
                        state.trailing_stop_violation_start = float(i)
                        logger.warning(
                            f"Bar {i}: Trailing Stop threshold breached "
                            f"(DD {drawdown_from_ath:.2f}% from ATH). "
                            f"Timeout: {trailing_stop_timeout_sec}s"
                        )
                    else:
                        elapsed_bars = i - int(state.trailing_stop_violation_start)
                        # Convert timeout_sec to bars (1 bar = 1 minute)
                        timeout_bars = max(1, trailing_stop_timeout_sec // 60)
                        if elapsed_bars >= timeout_bars:
                            logger.warning(
                                f"Bar {i}: Trailing Stop triggered "
                                f"(DD {drawdown_from_ath:.2f}% for {elapsed_bars} bars / {trailing_stop_timeout_sec}s). "
                                f"Stopping backtest."
                            )
                            state.trailing_stop_triggered = True
                            state.history.append(float(total_tpv_with_reserve))
                            break
                else:
                    # Recovery: reset violation timer if drawdown is back under threshold
                    if state.trailing_stop_violation_start > 0:
                        logger.info(
                            f"Bar {i}: Trailing Stop recovered "
                            f"(DD {drawdown_from_ath:.2f}% < {trailing_stop_pct}%)"
                        )
                        state.trailing_stop_violation_start = 0.0

            # --- 5. Проверка математического инварианта (Sanity Check) ---
            # Настоящий TPV = Свободный кэш + Чистая стоимость LONG + Чистая стоимость SHORT + Чистая стоимость VIRTUAL
            unrealized_pnl_long = state.pos_long * (mid_price - state.long_entry_price) if state.pos_long > 0 else Decimal('0')
            unrealized_pnl_short = state.pos_short * (state.short_entry_price - mid_price) if state.pos_short > 0 else Decimal('0')
            margin_long = (state.pos_long * state.long_entry_price) / Decimal('5') if state.pos_long > 0 else Decimal('0')
            margin_short = (state.pos_short * state.short_entry_price) / Decimal('5') if state.pos_short > 0 else Decimal('0')
            virtual_equity = (state.virt_qty * mid_price) - state.virt_debt

            current_tpv = state.val_cash + margin_long + unrealized_pnl_long + margin_short + unrealized_pnl_short + virtual_equity
            assert current_tpv > Decimal('0'), "Критический дефолт портфеля: TPV <= 0"

            state.history.append(float(total_tpv_with_reserve))

        equity_curve: npt.NDArray[np.float64] = np.array(state.history)
        profit_pct: float = (equity_curve[-1] / (initial_capital + 1e-9) - 1) * 100
        max_eq: npt.NDArray[np.float64] = np.maximum.accumulate(equity_curve)
        dd: npt.NDArray[np.float64] = (max_eq - equity_curve) / (max_eq + 1e-9)
        max_dd_pct: float = np.max(dd) * 100 if len(dd) > 0 else 0.0

        asset_chg_pct: float = (close_prices[-1] / (close_prices[0] + 1e-9) - 1) * 100

        if not quiet:
            logger.info(f"Refactored Synchronized Backtest for {base_ticker}:")
            logger.info(f"Profit: {profit_pct:+.2f}% | MaxDD: {max_dd_pct:.2f}% | Cycles: {state.cycles}")
            logger.info(f"Asset Change: {asset_chg_pct:+.2f}% | SAFE Reserve: {state.siphoning_reserve:.2f} USDT")
            logger.info(f"Skipped Expansions: {state.skipped_expansions_counter} | Liquidations: {state.liquidations_counter}")
            if state.trailing_stop_triggered:
                logger.info(f"*** TRAILING STOP TRIGGERED at bar {len(state.history)} ***")

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
            "trailing_stop_triggered": state.trailing_stop_triggered
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
    # Assuming the script is in 'futures_portfolio' and 'data' is a subfolder
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days,
                             capital_override=args.capital,
                             threshold_surplus_override=args.threshold_surplus,
                             threshold_deficit_override=args.threshold_deficit))
