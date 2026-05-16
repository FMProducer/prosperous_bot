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
    history: List[float] = field(default_factory=list)
    rebalance_log: List[Dict[str, Any]] = field(default_factory=list)

    def get_tpv(self, price: Decimal) -> Decimal:
        unrealized_pnl_long = self.pos_long * (price - self.long_entry_price) if self.pos_long > 0 else Decimal('0')
        unrealized_pnl_short = self.pos_short * (self.short_entry_price - price) if self.pos_short > 0 else Decimal('0')
        margin_long = (self.pos_long * self.long_entry_price) / Decimal('5') if self.pos_long > 0 else Decimal('0')
        margin_short = (self.pos_short * self.short_entry_price) / Decimal('5') if self.pos_short > 0 else Decimal('0')
        virtual_equity = (self.virt_qty * price) - self.virt_debt
        return self.val_cash + margin_long + unrealized_pnl_long + margin_short + unrealized_pnl_short + virtual_equity

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0) -> str:
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Downloading live data for {symbol}...")
    params = {"symbol": symbol, "interval": "1m", "limit": 1500}
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            if resp.status == 200:
                all_data = await resp.json()
            else:
                raise Exception(f"Binance API error: {resp.status}")

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
                        quiet: bool = False, threshold_override: Optional[float] = None,
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
        initial_capital: float = capital_override if capital_override is not None else portfolio_cfg.get("initial_capital", 1000.0)
        targets: Dict[str, Any] = portfolio_cfg["targets"]
        ticker_thresholds: Dict[str, float] = portfolio_cfg.get("ticker_thresholds", {})
        
        if threshold_override is not None:
            threshold = threshold_override
        else:
            threshold = ticker_thresholds.get(base_ticker, portfolio_cfg.get("rebalance_threshold", 0.02))
            
        siphoning_threshold_pct: float = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio: float = portfolio_cfg.get("reinvestment_ratio", 0.0)
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
                initial_capital=float(state.initial_capital)
            )

            calc_res = calculator.calculate_rebalance(targets, threshold)
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
                        state.virt_qty -= qty
                        state.virt_debt -= (qty * exec_price) - commission

                # --- Фаза 2: Выполнение Expansions (BUY для Лонга, SELL для Шорта) ---
                expansions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)

                for act in expansions:
                    key = act["key"]
                    lev = Decimal(str(act["leverage"]))
                    needed_usdt = abs(Decimal(str(act["diff_usdt"])))

                    if state.val_cash <= Decimal('0'):
                        state.skipped_expansions_counter += 1
                        continue

                    raw_qty = needed_usdt / mid_price
                    qty = quantize_qty(raw_qty, Decimal(str(step_sizes.get(base_ticker, 0.001))))

                    if qty <= Decimal('0') or not validate_notional(qty, mid_price, min_notional_usdt):
                        continue

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
                        state.virt_qty += qty
                        state.virt_debt += (qty * exec_price) + commission

                state.cycles += 1
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

            # --- 3. Liquidation Check (Per Leg) ---
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
            tpv = state.get_tpv(mid_price)
            total_tpv_with_reserve = tpv + state.siphoning_reserve
            total_surplus = total_tpv_with_reserve - state.initial_capital
            siphoning_threshold_abs = state.initial_capital * (Decimal(str(siphoning_threshold_pct)) / 100)

            if total_surplus > state.siphoning_reserve + max(Decimal('0.1'), siphoning_threshold_abs):
                new_profit = total_surplus - state.siphoning_reserve
                siphon_amount = new_profit * (1 - Decimal(str(reinvestment_ratio)))
                if siphon_amount > Decimal('0.1'):
                    state.siphoning_reserve += siphon_amount
                    state.val_cash -= siphon_amount
                    tpv -= siphon_amount
                    total_tpv_with_reserve -= siphon_amount

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
            "liquidations": int(state.liquidations_counter)
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
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    # Assuming the script is in 'futures_portfolio' and 'data' is a subfolder
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days, capital_override=args.capital, threshold_override=args.threshold))
