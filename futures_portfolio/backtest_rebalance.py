import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple, Any, Union
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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class LimitOrderSimulator:
    """Симулятор лимитных ордеров для учета рыночных фрикций."""
    def __init__(self, commission_pct: float = 0.0004, offset_pct: float = 0.1):
        self.commission_pct = Decimal(str(commission_pct))
        self.offset_pct = Decimal(str(offset_pct))
        self.stats: Dict[str, Any] = {"attempted": 0, "filled": 0, "fallback": 0}

    def simulate_limit_execution(self, side: str, qty: Decimal, mid_price: Decimal,
                                  candle_high: Decimal, candle_low: Decimal) -> Tuple[bool, Decimal, Decimal, str]:
        self.stats["attempted"] += 1
        if side == "SELL":
            limit_price = mid_price * (1 + self.offset_pct / 100)
            if candle_high >= limit_price:
                self.stats["filled"] += 1
                return True, qty, max(limit_price, mid_price), "LIMIT_FILLED"
            self.stats["fallback"] += 1
            return True, qty, candle_low, "FALLBACK_MARKET"
        else: # BUY
            limit_price = mid_price * (1 - self.offset_pct / 100)
            if candle_low <= limit_price:
                self.stats["filled"] += 1
                return True, qty, min(limit_price, mid_price), "LIMIT_FILLED"
            self.stats["fallback"] += 1
            return True, qty, candle_high, "FALLBACK_MARKET"

    def get_summary(self) -> Dict[str, Any]:
        return self.stats

class PortfolioState:
    """Управление состоянием портфеля во время бэктеста."""
    def __init__(self, initial_capital: float, commission: float, targets: Dict[str, Any], threshold: float,
                 siphoning_threshold_pct: float = 0.0, reinvestment_ratio: float = 0.0):
        self.initial_capital = Decimal(str(initial_capital))
        self.commission = Decimal(str(commission))
        self.targets = targets
        self.threshold = Decimal(str(threshold))
        self.siphoning_threshold_pct = Decimal(str(siphoning_threshold_pct))
        self.reinvestment_ratio = Decimal(str(reinvestment_ratio))

        self.real_equity = self.initial_capital
        self.siphoning_reserve = Decimal('0.0')

        self.l_target = Decimal(str(targets["BASE_LONG"]["share"]))
        self.s_target = Decimal(str(targets["BASE_SHORT"]["share"]))
        self.v_target = Decimal(str(targets["VIRTUAL"]["share"]))
        self.l_lev = Decimal(str(targets["BASE_LONG"]["leverage"]))
        self.s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"]))

        self.positions: Dict[str, Decimal] = {"LONG": Decimal('0.0'), "SHORT": Decimal('0.0')}
        self.entry_prices: Dict[str, Decimal] = {"LONG": Decimal('0.0'), "SHORT": Decimal('0.0')}
        
        self.virt_basis_price = Decimal('0.0')
        self.virt_allocated_usdt = Decimal('0.0')

        self.history: List[float] = []
        self.cycles: int = 0

    def init_state(self, price: float) -> None:
        """Инициализация начальных позиций."""
        dec_price = Decimal(str(price))
        # Initial allocation based on target shares of initial capital
        self.positions["LONG"] = (self.initial_capital * self.l_target * self.l_lev) / dec_price
        self.positions["SHORT"] = (self.initial_capital * self.s_target * self.s_lev) / dec_price
        self.entry_prices["LONG"] = dec_price
        self.entry_prices["SHORT"] = dec_price

        # Initial commission for entry
        init_comm = (abs(self.positions["LONG"] * dec_price) + abs(self.positions["SHORT"] * dec_price)) * self.commission
        self.real_equity -= init_comm

        self.virt_basis_price = dec_price
        self.virt_allocated_usdt = self.initial_capital * self.v_target

    def update(self, price: float, high: float, low: float, sim: Optional[LimitOrderSimulator] = None) -> float:
        dec_price = Decimal(str(price))
        dec_high = Decimal(str(high))
        dec_low = Decimal(str(low))

        # 1. Update Real Equity based on futures PnL
        l_pnl = self.positions["LONG"] * (dec_price - self.entry_prices["LONG"])
        s_pnl = self.positions["SHORT"] * (self.entry_prices["SHORT"] - dec_price)
        current_real_equity = self.real_equity + l_pnl + s_pnl

        # 2. Update Virtual Value (Recalculated every iteration)
        # V_current = Qty_virt * Price_current = (Allocated / Basis) * Price_current
        virt_current_value = self.virt_allocated_usdt * (dec_price / self.virt_basis_price)
        virt_pnl = virt_current_value - self.virt_allocated_usdt

        # 3. Calculate Global TPV
        # TPV = Real_Equity + Virtual_PnL
        tpv = current_real_equity + virt_pnl
        total_tpv = tpv + self.siphoning_reserve

        if total_tpv <= 0:
            self.history.append(0.0)
            return 0.0

        # 4. SAFE Siphoning logic (Triggered if global total_tpv > initial_capital)
        total_surplus = total_tpv - self.initial_capital
        siphoning_threshold_abs = self.initial_capital * (self.siphoning_threshold_pct / 100)

        if total_surplus > self.siphoning_reserve + max(Decimal('0.1'), siphoning_threshold_abs):
            new_profit = total_surplus - self.siphoning_reserve
            siphon_amount = new_profit * (1 - self.reinvestment_ratio)
            if siphon_amount > Decimal('0.1'):
                self.siphoning_reserve += siphon_amount
                # Reserve is taken from real_equity
                self.real_equity -= siphon_amount
                current_real_equity -= siphon_amount
                tpv -= siphon_amount

        # 5. Rebalance Check
        # Use actual Market Value calculation (Basis + PnL)
        val_l = (self.positions["LONG"] * self.entry_prices["LONG"] / self.l_lev) + \
                (self.positions["LONG"] * (dec_price - self.entry_prices["LONG"]))
        
        val_s = (self.positions["SHORT"] * self.entry_prices["SHORT"] / self.s_lev) + \
                (self.positions["SHORT"] * (self.entry_prices["SHORT"] - dec_price))
        
        val_v = virt_current_value

        # Shares relative to working capital (TPV)
        share_l = val_l / tpv if tpv > 0 else Decimal('0')
        share_s = val_s / tpv if tpv > 0 else Decimal('0')
        share_v = val_v / tpv if tpv > 0 else Decimal('0')

        if abs(share_l - self.l_target) > self.threshold or \
           abs(share_s - self.s_target) > self.threshold or \
           abs(share_v - self.v_target) > self.threshold:

            # Rebalance triggers
            rebalanced = False
            
            # Action candidate: Long
            notional_l = self.positions["LONG"] * dec_price
            target_vol_l = tpv * self.l_target * self.l_lev
            diff_usdt_l = target_vol_l - notional_l
            if abs(diff_usdt_l) >= Decimal('6.0'):
                self.cycles += 1
                rebalanced = True
                side = "BUY" if diff_usdt_l > 0 else "SELL"
                f_price = dec_price
                if sim:
                    _, _, f_price, _ = sim.simulate_limit_execution(side, abs(diff_usdt_l)/dec_price, dec_price, dec_high, dec_low)
                
                # Realize PnL for the leg partially
                self.real_equity += self.positions["LONG"] * (dec_price - self.entry_prices["LONG"])
                self.real_equity -= abs(diff_usdt_l) * self.commission
                self.positions["LONG"] = target_vol_l / f_price
                self.entry_prices["LONG"] = f_price

            # Action candidate: Short
            notional_s = self.positions["SHORT"] * dec_price
            target_vol_s = tpv * self.s_target * self.s_lev
            diff_usdt_s = target_vol_s - notional_s
            if abs(diff_usdt_s) >= Decimal('6.0'):
                if not rebalanced: self.cycles += 1
                rebalanced = True
                side = "SELL" if diff_usdt_s > 0 else "BUY"
                f_price = dec_price
                if sim:
                    _, _, f_price, _ = sim.simulate_limit_execution(side, abs(diff_usdt_s)/dec_price, dec_price, dec_high, dec_low)
                
                self.real_equity += self.positions["SHORT"] * (self.entry_prices["SHORT"] - dec_price)
                self.real_equity -= abs(diff_usdt_s) * self.commission
                self.positions["SHORT"] = target_vol_s / f_price
                self.entry_prices["SHORT"] = f_price

            # Action candidate: Virtual
            diff_usdt_v = (tpv * self.v_target) - virt_current_value
            if abs(diff_usdt_v) >= Decimal('6.0'):
                if not rebalanced: self.cycles += 1
                rebalanced = True
                # Reset virtual basis
                self.virt_basis_price = dec_price
                self.virt_allocated_usdt = tpv * self.v_target

        self.history.append(float(total_tpv))
        return float(total_tpv)

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
                        days: float = 2.0, commission: float = 0.0004, use_limit_orders: bool = False,
                        limit_offset_pct: float = 0.1, limit_timeout_sec: int = 30, quiet: bool = False,
                        threshold_override: Optional[float] = None) -> Optional[Dict[str, Any]]:
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
        initial_capital: float = portfolio_cfg.get("initial_capital", 1000.0)
        targets: Dict[str, Any] = portfolio_cfg["targets"]
        ticker_thresholds: Dict[str, float] = portfolio_cfg.get("ticker_thresholds", {})
        
        if threshold_override is not None:
            threshold = threshold_override
        else:
            threshold = ticker_thresholds.get(base_ticker, portfolio_cfg.get("rebalance_threshold", 0.02))
            
        siphoning_threshold_pct: float = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio: float = portfolio_cfg.get("reinvestment_ratio", 0.0)
        
        close_prices: npt.NDArray[np.float64] = df['close'].values.astype(np.float64)
        high_prices: npt.NDArray[np.float64] = df['high'].values.astype(np.float64)
        low_prices: npt.NDArray[np.float64] = df['low'].values.astype(np.float64)

        limit_enabled: bool = config.get("limit_order_enabled", use_limit_orders)
        sim: Optional[LimitOrderSimulator] = LimitOrderSimulator(commission_pct=commission, offset_pct=limit_offset_pct) if limit_enabled else None

        state = PortfolioState(initial_capital, commission, targets, threshold, siphoning_threshold_pct, reinvestment_ratio)
        state.init_state(close_prices[0])

        for i in range(len(df)):
            state.update(close_prices[i], high_prices[i], low_prices[i], sim)

        equity_curve: npt.NDArray[np.float64] = np.array(state.history)
        profit_pct: float = (equity_curve[-1] / (initial_capital + 1e-9) - 1) * 100
        max_eq: npt.NDArray[np.float64] = np.maximum.accumulate(equity_curve)
        dd: npt.NDArray[np.float64] = (max_eq - equity_curve) / (max_eq + 1e-9)
        max_dd_pct: float = np.max(dd) * 100 if len(dd) > 0 else 0.0
        asset_chg_pct: float = (close_prices[-1] / (close_prices[0] + 1e-9) - 1) * 100

        if not quiet:
            logger.info(f"Iterative Backtest for {base_ticker}:")
            logger.info(f"Profit: {profit_pct:+.2f}% | MaxDD: {max_dd_pct:.2f}% | Cycles: {state.cycles}")
            logger.info(f"Asset Change: {asset_chg_pct:+.2f}% | SAFE Reserve: {state.siphoning_reserve:.2f} USDT")
            if sim:
                s = sim.get_summary()
                logger.info(f"Limit Orders: {s['filled']}/{s['attempted']} filled ({s['fallback']} fallbacks)")

        return {
            "profit_pct": float(profit_pct),
            "max_dd_pct": float(max_dd_pct),
            "cycles": int(state.cycles),
            "asset_chg_pct": float(asset_chg_pct),
            "siphoning_reserve": float(state.siphoning_reserve)
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
    args = parser.parse_args()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days))
