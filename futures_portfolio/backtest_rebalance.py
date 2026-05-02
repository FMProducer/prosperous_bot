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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

class LimitOrderSimulator:
    """Симулятор лимитных ордеров для учета рыночных фрикций."""
    def __init__(self, commission_pct: float = 0.0004, offset_pct: float = 0.1):
        self.commission_pct: float = commission_pct
        self.offset_pct: float = offset_pct
        self.stats: Dict[str, Any] = {"attempted": 0, "filled": 0, "fallback": 0}

    def simulate_limit_execution(self, side: str, qty: float, mid_price: float,
                                  candle_high: float, candle_low: float) -> Tuple[bool, float, float, str]:
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
        self.initial_capital: float = initial_capital
        self.commission: float = commission
        self.targets: Dict[str, Any] = targets
        self.threshold: float = threshold
        self.siphoning_threshold_pct: float = siphoning_threshold_pct
        self.reinvestment_ratio: float = reinvestment_ratio

        self.current_equity: float = initial_capital
        self.siphoning_reserve: float = 0.0

        self.l_target: float = targets["BASE_LONG"]["share"]
        self.s_target: float = targets["BASE_SHORT"]["share"]
        self.v_target: float = targets["VIRTUAL"]["share"]
        self.l_lev: float = targets["BASE_LONG"]["leverage"]
        self.s_lev: float = targets["BASE_SHORT"]["leverage"]

        self.positions: Dict[str, float] = {"LONG": 0.0, "SHORT": 0.0}
        self.entry_prices: Dict[str, float] = {"LONG": 0.0, "SHORT": 0.0}
        self.virt_basis_price: float = 0.0
        self.virt_allocated_usdt: float = 0.0

        self.history: List[float] = []
        self.cycles: int = 0

    def init_state(self, price: float) -> None:
        """Инициализация начальных позиций."""
        self.positions["LONG"] = (self.initial_capital * self.l_target * self.l_lev) / (price + 1e-9)
        self.positions["SHORT"] = (self.initial_capital * self.s_target * self.s_lev) / (price + 1e-9)
        self.entry_prices["LONG"] = price
        self.entry_prices["SHORT"] = price

        init_comm = (abs(self.positions["LONG"] * price) + abs(self.positions["SHORT"] * price)) * self.commission
        self.current_equity -= init_comm

        self.virt_basis_price = price
        self.virt_allocated_usdt = self.initial_capital * self.v_target

    def update(self, price: float, high: float, low: float, sim: Optional[LimitOrderSimulator] = None) -> float:
        l_pnl: float = self.positions["LONG"] * (price - self.entry_prices["LONG"])
        s_pnl: float = self.positions["SHORT"] * (self.entry_prices["SHORT"] - price)
        virt_pnl: float = self.virt_allocated_usdt * (price / (self.virt_basis_price + 1e-9) - 1)

        total_tpv: float = self.current_equity + l_pnl + s_pnl + virt_pnl + self.siphoning_reserve

        if total_tpv <= 0:
            self.history.append(0.0)
            return 0.0

        # SAFE Siphoning logic (Sync with main.py)
        active_part: float = self.current_equity + l_pnl + s_pnl + virt_pnl
        total_surplus: float = active_part - self.initial_capital
        siphoning_threshold_abs: float = self.initial_capital * (self.siphoning_threshold_pct / 100)

        if total_surplus > max(0.1, siphoning_threshold_abs):
            siphon_amount: float = total_surplus * (1 - self.reinvestment_ratio)
            if siphon_amount > 0.1:
                self.siphoning_reserve += siphon_amount
                self.current_equity -= siphon_amount
                active_part -= siphon_amount

        active_tpv: float = min(total_tpv, self.initial_capital)

        # Rebalance Check
        val_l: float = (self.positions["LONG"] * price) / self.l_lev
        val_s: float = (self.positions["SHORT"] * price) / self.s_lev
        val_virt: float = self.virt_allocated_usdt * (price / (self.virt_basis_price + 1e-9))

        share_l: float = val_l / (active_tpv + 1e-9) if active_tpv > 0 else 0
        share_s: float = val_s / (active_tpv + 1e-9) if active_tpv > 0 else 0
        share_v: float = val_virt / (active_tpv + 1e-9) if active_tpv > 0 else 0

        if not math.isclose(share_l, self.l_target, abs_tol=self.threshold) or \
           not math.isclose(share_s, self.s_target, abs_tol=self.threshold) or \
           not math.isclose(share_v, self.v_target, abs_tol=self.threshold):

            self.cycles += 1
            # Realize PnL
            self.current_equity += l_pnl + s_pnl + virt_pnl

            # Rebalance Long
            target_vol_l: float = active_tpv * self.l_target * self.l_lev
            diff_usdt_l: float = target_vol_l - (self.positions["LONG"] * price)
            if abs(diff_usdt_l) > 5.0:
                side = "BUY" if diff_usdt_l > 0 else "SELL"
                f_price = price
                if sim:
                    _, _, f_price, _ = sim.simulate_limit_execution(side, abs(diff_usdt_l)/(price + 1e-9), price, high, low)
                self.current_equity -= abs(diff_usdt_l) * self.commission
                self.positions["LONG"] = target_vol_l / (f_price + 1e-9)
                self.entry_prices["LONG"] = f_price
            else:
                self.entry_prices["LONG"] = price

            # Rebalance Short
            target_vol_s: float = active_tpv * self.s_target * self.s_lev
            diff_usdt_s: float = target_vol_s - (self.positions["SHORT"] * price)
            if abs(diff_usdt_s) > 5.0:
                side = "SELL" if diff_usdt_s > 0 else "BUY"
                f_price = price
                if sim:
                    _, _, f_price, _ = sim.simulate_limit_execution(side, abs(diff_usdt_s)/(price + 1e-9), price, high, low)
                self.current_equity -= abs(diff_usdt_s) * self.commission
                self.positions["SHORT"] = target_vol_s / (f_price + 1e-9)
                self.entry_prices["SHORT"] = f_price
            else:
                self.entry_prices["SHORT"] = price

            self.virt_basis_price = price
            self.virt_allocated_usdt = active_tpv * self.v_target

        self.history.append(total_tpv)
        return total_tpv

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
                        limit_offset_pct: float = 0.1, limit_timeout_sec: int = 30, quiet: bool = False) -> Optional[Dict[str, Any]]:
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
        threshold: float = ticker_thresholds.get(base_ticker, portfolio_cfg.get("rebalance_threshold", 0.02))
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
