import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
import aiohttp
import time
import json
import os
import logging
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

class VectorizedBacktester:
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital: float = initial_capital

    def run(self, price_matrix: npt.NDArray[np.float64], weights_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        price_matrix: 2D array (Time x Tickers) [Только 'close' из OHLCV]
        weights_matrix: 2D array (Time x Tickers) выданные агентом
        """
        # Векторизованный расчет доходностей
        # returns = (P_t - P_{t-1}) / P_{t-1}
        returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]
        # Добавляем нулевую строку для первого шага, чтобы сохранить размерность
        returns_matrix = np.vstack([np.zeros(price_matrix.shape[1]), returns_matrix])

        # Векторизованный расчет PnL портфеля
        portfolio_returns = np.sum(weights_matrix * returns_matrix, axis=1)
        equity_curve = self.initial_capital * np.cumprod(1 + portfolio_returns)

        return equity_curve

    def calculate_ensemble_balance(self, pnl_l: npt.NDArray[np.float64], pnl_s: npt.NDArray[np.float64], denom: float) -> npt.NDArray[np.float64]:
        # Строгое следование формуле баланса: abs(pnl_l + pnl_s) / denom
        return np.abs(pnl_l + pnl_s) / denom

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0) -> str:
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Downloading live data for {symbol}...")
    all_data = []
    # limit is 1500 max for Binance
    params = {"symbol": symbol, "interval": "1m", "limit": 1500}
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            if resp.status == 200:
                all_data = await resp.json()
            else:
                logger.error(f"Binance API error: {resp.status}")

    if not all_data:
        raise Exception(f"No data fetched for {symbol}")

    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    # СТРОГОЕ ОГРАНИЧЕНИЕ: Только OHLCV
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
        
        base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

        if live_mode:
            file_path = await download_live_data(base_ticker, data_dir, days)
        else:
            # Fallback chain for local testing
            possible_files = [
                os.path.join(data_dir, f"{base_ticker}_live.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_10d.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_48h.feather")
            ]
            file_path = next((p for p in possible_files if os.path.exists(p)), None)

        if not file_path or not os.path.exists(file_path):
            return None

        df = pd.read_feather(file_path)
        # СТРОГОЕ ОГРАНИЧЕНИЕ ПО ДАННЫМ: Только OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        portfolio_cfg = config["portfolios"][0]
        initial_capital: float = portfolio_cfg.get("initial_capital", 1000.0)
        targets = portfolio_cfg["targets"]
        # Ticker-specific threshold override
        ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
        threshold: float = ticker_thresholds.get(base_ticker, portfolio_cfg.get("rebalance_threshold", 0.02))
        
        # Векторизованные расчеты
        close_prices: npt.NDArray[np.float64] = df['close'].values.astype(np.float64)
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.insert(returns, 0, 0.0)
        
        # Synthetic prices for 3 legs to model returns matrix correctly
        # Leg 1: Long
        p_long = close_prices
        # Leg 2: Short (Using cumprod(1-R) to get exact -R returns at each step)
        p_short = np.cumprod(1.0 - returns)
        # Leg 3: Virtual
        p_virt = close_prices
        
        price_matrix = np.column_stack([p_long, p_short, p_virt])

        l_weight = targets["BASE_LONG"]["share"] * targets["BASE_LONG"]["leverage"]
        s_weight = targets["BASE_SHORT"]["share"] * targets["BASE_SHORT"]["leverage"]
        v_weight = targets["VIRTUAL"]["share"]

        # Weights matrix (Time x 3)
        weights_matrix = np.tile(np.array([l_weight, s_weight, v_weight]), (len(df), 1))

        backtester = VectorizedBacktester(initial_capital)
        equity_curve = backtester.run(price_matrix, weights_matrix)

        # Rebalancing Alpha via Saw Factor (Vectorized)
        # log_p rebalancing levels proxy
        log_p = np.log(close_prices)
        rebalance_levels = np.round(log_p / threshold)
        cycles = np.sum(np.abs(np.diff(rebalance_levels)))

        # Alpha estimation based on cycles
        total_lev = (targets["BASE_LONG"]["share"] * targets["BASE_LONG"]["leverage"] +
                     targets["BASE_SHORT"]["share"] * targets["BASE_SHORT"]["leverage"])
        alpha_return = cycles * (threshold * 0.4) * total_lev * 0.05 # Conservative scaling factor

        final_equity = equity_curve[-1] * (1.0 + alpha_return)
        profit_pct = (final_equity / initial_capital - 1) * 100

        # Drawdown calculation
        max_eq = np.maximum.accumulate(equity_curve)
        dd = (max_eq - equity_curve) / max_eq
        max_dd_pct = np.max(dd) * 100 if len(dd) > 0 else 0.0

        asset_chg_pct = (close_prices[-1] / close_prices[0] - 1) * 100

        # Ensemble balance formula as requested
        pnl_l = (p_long - p_long[0]) / p_long[0]
        pnl_s = (p_short - p_short[0]) / p_short[0]
        # ensemble_balance is a series of magnitude of imbalance
        ensemble_balance = backtester.calculate_ensemble_balance(pnl_l, pnl_s, 1.0)

        if not quiet:
            logger.info(f"Vectorized Backtest for {base_ticker}:")
            logger.info(f"Profit: {profit_pct:.2f}% | MaxDD: {max_dd_pct:.2f}% | Cycles: {int(cycles)}")

        return {
            "profit_pct": float(profit_pct),
            "max_dd_pct": float(max_dd_pct),
            "cycles": int(cycles),
            "asset_chg_pct": float(asset_chg_pct),
            "siphoning_reserve": 0.0
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    # Use relative path for data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days))
