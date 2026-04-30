import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
import json
import os
import math
import logging
import traceback
import argparse
from typing import Dict, List, Optional, Tuple, Any
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

class LimitOrderSimulator:
    def __init__(self, commission_pct: float = 0.02, timeout_sec: int = 30, offset_pct: float = 0.1):
        self.commission_pct = commission_pct
        self.timeout_sec = timeout_sec
        self.offset_pct = offset_pct
        self.stats = {
            "attempted": 0, "filled": 0, "fallback": 0,
            "total_improvement_usdt": 0.0, "total_improvement_pct": 0.0,
            "missed_opportunities": 0
        }

    def simulate_limit_execution(self, side: str, qty: float, mid_price: float,
                                  candle_high: float, candle_low: float) -> Tuple[bool, float, float, str]:
        self.stats["attempted"] += 1
        if side == "SELL":
            limit_price = mid_price * (1 + self.offset_pct / 100)
            if candle_high >= limit_price:
                fill_price = max(limit_price, mid_price)
                improvement = fill_price - mid_price
                self.stats["filled"] += 1
                self.stats["total_improvement_usdt"] += improvement * qty
                self.stats["total_improvement_pct"] += (improvement / mid_price) * 100
                return True, qty, fill_price, "LIMIT_FILLED"
            else:
                self.stats["fallback"] += 1
                # ХУДШИЙ сценарий для продавца - продажа по LOW свечи
                return True, qty, candle_low, "FALLBACK_MARKET"
        else: # BUY
            limit_price = mid_price * (1 - self.offset_pct / 100)
            if candle_low <= limit_price:
                fill_price = min(limit_price, mid_price)
                improvement = mid_price - fill_price
                self.stats["filled"] += 1
                self.stats["total_improvement_usdt"] += improvement * qty
                self.stats["total_improvement_pct"] += (improvement / mid_price) * 100
                return True, qty, fill_price, "LIMIT_FILLED"
            else:
                self.stats["fallback"] += 1
                # ХУДШИЙ сценарий для покупателя - покупка по HIGH свечи
                return True, qty, candle_high, "FALLBACK_MARKET"

    def get_summary(self) -> Dict:
        fill_rate = self.stats["filled"] / self.stats["attempted"] if self.stats["attempted"] > 0 else 0
        avg_improvement_pct = self.stats["total_improvement_pct"] / self.stats["filled"] if self.stats["filled"] > 0 else 0
        return {**self.stats, "fill_rate": fill_rate, "avg_improvement_pct": avg_improvement_pct}

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0):
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Step 0: Downloading LIVE data for {symbol} (Last {days} days)...")
    all_data = []
    now = int(time.time() * 1000)
    total_minutes = int(days * 1440)
    chunk_size = 1440
    num_chunks = int(np.ceil(total_minutes / chunk_size))
    async with aiohttp.ClientSession() as session:
        for i in range(num_chunks):
            end_time = now - i * chunk_size * 60 * 1000
            params = {"symbol": symbol, "interval": "1m", "limit": chunk_size, "endTime": end_time}
            async with session.get(endpoint, params=params) as resp:
                if resp.status != 200: raise Exception(f"Binance API Error: {resp.status}")
                chunk = await resp.json()
                all_data.extend(chunk)
    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    for col in ['close', 'high', 'low']: df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset=['time']).sort_values('time').tail(total_minutes)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    file_path = os.path.join(data_dir, f"{symbol}_live_{days}d.feather")
    df.to_feather(file_path)
    logger.info(f"Success. Saved {days}d live data ({len(df)} candles) to {file_path}")
    return file_path

async def run_backtest(config_path: str, data_dir: str, live_mode: bool = False, ticker_override: Optional[str] = None,
                        days: float = 2.0, commission: float = 0.0004, use_limit_orders: bool = False,
                        limit_offset_pct: float = 0.1, limit_timeout_sec: int = 30, quiet: bool = False) -> Optional[Dict[str, Any]]:
    try:
        if quiet:
            logger.setLevel(logging.WARNING)
        
        with open(config_path, "r", encoding="utf-8") as f: config = json.load(f)
        
        # Check Black List
        base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
        black_list = config.get("black_list", [])
        if base_ticker in black_list:
            if not quiet: logger.info(f"Ticker {base_ticker} is in BLACK LIST. Skipping backtest.")
            return None

        # Настройки лимитов из конфига
        limit_enabled = config.get("limit_order_enabled", False)
        limit_offset = config.get("limit_offset_pct", 0.1)
        
        portfolio_cfg = config["portfolios"][0]
        targets = portfolio_cfg["targets"]
        threshold = portfolio_cfg["rebalance_threshold"]
        
        # Ticker-specific threshold override
        base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
        ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
        threshold = ticker_thresholds.get(base_ticker, threshold)

        if live_mode: file_path = await download_live_data(base_ticker, data_dir, days)
        else:
            # Приоритет свежим лайв-файлам
            possible_files = [
                os.path.join(data_dir, f"{base_ticker}_live_{days}d.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_10d.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_48h.feather"),
                os.path.join(data_dir, f"{base_ticker.replace('USDT', '_USDT')}_USDT-1m-futures.feather")
            ]
            file_path = next((p for p in possible_files if os.path.exists(p)), None)
        
        if not file_path: raise Exception(f"Data for {base_ticker} not found")
        logger.info(f"Starting backtest for {base_ticker} using {file_path}...")
        logger.info(f"Strategy Mode: {'LIMIT' if limit_enabled else 'MARKET'} (Threshold: {threshold*100}%, Offset: {limit_offset}%)")
        
        df = pd.read_feather(file_path).copy().reset_index(drop=True)
        if len(df) > int(days * 1440): df = df.tail(int(days * 1440)).reset_index(drop=True)

        initial_capital = portfolio_cfg.get("initial_capital", 39.0)

        current_equity = initial_capital
        virt_basis_price = df.iloc[0]['close']
        virt_allocated_usdt = initial_capital * targets["VIRTUAL"]["share"]
        siphoning_reserve = 0.0
        initial_tpv = initial_capital
        tpv_ath = initial_capital
        prev_tpv = initial_capital
        
        # Инициализация нулевой свечи
        first_row = df.iloc[0]
        curr_price = first_row['close']

        # Устанавливаем начальные физические позиции для дельта-нейтральности
        long_vol = (initial_capital * targets["BASE_LONG"]["share"] * targets["BASE_LONG"]["leverage"])
        short_vol = (initial_capital * targets["BASE_SHORT"]["share"] * targets["BASE_SHORT"]["leverage"])

        positions = {
            f"{base_ticker}_LONG": long_vol / curr_price,
            f"{base_ticker}_SHORT": short_vol / curr_price
        }
        entry_prices = {"LONG": curr_price, "SHORT": curr_price}

        # Deduct initial setup commission
        current_equity -= (long_vol + short_vol) * commission

        stats = {
            "rebalance_cycles": 0, "total_volume_usdt": 0.0, "max_tpv": initial_capital,
            "max_drawdown_pct": 0.0, "max_drawdown_duration": 0, "current_drawdown_duration": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "daily_returns": [],
            "trailing_stop_triggered": False, "trailing_stop_step": 0
        }

        limit_simulator = LimitOrderSimulator(offset_pct=limit_offset) if limit_enabled else None
        history = []

        # Переход на NumPy для максимальной скорости
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        l_lev = targets["BASE_LONG"]["leverage"] if "BASE_LONG" in targets else 5.0
        s_lev = targets["BASE_SHORT"]["leverage"] if "BASE_SHORT" in targets else 5.0
        l_target = targets["BASE_LONG"]["share"]
        s_target = targets["BASE_SHORT"]["share"]
        v_target = targets["VIRTUAL"]["share"]

        for i in range(len(df)):
            curr_price = close_prices[i]
            
            # Внутренняя логика PortfolioCalculator (упрощенно для скорости)
            price_change_virt = curr_price / virt_basis_price
            virt_current_value = virt_allocated_usdt * price_change_virt
            
            total_tpv = current_equity + (virt_current_value - virt_allocated_usdt) + siphoning_reserve
            tpv = min(total_tpv, initial_capital) if total_tpv > initial_capital else total_tpv
            if tpv <= 0: tpv = 1e-9

            if total_tpv > tpv_ath: tpv_ath = total_tpv
            
            # Analytics Tracking
            tpv_change = total_tpv - prev_tpv
            if tpv_change > 0: stats["gross_profit"] += tpv_change
            else: stats["gross_loss"] += abs(tpv_change)
            prev_tpv = total_tpv

            if total_tpv > stats["max_tpv"]:
                stats["max_tpv"] = total_tpv
                stats["current_drawdown_duration"] = 0
            else: stats["current_drawdown_duration"] += 1
            
            dd = (stats["max_tpv"] - total_tpv) / stats["max_tpv"] if stats["max_tpv"] > 0 else 0
            if dd > stats["max_drawdown_pct"]: stats["max_drawdown_pct"] = dd

            # Trailing Stop
            if portfolio_cfg.get("equity_trailing_stop_pct", 0) > 0:
                if (1 - total_tpv / tpv_ath) * 100 >= portfolio_cfg["equity_trailing_stop_pct"]:
                    stats["trailing_stop_triggered"] = True
                    stats["trailing_stop_step"] = i
                    break

            # Ребалансировка (логика из calculator.py)
            l_qty = abs(positions[f"{base_ticker}_LONG"])
            s_qty = abs(positions[f"{base_ticker}_SHORT"])
            
            le = entry_prices["LONG"] if entry_prices["LONG"] > 0 else curr_price
            se = entry_prices["SHORT"] if entry_prices["SHORT"] > 0 else curr_price
            
            val_long = (l_qty * le / l_lev) + (l_qty * (curr_price - le))
            val_short = (s_qty * se / s_lev) + (s_qty * (se - curr_price))
            
            share_long = val_long / tpv
            share_short = val_short / tpv
            share_virt = virt_current_value / tpv

            # Проверка отклонений
            any_exceeded = False
            if not math.isclose(share_long, l_target, abs_tol=threshold): any_exceeded = True
            elif not math.isclose(share_short, s_target, abs_tol=threshold): any_exceeded = True
            elif not math.isclose(share_virt, v_target, abs_tol=threshold): any_exceeded = True

            if any_exceeded:
                stats["rebalance_cycles"] += 1
                min_notional = config.get("min_notional_usdt", 6.0)
                
                # Ребаланс Виртуальной части
                virt_basis_price = curr_price
                virt_allocated_usdt = tpv * v_target
                
                # Ребаланс Long
                target_val_long = tpv * l_target
                diff_share_l = share_long - l_target
                diff_usdt_l = -diff_share_l * tpv * l_lev
                
                if abs(diff_usdt_l) >= min_notional:
                    side = "BUY" if diff_usdt_l > 0 else "SELL"
                    stats["total_volume_usdt"] += abs(diff_usdt_l)
                    
                    f_price = curr_price
                    if limit_simulator and i < len(df) - 1:
                        _, _, f_price, _ = limit_simulator.simulate_limit_execution(
                            side, abs(diff_usdt_l)/curr_price, curr_price, high_prices[i+1], low_prices[i+1]
                        )
                    
                    change_qty = abs(diff_usdt_l) / f_price
                    current_equity -= (abs(diff_usdt_l) * commission)
                    
                    if side == "BUY":
                        if positions[f"{base_ticker}_LONG"] > 0:
                            entry_prices["LONG"] = (positions[f"{base_ticker}_LONG"] * entry_prices["LONG"] + change_qty * f_price) / (positions[f"{base_ticker}_LONG"] + change_qty)
                        else: entry_prices["LONG"] = f_price
                        positions[f"{base_ticker}_LONG"] += change_qty
                    else:
                        # Siphoning check during reduction
                        trade_pnl = (f_price - entry_prices["LONG"]) * change_qty
                        if trade_pnl > 0 and total_tpv > initial_capital:
                            siphon_amount = min(trade_pnl, total_tpv - initial_capital)
                            siphoning_reserve += siphon_amount
                            current_equity -= siphon_amount
                        positions[f"{base_ticker}_LONG"] = max(0, positions[f"{base_ticker}_LONG"] - change_qty)

                # Ребаланс Short
                target_val_short = tpv * s_target
                diff_share_s = share_short - s_target
                diff_usdt_s = -diff_share_s * tpv * s_lev

                if abs(diff_usdt_s) >= min_notional:
                    side = "SELL" if diff_usdt_s > 0 else "BUY"
                    stats["total_volume_usdt"] += abs(diff_usdt_s)
                    
                    f_price = curr_price
                    if limit_simulator and i < len(df) - 1:
                        _, _, f_price, _ = limit_simulator.simulate_limit_execution(
                            side, abs(diff_usdt_s)/curr_price, curr_price, high_prices[i+1], low_prices[i+1]
                        )
                    
                    change_qty = abs(diff_usdt_s) / f_price
                    current_equity -= (abs(diff_usdt_s) * commission)
                    
                    if side == "SELL":
                        if positions[f"{base_ticker}_SHORT"] > 0:
                            entry_prices["SHORT"] = (positions[f"{base_ticker}_SHORT"] * entry_prices["SHORT"] + change_qty * f_price) / (positions[f"{base_ticker}_SHORT"] + change_qty)
                        else: entry_prices["SHORT"] = f_price
                        positions[f"{base_ticker}_SHORT"] += change_qty
                    else:
                        # Siphoning check during reduction
                        trade_pnl = (entry_prices["SHORT"] - f_price) * change_qty
                        if trade_pnl > 0 and total_tpv > initial_capital:
                            siphon_amount = min(trade_pnl, total_tpv - initial_capital)
                            siphoning_reserve += siphon_amount
                            current_equity -= siphon_amount
                        positions[f"{base_ticker}_SHORT"] = max(0, positions[f"{base_ticker}_SHORT"] - change_qty)

            if i < len(df) - 1:
                p_diff = close_prices[i+1] - curr_price
                current_equity += (positions[f"{base_ticker}_LONG"] * p_diff) + (positions[f"{base_ticker}_SHORT"] * (-p_diff))
            history.append(total_tpv)

        # Final Report
        asset_start, asset_end = close_prices[0], close_prices[-1]
        asset_chg: float = ((asset_end / asset_start) - 1) * 100
        total_final_value: float = total_tpv
        strat_chg: float = ((total_final_value / initial_capital) - 1) * 100

        if not quiet:
            logger.info("\n" + "="*70 + "\n                 LEG-SPECIFIC NEUTRAL ANALYTICS\n" + "="*70)
            logger.info(f"Asset Perf: {base_ticker} {asset_chg:+.2f}% | Strategy: {strat_chg:+.2f}%")
            logger.info(f"Alpha:      {strat_chg - asset_chg:+.2f}% vs HODL")
            logger.info(f"Final Total TPV: {total_final_value:.2f} (SAFE: {siphoning_reserve:.2f})")
            logger.info(f"Max DD:     {stats['max_drawdown_pct']*100:.2f}% | Rebalances: {stats['rebalance_cycles']}")
            if limit_simulator:
                l_stats = limit_simulator.get_summary()
                logger.info(f"Limit Fill: {l_stats['fill_rate']*100:.1f}% | Price Imp: +{l_stats['total_improvement_usdt']:.2f} USDT")
            logger.info("="*70)

        return {
            "profit_pct": strat_chg,
            "max_dd_pct": stats["max_drawdown_pct"] * 100,
            "cycles": stats["rebalance_cycles"],
            "asset_chg_pct": asset_chg,
            "siphoning_reserve": siphoning_reserve
        }

    except Exception as e:
        logger.error(f"Failed: {e}\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_backtest(args.config, r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures", args.live, args.ticker, args.days))
