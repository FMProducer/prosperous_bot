import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
import json
import os
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

async def download_live_data(symbol: str, data_dir: str, days: int = 2):
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Step 0: Downloading LIVE data for {symbol} (Last {days} days)...")
    all_data = []
    now = int(time.time() * 1000)
    total_minutes = days * 1440
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
                        days: int = 2, commission: float = 0.0004, use_limit_orders: bool = False,
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
        if len(df) > days * 1440: df = df.tail(days * 1440).reset_index(drop=True)

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

        # Переход на itertuples для скорости
        for i, row in enumerate(df.itertuples()):
            curr_price = row.close
            calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt,
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                     long_entry_price=entry_prices["LONG"], short_entry_price=entry_prices["SHORT"],
                                     initial_capital=initial_capital)

            if calc.total_tpv > tpv_ath: tpv_ath = calc.total_tpv
            
            # Analytics Tracking
            tpv_change = calc.total_tpv - prev_tpv
            if tpv_change > 0: stats["gross_profit"] += tpv_change
            else: stats["gross_loss"] += abs(tpv_change)
            prev_tpv = calc.total_tpv

            if calc.total_tpv > stats["max_tpv"]:
                stats["max_tpv"] = calc.total_tpv
                stats["current_drawdown_duration"] = 0
            else: stats["current_drawdown_duration"] += 1
            
            dd = (stats["max_tpv"] - calc.total_tpv) / stats["max_tpv"] if stats["max_tpv"] > 0 else 0
            if dd > stats["max_drawdown_pct"]: stats["max_drawdown_pct"] = dd

            # Trailing Stop
            if portfolio_cfg.get("equity_trailing_stop_pct", 0) > 0:
                if (1 - calc.total_tpv / tpv_ath) * 100 >= portfolio_cfg["equity_trailing_stop_pct"]:
                    stats["trailing_stop_triggered"] = True
                    stats["trailing_stop_step"] = i
                    break

            # Siphoning (simplified for backtest: uses trade-based logic estimate)
            # In backtest we can't perfectly replicate per-trade siphoning without detailed fill data,
            # but we use a reasonable approximation.

            # Rebalancing
            actions = calc.calculate_deviations(targets, threshold)
            if actions:
                min_notional = config.get("min_notional_usdt", 6.0)
                valid_actions = [a for a in actions if a["type"] == "VIRTUAL_RESET" or abs(a.get("diff_usdt", 0)) >= min_notional]
                
                if valid_actions:
                    stats["rebalance_cycles"] += 1
                    
                    for action in valid_actions:
                        if action["type"] == "VIRTUAL_RESET":
                            virt_basis_price = curr_price
                            virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                        else:
                            key, diff_usdt = action["symbol"], action["diff_usdt"]
                            pos_side = key.split('_')[1]
                            if pos_side == "LONG": side = "BUY" if diff_usdt > 0 else "SELL"
                            else: side = "SELL" if diff_usdt > 0 else "BUY"
                            
                            stats["total_volume_usdt"] += abs(diff_usdt)

                            if limit_simulator:
                                # Look-ahead bias fix: use NEXT candle high/low for limit execution simulation
                                if i < len(df) - 1:
                                    next_high = df['high'].values[i+1]
                                    next_low = df['low'].values[i+1]
                                    _, _, f_price, exec_type = limit_simulator.simulate_limit_execution(
                                        side, abs(diff_usdt)/curr_price, curr_price,
                                        float(next_high), float(next_low)
                                    )
                                else:
                                    # Last candle: fallback to market at current close
                                    f_price = curr_price
                                    exec_type = "MARKET"
                            else: 
                                f_price = curr_price
                                exec_type = "MARKET"
                            
                            change_qty = abs(diff_usdt) / f_price
                            action_name = "DEFICIT" if diff_usdt > 0 else "EXCESS"

                            current_equity -= (abs(diff_usdt) * commission)

                            if (pos_side == "LONG" and side == "BUY") or (pos_side == "SHORT" and side == "SELL"):
                                if positions[key] > 0:
                                    entry_prices[pos_side] = (positions[key] * entry_prices[pos_side] + change_qty * f_price) / (positions[key] + change_qty)
                                else: entry_prices[pos_side] = f_price
                                positions[key] += change_qty
                            else: 
                                # 💰 Simulation of siphoning during reduction
                                is_reduction = True
                                if is_reduction:
                                    old_entry = entry_prices[pos_side]
                                    if pos_side == "LONG":
                                        trade_pnl = (f_price - old_entry) * change_qty
                                    else:
                                        trade_pnl = (old_entry - f_price) * change_qty
                                    
                                    if trade_pnl > 0 and calc.total_tpv > initial_capital:
                                        siphon_amount: float = min(trade_pnl, calc.total_tpv - initial_capital)
                                        siphoning_reserve += siphon_amount
                                        current_equity -= siphon_amount

                                positions[key] = max(0, positions[key] - change_qty)

                    calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt,
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                             long_entry_price=entry_prices["LONG"], short_entry_price=entry_prices["SHORT"],
                                             initial_capital=initial_capital)

            if i < len(df) - 1:
                next_close = df['close'].values[i+1]
                p_diff = float(next_close) - curr_price
                current_equity += (positions[f"{base_ticker}_LONG"] * p_diff) + (positions[f"{base_ticker}_SHORT"] * (-p_diff))
            history.append(calc.total_tpv)

        # Final Report
        asset_start, asset_end = df.iloc[0]['close'], df.iloc[-1]['close']
        asset_chg: float = ((asset_end / asset_start) - 1) * 100
        total_final_value: float = calc.total_tpv
        strat_chg: float = ((total_final_value / initial_capital) - 1) * 100

        if not quiet:
            logger.info("\n" + "="*70 + "\n                 LEG-SPECIFIC NEUTRAL ANALYTICS\n" + "="*70)
            logger.info(f"Asset Perf: {base_ticker} {asset_chg:+.2f}% | Strategy: {strat_chg:+.2f}%")
            logger.info(f"Alpha:      {strat_chg - asset_chg:+.2f}% vs HODL")
            logger.info(f"Final Total TPV: {total_final_value:.2f} (Active: {calc.tpv:.2f}, SAFE: {siphoning_reserve:.2f})")
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
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_backtest(args.config, r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures", args.live, args.ticker, args.days))
