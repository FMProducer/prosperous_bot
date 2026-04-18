import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import os
import logging
import traceback
import argparse
from typing import Dict, List
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

async def download_live_data(symbol: str, data_dir: str):
    """Загружает последние 1500 минут (24ч+) напрямую с Binance Futures"""
    endpoint = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": 1500}
    
    logger.info(f"Step 0: Downloading LIVE data for {symbol} (Last 24h)...")
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Binance API Error: {resp.status}")
            data = await resp.json()
            
    # Формируем DataFrame в стиле Freqtrade/нашего бэктестера
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file_path = os.path.join(data_dir, f"{symbol}_live_24h.feather")
    df.to_feather(file_path)
    logger.info(f"Success. Saved live data to {file_path}")
    return file_path

async def run_backtest(config_path: str, data_dir: str, live_mode: bool = False, ticker_override: str = None):
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        portfolio_cfg = config["portfolios"][0]
        targets = portfolio_cfg["targets"]
        threshold = portfolio_cfg["rebalance_threshold"]
        base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
        
        # 1. Загрузка данных
        if live_mode:
            file_path = await download_live_data(base_ticker, data_dir)
        else:
            ticker_part = base_ticker.replace("USDT", "_USDT")
            file_name = f"{ticker_part}_USDT-1m-futures.feather"
            file_path = os.path.join(data_dir, file_name)
            
            if not os.path.exists(file_path):
                # Пробуем найти любой подходящий файл для этого тикера
                files = [f for f in os.listdir(data_dir) if base_ticker in f and f.endswith('.feather')]
                if files:
                    file_path = os.path.join(data_dir, files[0])
                else:
                    logger.error(f"Data file not found for {base_ticker} in {data_dir}")
                    return

        logger.info(f"Starting backtest for {base_ticker} using {file_path}...")
        df = pd.read_feather(file_path)
        df = df.copy().reset_index(drop=True)
        
        # Если это исторический файл, но мы хотим только последние 24ч
        if not live_mode and len(df) > 1440:
             logger.info(f"Truncating historical data to last 1440 minutes...")
             df = df.tail(1440).reset_index(drop=True)

        logger.info(f"Processing {len(df)} minutes of data.")

        # 2. Инициализация
        initial_capital = 10000.0
        real_balance = initial_capital
        virt_basis_price = df.iloc[0]['close']
        virt_allocated_usdt = real_balance * targets["VIRTUAL"]["share"]
        current_equity = real_balance
        
        siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
        equity_trailing_stop_pct = portfolio_cfg.get("equity_trailing_stop_pct", 0.0)
        siphoning_reserve = 0.0
        initial_tpv = initial_capital
        tpv_ath = initial_capital
        
        positions = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
        
        stats = {
            "rebalance_cycles": 0,
            "total_volume_usdt": 0.0,
            "max_tpv": initial_capital,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration": 0,
            "current_drawdown_duration": 0,
            "daily_returns": [],
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "trailing_stop_triggered": False,
            "trailing_stop_step": 0
        }

        history = []
        prev_tpv = initial_capital
        fee_rate = 0.0004 # 0.04%

        # 3. Цикл бэктеста
        for i in range(0, len(df)):
            curr_price = df.iloc[i]['close']
            
            calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt, 
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)
            
            # Equity Trailing Stop Tracking
            if calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv
            
            if equity_trailing_stop_pct > 0 and tpv_ath > 0 and siphoning_reserve > 0:
                drawdown_from_ath = (1 - calc.total_tpv / tpv_ath) * 100
                if drawdown_from_ath >= equity_trailing_stop_pct:
                    stats["trailing_stop_triggered"] = True
                    stats["trailing_stop_step"] = i
                    logger.warning(f"!!! [STOP] Step {i}: Equity Trailing Stop triggered at {calc.total_tpv:.2f} (ATH: {tpv_ath:.2f}, Drop: {drawdown_from_ath:.2f}%)")
                    break

            # PnL Tracking
            tpv_change = calc.total_tpv - prev_tpv
            if tpv_change > 0: stats["gross_profit"] += tpv_change
            else: stats["gross_loss"] += abs(tpv_change)
            prev_tpv = calc.total_tpv

            # Drawdown & Peak Tracking
            if calc.total_tpv > stats["max_tpv"]:
                stats["max_tpv"] = calc.total_tpv
                stats["max_drawdown_duration"] = max(stats["max_drawdown_duration"], stats["current_drawdown_duration"])
                stats["current_drawdown_duration"] = 0
            else:
                stats["current_drawdown_duration"] += 1
            
            drawdown = (stats["max_tpv"] - calc.total_tpv) / stats["max_tpv"]
            if drawdown > stats["max_drawdown_pct"]:
                stats["max_drawdown_pct"] = drawdown
            
            # Daily Returns for Sharpe
            if i % 1440 == 0 and i > 0:
                day_start_tpv = history[i-1440]["tpv"] if len(history) >= 1440 else initial_capital
                day_return = (calc.total_tpv / day_start_tpv) - 1
                stats["daily_returns"].append(day_return)

            # Profit Siphoning
            if siphoning_threshold_pct > 0:
                if calc.tpv > initial_tpv * (1 + siphoning_threshold_pct / 100):
                    profit = calc.tpv - initial_tpv
                    to_reinvest = profit * reinvestment_ratio
                    to_reserve = profit - to_reinvest
                    siphoning_reserve += to_reserve
                    initial_tpv += to_reinvest
                    calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt, 
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)

            # Rebalancing
            current_threshold = -1.0 if i == 0 else threshold
            
            # Мягкий гистерезис: увеличиваем порог в 2 раза при просадке
            if i > 0 and calc.tpv < initial_tpv:
                current_threshold *= 2.0
                
            deviations = calc.calculate_deviations(targets, current_threshold)
            
            if deviations:
                stats["rebalance_cycles"] += 1
                for dev in deviations:
                    key = dev["symbol"]
                    order_qty = dev["diff_usdt"] / curr_price
                    stats["total_volume_usdt"] += abs(dev["diff_usdt"])
                    positions[key] += order_qty
                
                virt_basis_price = curr_price
                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt, 
                                         base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)

            if i % 1000 == 0 and live_mode:
                res_str = f" SAFE:{siphoning_reserve:7.2f}" if siphoning_reserve > 0 else ""
                logger.info(f"Step {i:6d}: TPV={calc.total_tpv:8.2f}{res_str} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}% | {base_ticker}={curr_price:8.4f}")
            elif i % 10000 == 0:
                res_str = f" SAFE:{siphoning_reserve:7.2f}" if siphoning_reserve > 0 else ""
                logger.info(f"Step {i:6d}: TPV={calc.total_tpv:8.2f}{res_str} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}% | {base_ticker}={curr_price:8.4f}")
            
            if i < len(df) - 1:
                next_price = df.iloc[i+1]['close']
                price_diff = next_price - curr_price
                current_equity += (positions[f"{base_ticker}_LONG"] * price_diff) + (positions[f"{base_ticker}_SHORT"] * (-price_diff))

            history.append({"tpv": calc.total_tpv, "equity": current_equity, "reserve": siphoning_reserve})

        # 4. Final Report
        final_total_tpv = history[-1]["tpv"]
        final_reserve = history[-1]["reserve"]
        final_active_tpv = final_total_tpv - final_reserve
        
        asset_start = df.iloc[0]['close']
        asset_end = df.iloc[-1]['close']
        asset_change_pct = ((asset_end / asset_start) - 1) * 100
        
        total_profit_usdt = final_total_tpv - initial_capital
        total_profit_pct = (total_profit_usdt / initial_capital) * 100
        
        est_commissions = stats["total_volume_usdt"] * fee_rate
        net_profit_after_fees = total_profit_usdt - est_commissions
        
        returns_arr = np.array(stats["daily_returns"])
        sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(365) if len(returns_arr) > 1 and np.std(returns_arr) > 0 else 0
        recovery_factor = total_profit_usdt / (initial_capital * stats["max_drawdown_pct"]) if stats["max_drawdown_pct"] > 0 else 0
        profit_factor = stats["gross_profit"] / stats["gross_loss"] if stats["gross_loss"] > 0 else 0

        logger.info("\n" + "="*70)
        logger.info("                 ADVANCED MARKET NEUTRAL ANALYTICS")
        logger.info("="*70)
        logger.info(f"Period:             {len(df)} min ({len(df)/1440:.1f} days)")
        logger.info(f"Asset Performance:  {base_ticker} {asset_change_pct:+.2f}%")
        logger.info(f"Strategy Alpha:     {total_profit_pct - asset_change_pct:+.2f}% vs HODL")

        logger.info("-" * 70)
        logger.info(f"Initial Capital:    {initial_capital:.2f} USDT")
        logger.info(f"Final Total TPV:    {final_total_tpv:.2f} USDT")
        logger.info(f"  ├── Active TPV:   {final_active_tpv:.2f} USDT")
        logger.info(f"  └── Profit SAFE:  {final_reserve:.2f} USDT")
        
        logger.info("-" * 70)
        logger.info(f"Gross Profit:       {total_profit_usdt:+.2f} USDT")
        logger.info(f"Est. Commissions:   {est_commissions:.2f} USDT (at {fee_rate*100:.3f}%)")
        logger.info(f"NET PROFIT (FEES):  {net_profit_after_fees:+.2f} USDT ({ (net_profit_after_fees/initial_capital)*100:+.2f}%)")

        logger.info("-" * 70)
        logger.info(f"Max Drawdown:       {stats['max_drawdown_pct']*100:.4f}% (Peak-to-Trough)")
        logger.info(f"Max DD Duration:    {stats['max_drawdown_duration']/1440:.2f} days")
        logger.info(f"Max Run-up:         {(stats['max_tpv']/initial_capital - 1)*100:.4f}%")
        logger.info(f"Recovery Factor:    {recovery_factor:.2f}")
        
        if stats["trailing_stop_triggered"]:
            logger.warning(f"Trailing Stop:      TRIGGERED at step {stats['trailing_stop_step']}")
        else:
            logger.info(f"Trailing Stop:      Not triggered (Threshold: {equity_trailing_stop_pct}%)")
        
        logger.info("-" * 70)
        logger.info(f"Sharpe Ratio:       {sharpe:.2f}")
        logger.info(f"Profit Factor:      {profit_factor:.2f}")
        logger.info(f"Rebalance Cycles:   {stats['rebalance_cycles']}")
        logger.info(f"Total Turnover:     {stats['total_volume_usdt']:.2f} USDT")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data_dir", default=r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures")
    parser.add_argument("--live", action="store_true", help="Download last 24h data and test it")
    parser.add_argument("--ticker", default=None, help="Override ticker for backtest (e.g. MOVRUSDT)")
    args = parser.parse_args()
    
    asyncio.run(run_backtest(args.config, args.data_dir, args.live, args.ticker))
