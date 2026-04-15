import pandas as pd
import asyncio
import json
import os
import logging
import traceback
from typing import Dict, List
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

async def run_backtest(config_path: str, data_dir: str):
    try:
        logger.info(f"Starting backtest with config: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        portfolio_cfg = config["portfolios"][0]
        targets = portfolio_cfg["targets"]
        threshold = portfolio_cfg["rebalance_threshold"]
        base_ticker = config.get("base_ticker", "BTCUSDT")
        
        # 1. Загрузка 1-минутных данных
        # Формируем имя файла на основе base_ticker (например, ETHUSDT -> ETH_USDT_USDT-1m-futures.feather)
        ticker_part = base_ticker.replace("USDT", "_USDT")
        file_name = f"{ticker_part}_USDT-1m-futures.feather"
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            # Пытаемся найти любой .feather в папке
            files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
            if files:
                logger.info(f"Available files: {files}")
                file_path = os.path.join(data_dir, files[0])
                logger.info(f"Using alternative file: {file_path}")
            else:
                return

        logger.info(f"Loading data from {file_path}...")
        df = pd.read_feather(file_path)
        df = df.copy().reset_index(drop=True)
        logger.info(f"Loaded {len(df)} minutes of data.")

        # 2. Инициализация
        real_balance = 10000.0
        virt_basis_price = df.iloc[0]['close']
        virt_allocated_usdt = real_balance * targets["VIRTUAL"]["share"]
        current_equity = real_balance
        
        tpv = real_balance
        # Начальные позиции
        positions = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
        
        # Статистика
        stats = {
            "rebalance_cycles": 0,
            "long_buys": 0, "long_sells": 0,
            "short_buys": 0, "short_sells": 0,
            "total_volume_usdt": 0.0,
            "max_tpv": real_balance,
            "min_tpv": real_balance
        }

        history = []

        # 3. Цикл бэктеста
        for i in range(0, len(df)):
            curr_price = df.iloc[i]['close']
            
            # 3.1. Расчёт TPV и отклонений
            calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt, base_ticker=base_ticker)
            
            # Первый шаг или превышение порога
            current_threshold = -1.0 if i == 0 else threshold
            deviations = calc.calculate_deviations(targets, current_threshold)
            
            if deviations:
                stats["rebalance_cycles"] += 1
                deviations.sort(key=lambda x: x["diff_usdt"])

                for dev in deviations:
                    key = dev["symbol"]
                    pos_side = key.split('_')[1]
                    order_qty = dev["diff_usdt"] / curr_price
                    volume = abs(dev["diff_usdt"])
                    stats["total_volume_usdt"] += volume
                    
                    if pos_side == "LONG":
                        side = "BUY" if dev["diff_usdt"] > 0 else "SELL"
                        positions[key] += order_qty
                        if side == "BUY": stats["long_buys"] += 1
                        else: stats["long_sells"] += 1
                    else: # SHORT
                        side = "SELL" if dev["diff_usdt"] > 0 else "BUY"
                        positions[key] += order_qty
                        if side == "SELL": stats["short_sells"] += 1 
                        else: stats["short_buys"] += 1 
                
                # Обновляем базис
                virt_basis_price = curr_price
                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                
                # Пересчитываем для лога
                calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt, base_ticker=base_ticker)

            stats["max_tpv"] = max(stats["max_tpv"], calc.tpv)
            stats["min_tpv"] = min(stats["min_tpv"], calc.tpv)
            
            if i % 10000 == 0:
                logger.info(f"Step {i:6d}: TPV={calc.tpv:8.2f} | Shares (%) L:{calc.share_long_pct} S:{calc.share_short_pct} V:{calc.share_virt_pct} | {base_ticker}={curr_price:8.2f}")
            
            # Начисление PnL для следующего шага
            if i < len(df) - 1:
                next_price = df.iloc[i+1]['close']
                price_diff = next_price - curr_price
                pnl = (positions[f"{base_ticker}_LONG"] * price_diff) + \
                      (positions[f"{base_ticker}_SHORT"] * (-price_diff))
                current_equity += pnl

            history.append({"tpv": calc.tpv, "equity": current_equity})

        # 4. Итоговый отчет
        final_tpv = history[-1]["tpv"]
        final_equity = history[-1]["equity"]
        asset_start = df.iloc[0]['close']
        asset_end = df.iloc[-1]['close']
        asset_change = ((asset_end / asset_start) - 1) * 100
        tpv_change = ((final_tpv / 10000) - 1) * 100

        logger.info("\n" + "="*60)
        logger.info("                 MARKET NEUTRAL BACKTEST REPORT")
        logger.info("="*60)
        logger.info(f"Period:            {len(df)} minutes ({len(df)/1440:.1f} days)")
        logger.info(f"{base_ticker} Price:      {asset_start:.2f} -> {asset_end:.2f} ({asset_change:+.2f}%)")

        logger.info("-"*60)
        logger.info(f"Initial Capital:   10000.00 USDT")
        logger.info(f"Final TPV:         {final_tpv:.2f} USDT")
        logger.info(f"Final Real Equity: {final_equity:.2f} USDT")
        logger.info(f"Total Profit:      {tpv_change:+.4f}%")
        logger.info(f"Max Drawdown:      {(1 - stats['min_tpv']/10000)*100:.4f}%")
        logger.info(f"Max Run-up:       {(stats['max_tpv']/10000 - 1)*100:.4f}%")
        logger.info("-"*60)
        logger.info(f"Rebalance Cycles:  {stats['rebalance_cycles']}")
        logger.info(f"LONG Orders:       {stats['long_buys']} Buy / {stats['long_sells']} Sell")
        logger.info(f"SHORT Orders:      {stats['short_sells']} Sell (Inc) / {stats['short_buys']} Buy (Dec)")
        logger.info(f"Total Turnover:    {stats['total_volume_usdt']:.2f} USDT")
        logger.info(f"Avg Order Size:    {stats['total_volume_usdt']/(sum(v for k,v in stats.items() if 'buys' in k or 'sells' in k) or 1):.2f} USDT")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Исправляем путь к данным
    DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"
    asyncio.run(run_backtest("config.json", DATA_DIR))
