import pandas as pd
import asyncio
import json
import os
import logging
from typing import Dict, List
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

async def run_backtest(config_path: str, data_dir: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    threshold = portfolio_cfg["rebalance_threshold"]
    
    # 1. Загрузка 1-минутных данных
    file_path = os.path.join(data_dir, "BTC_USDT_USDT-1m-futures.feather")
    df = pd.read_feather(file_path)
    df = df.copy().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} minutes (FULL PERIOD) of BTC data.")

    # 2. Инициализация
    real_balance = 10000.0
    virt_basis_price = df.iloc[0]['close']
    virt_allocated_usdt = real_balance * targets["BTC_VIRTUAL"]["share"]
    current_equity = real_balance
    
    tpv = real_balance
    long_qty = (tpv * targets["BTCUSDT_LONG"]["share"] * targets["BTCUSDT_LONG"]["leverage"]) / virt_basis_price
    short_qty = -(tpv * targets["BTCUSDT_SHORT"]["share"] * targets["BTCUSDT_SHORT"]["leverage"]) / virt_basis_price
    
    positions = {"BTCUSDT_LONG": long_qty, "BTCUSDT_SHORT": short_qty}
    
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
    for i in range(1, len(df)):
        prev_price = df.iloc[i-1]['close']
        curr_price = df.iloc[i]['close']
        price_change_pct = (curr_price / prev_price) - 1
        
        # Обновляем Equity (PnL)
        pnl = (positions["BTCUSDT_LONG"] * prev_price * price_change_pct) + \
              (positions["BTCUSDT_SHORT"] * prev_price * price_change_pct)
        current_equity += pnl
        
        if current_equity <= 0:
            logger.error(f"LIQUIDATED at step {i}!")
            break

        # Считаем TPV и отклонения
        calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt)
        deviations = calc.calculate_deviations(targets, threshold)
        
        if deviations:
            stats["rebalance_cycles"] += 1
            for dev in deviations:
                key = dev["symbol"]
                order_qty = dev["diff_usdt"] / curr_price
                volume = abs(dev["diff_usdt"])
                stats["total_volume_usdt"] += volume
                
                if "LONG" in key:
                    positions[key] += order_qty
                    if order_qty > 0: stats["long_buys"] += 1
                    else: stats["long_sells"] += 1
                else:
                    positions[key] -= order_qty
                    if order_qty > 0: stats["short_sells"] += 1 # Увеличение шорта
                    else: stats["short_buys"] += 1 # Уменьшение шорта
            
            virt_basis_price = curr_price
            virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]

        stats["max_tpv"] = max(stats["max_tpv"], calc.tpv)
        stats["min_tpv"] = min(stats["min_tpv"], calc.tpv)
        
        if i % 10000 == 0:
            logger.info(f"Step {i:6d}: TPV={calc.tpv:8.2f} | BTC={curr_price:8.2f}")
        
        history.append({"tpv": calc.tpv, "equity": current_equity})

    # 4. Итоговый отчет
    final = history[-1]
    btc_start = df.iloc[0]['close']
    btc_end = df.iloc[-1]['close']
    btc_change = ((btc_end / btc_start) - 1) * 100
    tpv_change = ((final['tpv'] / 10000) - 1) * 100

    logger.info("\n" + "="*60)
    logger.info("                 MARKET NEUTRAL BACKTEST REPORT")
    logger.info("="*60)
    logger.info(f"Period:            {len(df)} minutes ({len(df)/1440:.1f} days)")
    logger.info(f"BTC Price:         {btc_start:.2f} -> {btc_end:.2f} ({btc_change:+.2f}%)")
    logger.info("-"*60)
    logger.info(f"Initial Capital:   10000.00 USDT")
    logger.info(f"Final TPV:         {final['tpv']:.2f} USDT")
    logger.info(f"Final Real Equity: {final['equity']:.2f} USDT")
    logger.info(f"Total Profit:      {tpv_change:+.4f}%")
    logger.info(f"Max Drawdown:      {(1 - stats['min_tpv']/10000)*100:.4f}%")
    logger.info(f"Max Run-up:       {(stats['max_tpv']/10000 - 1)*100:.4f}%")
    logger.info("-"*60)
    logger.info(f"Rebalance Cycles:  {stats['rebalance_cycles']}")
    logger.info(f"LONG Orders:       {stats['long_buys']} Buy / {stats['long_sells']} Sell")
    logger.info(f"SHORT Orders:      {stats['short_sells']} Sell (Inc) / {stats['short_buys']} Buy (Dec)")
    logger.info(f"Total Turnover:    {stats['total_volume_usdt']:.2f} USDT")
    logger.info(f"Avg Order Size:    {stats['total_volume_usdt']/(stats['long_buys']+stats['long_sells']+stats['short_buys']+stats['short_sells']):.2f} USDT")
    logger.info("="*60)

if __name__ == "__main__":
    DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"
    asyncio.run(run_backtest("config.json", DATA_DIR))
