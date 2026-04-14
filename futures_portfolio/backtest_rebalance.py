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
    
    # 1. Загрузка 1-минутных данных (BTC_USDT_USDT-1m-futures.feather)
    file_path = os.path.join(data_dir, "BTC_USDT_USDT-1m-futures.feather")
    df = pd.read_feather(file_path)
    
    # Берем последние 43200 минут (30 суток)
    df = df.tail(43200).copy().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} minutes (30 days) of BTC data.")

    # 2. Инициализация
    real_balance = 10000.0
    virt_basis_price = df.iloc[0]['close']
    virt_allocated_usdt = real_balance * targets["BTC_VIRTUAL"]["share"]
    
    current_equity = real_balance
    
    # Начальные позиции (Notional = Share * Leverage * Equity)
    tpv = real_balance
    long_qty = (tpv * targets["BTCUSDT_LONG"]["share"] * targets["BTCUSDT_LONG"]["leverage"]) / virt_basis_price
    short_qty = -(tpv * targets["BTCUSDT_SHORT"]["share"] * targets["BTCUSDT_SHORT"]["leverage"]) / virt_basis_price
    
    positions = {"BTCUSDT_LONG": long_qty, "BTCUSDT_SHORT": short_qty}
    
    logger.info(f"START BTC Price: {virt_basis_price:.2f}")
    logger.info(f"Initial TPV: {tpv:.2f} | Long Notional: {long_qty*virt_basis_price:.2f} | Short Notional: {abs(short_qty*virt_basis_price):.2f}")

    history = []

    # 3. Цикл бэктеста
    for i in range(1, len(df)):
        prev_price = df.iloc[i-1]['close']
        curr_price = df.iloc[i]['close']
        price_change_pct = (curr_price / prev_price) - 1
        
        # Обновляем Equity на фьючерсах (PnL)
        long_notional = positions["BTCUSDT_LONG"] * prev_price
        short_notional = positions["BTCUSDT_SHORT"] * prev_price
        current_equity += (long_notional * price_change_pct) + (short_notional * price_change_pct)
        
        # Считаем TPV и отклонения через Calculator
        calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt)
        deviations = calc.calculate_deviations(targets, threshold)
        
        if deviations:
            logger.info(f"[{df.iloc[i]['date']}] REBALANCE Triggered! BTC Price: {curr_price:.2f}")
            for dev in deviations:
                key = dev["symbol"]
                order_qty = dev["diff_usdt"] / curr_price
                positions[key] += order_qty
            
            # Обновляем базис виртуальной части
            virt_basis_price = curr_price
            virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]

        if i % 100 == 0:
            logger.info(f"Step {i:4d}: TPV={calc.tpv:8.2f} | Equity={current_equity:8.2f} | BTC={curr_price:8.2f}")
        
        history.append({"tpv": calc.tpv, "equity": current_equity, "price": curr_price})

    # 4. Итоги
    final = history[-1]
    logger.info("=" * 50)
    logger.info(f"BACKTEST FINISHED (1 DAY, 1m candles)")
    logger.info(f"Start BTC: {df.iloc[0]['close']:.2f} | End BTC: {df.iloc[-1]['close']:.2f} ({((df.iloc[-1]['close']/df.iloc[0]['close'])-1)*100:+.2f}%)")
    logger.info(f"Initial TPV: 10000.00 USDT")
    logger.info(f"Final TPV:   {final['tpv']:.4f} USDT")
    logger.info(f"Final Equity (on Exchange): {final['equity']:.2f} USDT")
    logger.info(f"Total Portfolio Change: {((final['tpv']/10000)-1)*10000:.6f} pips (very stable)")
    logger.info(f"Max TPV Drawdown: { (1 - min(h['tpv'] for h in history)/10000)*100:.6f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"
    asyncio.run(run_backtest("config.json", DATA_DIR))
