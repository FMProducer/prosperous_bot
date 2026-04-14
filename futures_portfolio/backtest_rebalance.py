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
    # Long - положительное количество, Short - отрицательное
    long_qty = (tpv * targets["BTCUSDT_LONG"]["share"] * targets["BTCUSDT_LONG"]["leverage"]) / virt_basis_price
    short_qty = -(tpv * targets["BTCUSDT_SHORT"]["share"] * targets["BTCUSDT_SHORT"]["leverage"]) / virt_basis_price
    
    positions = {"BTCUSDT_LONG": long_qty, "BTCUSDT_SHORT": short_qty}
    
    logger.info(f"START Price: {virt_basis_price:.2f} | Initial TPV: {tpv:.2f}")

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
            logger.error(f"LIQUIDATED at step {i}! BTC Price: {curr_price:.2f}")
            break

        # Считаем TPV и отклонения
        calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt)
        deviations = calc.calculate_deviations(targets, threshold)
        
        if deviations:
            for dev in deviations:
                key = dev["symbol"]
                order_qty = dev["diff_usdt"] / curr_price
                
                # КОРРЕКТНАЯ ЛОГИКА ОБНОВЛЕНИЯ:
                if "LONG" in key:
                    positions[key] += order_qty
                else: # SHORT (хранится как отрицательное)
                    positions[key] -= order_qty # Если нужно больше шорта (diff > 0), вычитаем из qty
            
            # Обновляем виртуальный базис
            virt_basis_price = curr_price
            virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]

        if i % 5000 == 0:
            logger.info(f"Step {i:6d}: TPV={calc.tpv:8.2f} | Equity={current_equity:8.2f} | BTC={curr_price:8.2f}")
        
        history.append({"tpv": calc.tpv, "equity": current_equity})

    # 4. Итоги
    final = history[-1]
    logger.info("=" * 50)
    logger.info(f"FULL BACKTEST FINISHED")
    logger.info(f"Start BTC: {df.iloc[0]['close']:.2f} | End BTC: {df.iloc[-1]['close']:.2f} ({((df.iloc[-1]['close']/df.iloc[0]['close'])-1)*100:+.2f}%)")
    logger.info(f"Initial TPV: 10000.00 USDT")
    logger.info(f"Final TPV:   {final['tpv']:.4f} USDT")
    logger.info(f"Profit:      {((final['tpv']/10000)-1)*100:.4f}%")
    logger.info(f"Max Drawdown: { (1 - min(h['tpv'] for h in history)/10000)*100:.4f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"
    asyncio.run(run_backtest("config.json", DATA_DIR))
