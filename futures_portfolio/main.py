import asyncio
import json
import logging
import os
from typing import Dict, List

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "rebalance.log")
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0}

def save_state(price, allocated):
    with open(STATE_FILE, "w") as f:
        json.dump({"virt_basis_price": price, "virt_allocated_usdt": allocated}, f)

async def rebalance_loop(connector: BinanceConnector, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    threshold = portfolio_cfg["rebalance_threshold"]
    check_interval = portfolio_cfg["check_interval_sec"]
    
    state = load_state()
    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]

    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}

    while True:
        try:
            # 1. Данные
            positions = await connector.get_positions()
            prices = await connector.get_spot_prices(["BTCUSDT"])
            btc_price = prices.get("BTCUSDT")
            real_equity = await connector.get_free_balance()

            # Инициализация первого запуска
            if virt_basis_price == 0:
                virt_basis_price = btc_price
                virt_allocated_usdt = real_equity * targets["BTC_VIRTUAL"]["share"]
                save_state(virt_basis_price, virt_allocated_usdt)

            # 2. Расчёт
            calc = PortfolioCalculator(positions, btc_price, real_equity, virt_basis_price, virt_allocated_usdt)
            deviations = calc.calculate_deviations(targets, threshold)

            if deviations:
                executor = PortfolioExecutor(connector)
                for dev in deviations:
                    key = dev["symbol"]
                    symbol = key.split('_')[0]
                    pos_side = key.split('_')[1]
                    
                    order_qty = dev["diff_usdt"] / btc_price
                    if pos_side == "LONG":
                        side = "BUY" if dev["diff_usdt"] > 0 else "SELL"
                    else: # SHORT
                        side = "SELL" if dev["diff_usdt"] > 0 else "BUY"
                    
                    step_size = step_sizes.get(symbol, 0.0)
                    logger.info(f"Rebalance {key}: share {dev['current_share']:.4f} -> {dev['target_share']:.4f}")
                    await executor.execute_market_order(symbol, abs(order_qty), side, step_size, False, pos_side)
                
                # После ребалансировки обновляем базис виртуальной части
                virt_basis_price = btc_price
                virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]
                save_state(virt_basis_price, virt_allocated_usdt)

            logger.info(f"TPV (Portfolio Value): {calc.tpv:.2f} USDT | Real Equity: {real_equity:.2f} USDT")
        except Exception as e:
            logger.error(f"Error: {e}")

        await asyncio.sleep(check_interval)

if __name__ == "__main__":
    connector = BinanceConnector("KEY", "SECRET", testnet=True)
    asyncio.run(rebalance_loop(connector, "config.json"))
