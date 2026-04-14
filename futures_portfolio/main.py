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
PAPER_STATE_FILE = os.path.join(os.path.dirname(__file__), "paper_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

async def rebalance_loop(connector: BinanceConnector, config_path: str):
    config = load_json(config_path, {})
    paper_mode = config.get("paper_mode", False)
    
    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    threshold = portfolio_cfg["rebalance_threshold"]
    check_interval = portfolio_cfg["check_interval_sec"]
    
    # Состояние синтетической доли
    state = load_json(STATE_FILE, {"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0})
    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]

    # Состояние для Paper Trading
    if paper_mode:
        logger.info("!!! RUNNING IN PAPER TRADING MODE !!!")
        paper_state = load_json(PAPER_STATE_FILE, {
            "balance": 10000.0, 
            "positions": {"BTCUSDT_LONG": 0.0, "BTCUSDT_SHORT": 0.0}
        })

    # Инфо о бирже
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}

    while True:
        try:
            # 1. Получение цен (всегда живые)
            prices = await connector.get_spot_prices(["BTCUSDT"])
            btc_price = prices.get("BTCUSDT")
            if not btc_price: raise Exception("Could not fetch BTC price")

            # 2. Получение баланса и позиций
            if paper_mode:
                real_equity = paper_state["balance"]
                positions = paper_state["positions"]
            else:
                real_equity = await connector.get_free_balance()
                positions = await connector.get_positions()

            # Инициализация синтетического базиса
            if virt_basis_price == 0:
                virt_basis_price = btc_price
                virt_allocated_usdt = real_equity * targets["BTC_VIRTUAL"]["share"]
                save_json(STATE_FILE, {"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})

            # 3. Расчёт TPV и отклонений
            calc = PortfolioCalculator(positions, btc_price, real_equity, virt_basis_price, virt_allocated_usdt)
            deviations = calc.calculate_deviations(targets, threshold)

            if deviations:
                logger.info(f"Rebalance needed. TPV: {calc.tpv:.2f}")
                for dev in deviations:
                    key = dev["symbol"]
                    symbol = key.split('_')[0]
                    pos_side = key.split('_')[1]
                    
                    order_qty = dev["diff_usdt"] / btc_price
                    side = "BUY" if (pos_side == "LONG" and dev["diff_usdt"] > 0) or (pos_side == "SHORT" and dev["diff_usdt"] < 0) else "SELL"
                    # Корректная логика side для шорта: если diff > 0 (нужно больше шорта) -> SELL
                    if pos_side == "SHORT":
                        side = "SELL" if dev["diff_usdt"] > 0 else "BUY"

                    step_size = step_sizes.get(symbol, 0.0)
                    
                    if paper_mode:
                        # Имитация исполнения
                        qty_rounded = PortfolioExecutor(None).round_quantity(abs(order_qty), step_size)
                        if qty_rounded > 0:
                            if pos_side == "LONG":
                                paper_state["positions"]["BTCUSDT_LONG"] += (qty_rounded if side == "BUY" else -qty_rounded)
                            else:
                                paper_state["positions"]["BTCUSDT_SHORT"] += (qty_rounded if side == "SELL" else -qty_rounded)
                            logger.info(f"[PAPER] Order Executed: {side} {qty_rounded} {key}")
                        save_json(PAPER_STATE_FILE, paper_state)
                    else:
                        executor = PortfolioExecutor(connector)
                        await executor.execute_market_order(symbol, abs(order_qty), side, step_size, False, pos_side)
                
                # Обновляем базис
                virt_basis_price = btc_price
                virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]
                save_json(STATE_FILE, {"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})

            logger.info(f"Cycle complete. TPV: {calc.tpv:.2f} | Equity: {real_equity:.2f} | BTC: {btc_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error in cycle: {e}")

        await asyncio.sleep(check_interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    # В бумажном режиме ключи могут быть любыми, если API позволяет публичные запросы без подписи
    connector = BinanceConnector("PAPER_KEY", "PAPER_SECRET", testnet=True)
    asyncio.run(rebalance_loop(connector, args.config))
