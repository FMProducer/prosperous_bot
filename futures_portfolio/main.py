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

    # Для Paper Trading сохраняем цену входа, чтобы считать PNL
    if paper_mode:
        if "last_price" not in paper_state:
            paper_state["last_price"] = 0.0

    while True:
        try:
            # 1. Получение цен (всегда живые)
            prices = await connector.get_spot_prices(["BTCUSDT"])
            btc_price = prices.get("BTCUSDT")
            if not btc_price: raise Exception("Could not fetch BTC price")

            # 2. Имитация изменения Equity за счет PNL в Paper Mode
            if paper_mode and paper_state["last_price"] > 0:
                price_diff = btc_price - paper_state["last_price"]
                # PNL = Qty * (Price - EntryPrice)
                long_pnl = paper_state["positions"]["BTCUSDT_LONG"] * price_diff
                short_pnl = paper_state["positions"]["BTCUSDT_SHORT"] * (-price_diff) # Для шорта инверсия
                paper_state["balance"] += (long_pnl + short_pnl)
            
            if paper_mode:
                paper_state["last_price"] = btc_price
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
            # Если это первый запуск и позиций нет - форсируем ребалансировку
            current_threshold = -1.0 if sum(abs(v) for v in positions.values()) == 0 else threshold
            deviations = calc.calculate_deviations(targets, current_threshold)

            if deviations:
                logger.info(f"Rebalance needed. TPV (USDT): {calc.tpv:.2f} | Shares (%) | Long: {calc.share_long_pct} | Short: {calc.share_short_pct} | Virtual: {calc.share_virt_pct}")
                
                # Сортировка: сначала уменьшение позиций (diff_usdt < 0)
                deviations.sort(key=lambda x: x["diff_usdt"])

                for dev in deviations:
                    key = dev["symbol"]
                    symbol = key.split('_')[0]
                    pos_side = key.split('_')[1]
                    
                    order_qty = dev["diff_usdt"] / btc_price
                    reduce_only = dev["diff_usdt"] < 0
                    side = "BUY" if not reduce_only else "SELL"

                    step_size = step_sizes.get(symbol, 0.0)
                    
                    if paper_mode:
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
                        await executor.execute_market_order(symbol, abs(order_qty), side, step_size, reduce_only, pos_side)
                
                # Обновляем базис и пересчитываем доли для финального лога
                virt_basis_price = btc_price
                virt_allocated_usdt = calc.tpv * targets["BTC_VIRTUAL"]["share"]
                save_json(STATE_FILE, {"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})

                final_calc = PortfolioCalculator(
                    paper_state["positions"] if paper_mode else await connector.get_positions(),
                    btc_price, real_equity, virt_basis_price, virt_allocated_usdt
                )
                logger.info(f"Cycle complete. TPV (USDT): {final_calc.tpv:.2f} | Shares (%) | Long: {final_calc.share_long_pct} | Short: {final_calc.share_short_pct} | Virtual: {final_calc.share_virt_pct}")
            
        except Exception as e:
            logger.error(f"Error in cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

        await asyncio.sleep(check_interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Используем ключи из конфига
    connector = BinanceConnector(
        api_key=cfg.get("api_key", ""),
        secret_key=cfg.get("secret_key", ""),
        testnet=cfg.get("testnet", True)
    )
    asyncio.run(rebalance_loop(connector, args.config))
