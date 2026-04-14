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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
async def rebalance_loop(connector: BinanceConnector, config_path: str, check_interval: int):
    """Асинхронный цикл ребалансировки."""
    # Загрузка конфигурации
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    threshold = portfolio_cfg["rebalance_threshold"]
    tickers = portfolio_cfg["tickers"]
    check_interval_sec = portfolio_cfg["check_interval_sec"]
    reduce_only = portfolio_cfg.get("reduce_only", False)

    # Получение информации о бирже для шага лотов
    exchange_info = await connector.get_exchange_info()
    step_sizes = {}
    for sym_info in exchange_info["symbols"]:
        if sym_info["symbol"] in tickers:
            for filter in sym_info["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    step_sizes[sym_info["symbol"]] = float(filter["stepSize"])

    while True:
        try:
            # 1. Получение данных
            positions = await connector.get_positions()
            spot_prices = await connector.get_spot_prices(tickers)
            free_balance = await connector.get_free_balance()

            # 2. Расчёт стоимости и долей
            calculator = PortfolioCalculator(positions, spot_prices, free_balance)
            total_value = calculator.total_portfolio_value()
            current_shares = calculator.current_shares()

            # 3. Проверка отклонений
            deviations = calculator.calculate_deviations(targets, threshold)

            # 4. Выполнение ордеров при необходимости
            if deviations:
                logger.info(f"Found {len(deviations)} deviations exceeding threshold {threshold}")
                executor = PortfolioExecutor(connector)
                for dev in deviations:
                    symbol = dev["symbol"]
                    order_qty = executor.calculate_order_size(
                        target_share=targets[symbol],
                        current_value=dev["current_value"],
                        total_value=total_value,
                        spot_price=spot_prices[symbol],
                    )
                    side = "BUY" if order_qty > 0 else "SELL"
                    step_size = step_sizes.get(symbol, 0.0)
                    
                    logger.info(f"Rebalancing {symbol}: current_share={dev['current_share']:.4f}, target={dev['target_share']:.4f}, order_qty={order_qty:.6f}")
                    
                    res = await executor.execute_market_order(symbol, abs(order_qty), side, step_size, reduce_only)
                    
                    if res["status"] == "SUCCESS":
                        logger.info(f"Order SUCCESS: {symbol} {side} {abs(order_qty)} (result: {res['result'].get('orderId')})")
                    elif res["status"] == "NO_ORDER":
                        logger.info(f"Order SKIPPED: {symbol} - {res['message']}")
                    else:
                        logger.error(f"Order FAILED: {symbol} {side} {abs(order_qty)} - {res.get('message')}")
            else:
                logger.info("Portfolio is balanced. No orders needed.")

            logger.info(
                f"Rebalance cycle: total={total_value:.2f} USDT, "
                f"BTC_share={current_shares.get('BTCUSDT', 0):.4f}, "
                f"ETH_share={current_shares.get('ETHUSDT', 0):.4f}, "
                f"SOL_share={current_shares.get('SOLUSDT', 0):.4f}"
            )
        except Exception as e:
            logger.error(f"Error in rebalance cycle: {e}")

        await asyncio.sleep(check_interval_sec)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Futures Rebalance Bot")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    connector = BinanceConnector(
        api_key="YOUR_API_KEY",
        secret_key="YOUR_SECRET_KEY",
        testnet=args.testnet if hasattr(args, "testnet") else True,
    )

    asyncio.run(
        rebalance_loop(connector, args.config, args.interval)
    )