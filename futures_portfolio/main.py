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
    
    # Получаем базовый тикер из конфигурации
    base_ticker = config.get("base_ticker", "BTCUSDT")
    
    # Состояние синтетической доли
    state = load_json(STATE_FILE, {"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0, "base_ticker": base_ticker})
    # Если тикер сменился, сбрасываем базис виртуальной части
    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state.json changed from {state.get('base_ticker')} to {base_ticker}. Resetting virtual basis.")
        state = {"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0, "base_ticker": base_ticker}
        
    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]

    # Инфо о бирже
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}

    # Для Paper Trading сохраняем цену входа, чтобы считать PNL
    if paper_mode:
        default_paper_state = {
            "balance": 10000.0, 
            "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
            "last_price": 0.0,
            "base_ticker": base_ticker
        }
        paper_state = load_json(PAPER_STATE_FILE, default_paper_state)
        
        # Если тикер сменился, сбрасываем цену, чтобы не считать PNL на старой цене (например с BTC на SOL)
        if paper_state.get("base_ticker") != base_ticker:
            logger.info(f"Ticker changed from {paper_state.get('base_ticker')} to {base_ticker}. Resetting price tracking.")
            paper_state["last_price"] = 0.0
            paper_state["base_ticker"] = base_ticker
            
        # Гарантируем наличие ключей для текущего тикера
        if "positions" not in paper_state: paper_state["positions"] = {}
        if f"{base_ticker}_LONG" not in paper_state["positions"]:
            paper_state["positions"][f"{base_ticker}_LONG"] = 0.0
        if f"{base_ticker}_SHORT" not in paper_state["positions"]:
            paper_state["positions"][f"{base_ticker}_SHORT"] = 0.0
    else:
        paper_state = None

    i = 0
    while True:
        try:
            # 1. Получение цен (всегда живые)
            prices = await connector.get_spot_prices([base_ticker])
            price = prices.get(base_ticker)
            if not price: raise Exception(f"Could not fetch {base_ticker} price")
            
            # 2. Имитация изменения Equity за счет PNL в Paper Mode
            if paper_mode and paper_state["last_price"] > 0:
                price_diff = price - paper_state["last_price"]
                # PNL = Qty * (Price - EntryPrice)
                long_pnl = paper_state["positions"].get(f"{base_ticker}_LONG", 0.0) * price_diff
                short_pnl = paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0) * (-price_diff) # Для шорта инверсия
                paper_state["balance"] += (long_pnl + short_pnl)
            
            if paper_mode:
                paper_state["last_price"] = price
                real_equity = paper_state["balance"]
                positions = paper_state["positions"]
            else:
                real_equity = await connector.get_free_balance()
                positions = await connector.get_positions()

            # Инициализация синтетического базиса
            if virt_basis_price == 0:
                virt_basis_price = price
                virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                save_json(STATE_FILE, {"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt, "base_ticker": base_ticker})

            # 3. Расчёт TPV и отклонений
            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, base_ticker=base_ticker)
            
            # Если это первый запуск (для текущего тикера), позиций нет или они аномально большие - форсируем ребалансировку
            current_pos_sum = abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0))
            is_first_run = current_pos_sum == 0
            is_extreme = calc.share_long_pct > 100 or calc.share_short_pct > 100
            
            if i % 10 == 0: # Лог раз в 5 минут (при интервале 30с) или при каждом ребалансе
                 logger.info(f"Heartbeat: TPV={calc.tpv:.2f} | {base_ticker}={price:.2f} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}%")

            current_threshold = -1.0 if (is_first_run or is_extreme) else threshold
            deviations = calc.calculate_deviations(targets, current_threshold, ignore_limits=(current_threshold < 0))

            if deviations:
                logger.info(f"Rebalance needed. TPV (USDT): {calc.tpv:.2f} | Shares (%) | Long: {calc.share_long_pct} | Short: {calc.share_short_pct} | Virtual: {calc.share_virt_pct}")
                
                # Сортировка: сначала уменьшение позиций (diff_usdt < 0)
                deviations.sort(key=lambda x: x["diff_usdt"])

                for dev in deviations:
                    key = dev["symbol"]
                    pos_side = key.split('_')[1]
                    
                    order_qty = dev["diff_usdt"] / price
                    
                    # Определяем сторону сделки: BUY (увеличить LONG или уменьшить SHORT), SELL (уменьшить LONG или увеличить SHORT)
                    if pos_side == "LONG":
                        side = "BUY" if dev["diff_usdt"] > 0 else "SELL"
                    else: # SHORT
                        side = "SELL" if dev["diff_usdt"] > 0 else "BUY"

                    # Для реального исполнения: если мы уменьшаем позицию, ставим reduce_only
                    reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")

                    step_size = step_sizes.get(key, 0.0)
                    
                    if paper_mode:
                        qty_rounded = PortfolioExecutor(None).round_quantity(abs(order_qty), step_size)
                        if qty_rounded > 0:
                            if pos_side == "LONG":
                                paper_state["positions"][f"{base_ticker}_LONG"] += (qty_rounded if side == "BUY" else -qty_rounded)
                            else:
                                # Для SHORT: SELL увеличивает позицию (шорт), BUY уменьшает
                                paper_state["positions"][f"{base_ticker}_SHORT"] += (qty_rounded if side == "SELL" else -qty_rounded)
                            
                            # Защита от отрицательных позиций
                            paper_state["positions"][f"{base_ticker}_LONG"] = max(0, paper_state["positions"][f"{base_ticker}_LONG"])
                            paper_state["positions"][f"{base_ticker}_SHORT"] = max(0, paper_state["positions"][f"{base_ticker}_SHORT"])
                            
                            logger.info(f"[PAPER] Order Executed: {side} {qty_rounded:.6f} {key}")
                            save_json(PAPER_STATE_FILE, paper_state)
                    else:
                        executor = PortfolioExecutor(connector)
                        await executor.execute_market_order(key, abs(order_qty), side, step_size, reduce_only, pos_side)
                
                # Обновляем базис и пересчитываем доли для финального лога
                virt_basis_price = price
                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                save_json(STATE_FILE, {"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt, "base_ticker": base_ticker})

                final_calc = PortfolioCalculator(
                    paper_state["positions"] if paper_mode else await connector.get_positions(),
                    price, real_equity, virt_basis_price, virt_allocated_usdt,
                    base_ticker=base_ticker
                )
                logger.info(f"Cycle complete. TPV (USDT): {final_calc.tpv:.2f} | Shares (%) | Long: {final_calc.share_long_pct} | Short: {final_calc.share_short_pct} | Virtual: {final_calc.share_virt_pct}")
            
        except Exception as e:
            logger.error(f"Error in cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

        await asyncio.sleep(check_interval)
        i += 1

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
