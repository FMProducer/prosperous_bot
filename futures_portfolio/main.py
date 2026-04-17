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
    siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
    
    # Состояние синтетической доли и сейфа
    state = load_json(STATE_FILE, {
        "virt_basis_price": 0.0, 
        "virt_allocated_usdt": 0.0, 
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "initial_tpv": 0.0,
        "tpv_ath": 0.0
    })
    
    # Если тикер сменился, сбрасываем базис виртуальной части и начальный TPV
    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state.json changed from {state.get('base_ticker')} to {base_ticker}. Resetting basis, initial TPV and ATH.")
        state["virt_basis_price"] = 0.0
        state["virt_allocated_usdt"] = 0.0
        state["initial_tpv"] = 0.0 # Сброс для нового актива
        state["tpv_ath"] = 0.0
        state["base_ticker"] = base_ticker
        save_json(STATE_FILE, state)
        
    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]
    siphoning_reserve = state.get("siphoning_reserve", 0.0)
    initial_tpv = state.get("initial_tpv", 0.0)
    tpv_ath = state.get("tpv_ath", 0.0)

    # Инфо о бирже
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    # Получаем порог трейлинг стопа
    equity_trailing_stop_pct = portfolio_cfg.get("equity_trailing_stop_pct", 0.0)

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
            # 1. Получение цен (всегда живые, фьючерсные)
            prices = await connector.get_futures_prices([base_ticker])
            price = prices.get(base_ticker)
            if not price: raise Exception(f"Could not fetch {base_ticker} price from Futures market")
            
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

            # 2.5 Расчет динамического порога (ATR-based)
            try:
                klines = await connector.get_futures_klines(base_ticker, "1m", limit=15)
                atr = PortfolioCalculator.calculate_atr(klines, period=3)
                if atr > 0:
                    atr_pct = atr / price
                    # Порог = ATR% * 1.5 (настраиваемый коэффициент)
                    dynamic_threshold = round(atr_pct * 1.01, 3)
                    # Ограничиваем: минимум 0.2%, максимум 2.0%
                    threshold = max(0.004, min(0.006, dynamic_threshold))
                    
                    if i % 10 == 0:
                        logger.info(f"Dynamic Threshold: {threshold:.5f} (ATR%: {atr_pct*100:.3f}%)")
                else:
                    threshold = portfolio_cfg["rebalance_threshold"]
            except Exception as atr_err:
                logger.warning(f"Failed to calculate dynamic threshold: {atr_err}. Using config value.")
                threshold = portfolio_cfg["rebalance_threshold"]

            # Инициализация синтетического базиса и начального TPV
            if virt_basis_price == 0 or initial_tpv == 0:
                if virt_basis_price == 0:
                    virt_basis_price = price
                    virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                
                if initial_tpv == 0:
                    # Если сейф только внедрен, берем текущий TPV как базу
                    temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                                 base_ticker=base_ticker, siphoning_reserve=0.0)
                    initial_tpv = temp_calc.tpv
                    logger.info(f"Initialized initial_tpv to {initial_tpv:.2f}")

                state.update({
                    "virt_basis_price": virt_basis_price, 
                    "virt_allocated_usdt": virt_allocated_usdt, 
                    "base_ticker": base_ticker,
                    "siphoning_reserve": siphoning_reserve,
                    "initial_tpv": initial_tpv
                })
                save_json(STATE_FILE, state)

            # 3. Расчёт TPV и отклонений
            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)
            
            # Обновление ATH (All-Time High) для TPV
            if tpv_ath == 0 or calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv
                state["tpv_ath"] = tpv_ath
                save_json(STATE_FILE, state)

            # Проверка Equity Trailing Stop
            if equity_trailing_stop_pct > 0 and tpv_ath > 0 and siphoning_reserve > 0:
                drawdown_pct = (1 - calc.total_tpv / tpv_ath) * 100
                if drawdown_pct >= equity_trailing_stop_pct:
                    logger.warning(f"!!! [STOP] Equity Trailing Stop triggered! TPV: {calc.total_tpv:.2f} | ATH: {tpv_ath:.2f} | Drop: {drawdown_pct:.2f}%")
                    
                    # Закрытие всех позиций
                    logger.info("Closing all positions and stopping the bot...")
                    for pos_key, qty in positions.items():
                        if qty == 0 or base_ticker not in pos_key: continue
                        
                        pos_side = pos_key.split('_')[1] if '_' in pos_key else "BOTH"
                        side = "SELL" if qty > 0 else "BUY"
                        step_size = step_sizes.get(pos_key, 0.0)
                        
                        if paper_mode:
                            paper_state["positions"][pos_key] = 0.0
                            logger.info(f"[PAPER] Position closed: {pos_key}")
                        else:
                            executor = PortfolioExecutor(connector)
                            await executor.execute_market_order(pos_key.split('_')[0], abs(qty), side, step_size, True, pos_side)
                    
                    if paper_mode: save_json(PAPER_STATE_FILE, paper_state)
                    
                    # Сброс состояния для предотвращения немедленного перезапуска при ручном запуске (опционально)
                    # state["tpv_ath"] = 0 
                    # save_json(STATE_FILE, state)
                    
                    logger.info("All positions closed. Bot stopped.")
                    break # Выход из цикла

            # Проверка Механизма "Сейфа"
            if siphoning_threshold_pct > 0:
                # Если активный TPV вырос выше порога от начального
                if calc.tpv > initial_tpv * (1 + siphoning_threshold_pct / 100):
                    profit = calc.tpv - initial_tpv
                    to_reinvest = profit * reinvestment_ratio
                    to_reserve = profit - to_reinvest
                    
                    siphoning_reserve += to_reserve
                    initial_tpv += to_reinvest
                    
                    reinvest_str = f" ({to_reinvest:.2f} reinvested)" if to_reinvest > 0 else ""
                    logger.info(f"!!! [SAFE] Profit of {profit:.2f} USDT processed: {to_reserve:.2f} to reserve{reinvest_str}. Total reserve: {siphoning_reserve:.2f} USDT")
                    
                    # Пересчитываем калькулятор с новым резервом
                    calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)
                    
                    # Сохраняем состояние сейфа
                    state.update({
                        "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv,
                        "virt_basis_price": virt_basis_price,
                        "virt_allocated_usdt": virt_allocated_usdt
                    })
                    save_json(STATE_FILE, state)

            # Если это первый запуск (для текущего тикера), позиций нет или они аномально большие - форсируем ребалансировку
            current_pos_sum = abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0))
            is_first_run = current_pos_sum == 0
            is_extreme = calc.share_long_pct > 100 or calc.share_short_pct > 100
            
            if i % 10 == 0:
                 res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                 logger.info(f"Heartbeat: TPV={calc.total_tpv:.2f}{res_str} | {base_ticker}={price:.2f} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}%")

            current_threshold = -1.0 if (is_first_run or is_extreme) else threshold
            deviations = calc.calculate_deviations(targets, current_threshold, ignore_limits=(current_threshold < 0))

            if deviations:
                logger.info(f"Rebalance needed. TPV_Active: {calc.tpv:.2f} | Reserve: {siphoning_reserve:.2f} | Shares (%) | L:{calc.share_long_pct} S:{calc.share_short_pct} V:{calc.share_virt_pct}")
                
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
                state.update({
                    "virt_basis_price": virt_basis_price, 
                    "virt_allocated_usdt": virt_allocated_usdt, 
                    "base_ticker": base_ticker,
                    "siphoning_reserve": siphoning_reserve,
                    "initial_tpv": initial_tpv
                })
                save_json(STATE_FILE, state)

                final_calc = PortfolioCalculator(
                    paper_state["positions"] if paper_mode else await connector.get_positions(),
                    price, real_equity, virt_basis_price, virt_allocated_usdt,
                    base_ticker=base_ticker, siphoning_reserve=siphoning_reserve
                )
                logger.info(f"Cycle complete. TPV (USDT): {final_calc.total_tpv:.2f} | Active: {final_calc.tpv:.2f} | Shares (%) | L:{final_calc.share_long_pct} S:{final_calc.share_short_pct} V:{final_calc.share_virt_pct}")
            
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
