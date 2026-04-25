import asyncio
import json
import logging
import os
import time
import math
import sys
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier

async def load_json(path: str, default: dict) -> dict:
    def _read():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, PermissionError):
                pass
        return default
    return await asyncio.to_thread(_read)

async def save_json(path: str, data: dict, retries: int = 5) -> None:
    def _write():
        for i in range(retries):
            try:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                if os.path.exists(path): os.remove(path)
                os.rename(tmp_path, path)
                return
            except PermissionError:
                if i == retries - 1: break
                time.sleep(0.5)
    await asyncio.to_thread(_write)

async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None):
    config = await load_json(config_path, {})
    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

    live_swarm = config.get("live_swarm", [])
    if live_swarm:
        paper_mode = base_ticker not in live_swarm
        logger.info(f"Swarm Mode: {'LIVE' if not paper_mode else 'PAPER'} (Live list: {live_swarm})")
    else:
        paper_mode = config.get("paper_mode", False)
    
    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    global_threshold = portfolio_cfg["rebalance_threshold"]
    check_interval = portfolio_cfg["check_interval_sec"]
    
    state = await load_json(state_file_path, {
        "virt_basis_price": 0.0, "virt_allocated_usdt": 0.0, "base_ticker": base_ticker,
        "siphoning_reserve": 0.0, "initial_tpv": 0.0, "reference_tpv": 0.0, "tpv_ath": 0.0, "rebalance_cycles": 0
    })

    if state.get("base_ticker") != base_ticker:
        state.update({"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0, "initial_tpv": 0.0, "tpv_ath": 0.0, "base_ticker": base_ticker})
        await save_json(state_file_path, state)

    virt_basis_price, virt_allocated_usdt = state["virt_basis_price"], state["virt_allocated_usdt"]
    siphoning_reserve, initial_tpv, tpv_ath = state.get("siphoning_reserve", 0.0), state.get("initial_tpv", 0.0), state.get("tpv_ath", 0.0)

    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    if paper_mode:
        paper_state = await load_json(paper_state_file_path, {
            "balance": portfolio_cfg.get("max_capital_usdt", 10000.0), "total_profit": 0.0,
            "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
            "last_price": 0.0, "base_ticker": base_ticker, "long_entry_price": 0.0, "short_entry_price": 0.0
        })
    else:
        paper_state = None

    notifier = TelegramNotifier()
    if not paper_mode:
        if not await connector.get_hedge_mode(): return
    
    await notifier.send_message(f"🚀 <b>Bot Started</b>: {base_ticker}\nMode: {'PAPER' if paper_mode else 'REAL'}")

    i = 0
    while True:
        try:
            prices = await connector.get_futures_prices([base_ticker])
            price = prices.get(base_ticker)
            
            if paper_mode:
                if paper_state["last_price"] > 0:
                    pnl = paper_state["positions"].get(f"{base_ticker}_LONG", 0.0) * (price - paper_state["last_price"]) + \
                          paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0) * (paper_state["last_price"] - price)
                    paper_state["balance"] += pnl
                paper_state["last_price"] = price
                real_equity = paper_state["balance"]
                positions = {k: v for k, v in paper_state["positions"].items()}
                l_entry, s_entry = paper_state.get("long_entry_price", price), paper_state.get("short_entry_price", price)
            else:
                margin_info = await connector.get_margin_ratio()
                real_equity = margin_info.get("total_margin_balance", 0.0) - siphoning_reserve
                raw_pos = await connector.get_positions()
                positions = {k: v["qty"] for k, v in raw_pos.items()}
                l_entry, s_entry = raw_pos.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0), raw_pos.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)

            if virt_basis_price == 0 or initial_tpv == 0:
                virt_basis_price, virt_allocated_usdt = price, real_equity * targets["VIRTUAL"]["share"]
                temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, l_entry, s_entry, base_ticker, 0.0, targets, real_equity)
                initial_tpv = temp_calc.tpv
                state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt, "initial_tpv": initial_tpv})
                await save_json(state_file_path, state)

            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, l_entry, s_entry, base_ticker, siphoning_reserve, targets, initial_tpv)
            
            if not math.isclose(calc.siphoning_reserve, siphoning_reserve, abs_tol=1e-6):
                diff = calc.siphoning_reserve - siphoning_reserve
                siphoning_reserve = calc.siphoning_reserve
                state["siphoning_reserve"] = siphoning_reserve
                if paper_mode: paper_state["balance"] -= diff
                await save_json(state_file_path, state)

            if i % 5 == 0:
                 logger.info(f"Heartbeat: TPV={calc.total_tpv:.2f} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}%")
                 state["current_profit"] = calc.total_tpv - initial_tpv
                 await save_json(state_file_path, state)

            actions = calc.calculate_deviations(targets, global_threshold)
            valid_actions = [a for a in actions if a["type"] == "VIRTUAL_RESET" or abs(a.get("diff_usdt", 0)) >= 6.0]
            
            if valid_actions:
                cycles = state.get("rebalance_cycles", 0) + 1
                state["rebalance_cycles"] = cycles
                await notifier.send_message(f"<b>🔄 Rebalance #{cycles}</b>: {base_ticker}")
                
                executor = PortfolioExecutor(connector, base_ticker=base_ticker)
                results = await executor.execute_actions(valid_actions, price, base_ticker, paper_mode, portfolio_cfg, step_sizes)

                if results:
                    for res in results:
                        if res["type"] == "VIRTUAL_RESET":
                            virt_basis_price, virt_allocated_usdt = price, calc.tpv * targets["VIRTUAL"]["share"]
                            state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})
                            continue
                        if paper_mode and res["status"] == "SUCCESS":
                            pos_side, side, qty = res["type"], res["side"], res["qty"]
                            qty_r = executor.round_quantity(qty, step_sizes.get(f"{base_ticker}", 0.0001))
                            paper_state["balance"] -= (qty_r * price * 0.0004) # Только комиссия
                            pos_key = f"{base_ticker}_{pos_side}"
                            old_q, old_e = paper_state["positions"].get(pos_key, 0.0), paper_state.get(f"{pos_side.lower()}_entry_price", price)
                            if side == ("BUY" if pos_side == "LONG" else "SELL"):
                                paper_state[f"{pos_side.lower()}_entry_price"] = (old_q * old_e + qty_r * price) / (old_q + qty_r) if old_q > 0 else price
                                paper_state["positions"][pos_key] += qty_r
                            else:
                                paper_state["positions"][pos_key] = max(0, paper_state["positions"][pos_key] - qty_r)
                                if paper_state["positions"][pos_key] == 0: paper_state[f"{pos_side.lower()}_entry_price"] = 0.0
                    if paper_mode: await save_json(paper_state_file_path, paper_state)
                    await save_json(state_file_path, state)
                    await notifier.send_message(f"<b>✅ Rebalance Complete</b>: {base_ticker}")

            if i % 100 == 0:
                try:
                    m = await connector.get_margin_ratio()
                    await notifier.send_status(base_ticker, calc.total_tpv, calc.total_tpv - initial_tpv, state.get("rebalance_cycles", 0), siphoning_reserve, m.get("total_margin_balance"), await connector.get_bnb_balance())
                except: pass
        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(10)
        await asyncio.sleep(check_interval)
        i += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f: cfg = json.load(f)
    base_ticker = args.ticker if args.ticker else cfg.get("base_ticker", "BTCUSDT")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(os.path.join(log_dir, f"rebalance_{base_ticker}.log"), encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(base_ticker)
    connector = BinanceConnector(api_key=os.environ.get("BINANCE_API_KEY"), secret_key=os.environ.get("BINANCE_SECRET_KEY"), testnet=cfg.get("testnet", True))
    asyncio.run(rebalance_loop(connector, args.config, os.path.abspath(f"state_{base_ticker}.json"), os.path.abspath(f"paper_state_{base_ticker}.json"), logger, ticker_override=base_ticker))
