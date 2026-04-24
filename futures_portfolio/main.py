import asyncio
import json
import logging
import os
import time
import math
import sys
from typing import Dict, List
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier

# Integration test support: Mock Binance if requested
if os.environ.get("MOCK_MODE") == "1":
    class BinanceConnectorMock:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.futures_client = self
        async def get_futures_prices(self, tickers): return {t: 50000.0 for t in tickers}
        async def get_exchange_info(self):
            return {"symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.0001"}]}]}
        async def get_hedge_mode(self): return True
        async def get_positions(self): return {}
        async def get_margin_ratio(self): return {"total_margin_balance": 10000.0, "margin_ratio": 10.0}
        async def get_free_balance(self): return 10000.0
        async def get_bnb_balance(self): return 1.0
        def futures_create_order(self, **kwargs): return {"orderId": 123, "status": "FILLED", "executedQty": kwargs.get("quantity"), "avgPrice": 50000.0}
        async def get_order_book(self, symbol, limit=5): return {"bids": [[50000, 1]], "asks": [[50001, 1]]}
        async def place_limit_maker_order(self, **kwargs): return {"orderId": 124}
        async def get_order_status(self, symbol, order_id): return {"status": "FILLED", "executedQty": 1.0, "avgPrice": 50000.0}
        async def cancel_order(self, symbol, order_id): return {}
    BinanceConnector = BinanceConnectorMock

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
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tmp_path, path)
                return
            except PermissionError:
                if i == retries - 1:
                    break
                time.sleep(0.5)
    await asyncio.to_thread(_write)

async def rebalance_loop(connector: BinanceConnector, executor: PortfolioExecutor, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None):
    config = await load_json(config_path, {})

    # Приоритет тикера: override > config > default
    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

    # Swarm Rotation: определяем режим на основе live_swarm
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
    
    # Ticker-specific threshold override
    ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
    threshold = ticker_thresholds.get(base_ticker, global_threshold)
    
    logger.info(f"Using rebalance threshold: {threshold*100:.2f}% for {base_ticker}")
    
    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
    
    # Состояние синтетической доли и сейфа
    state = await load_json(state_file_path, {
        "virt_basis_price": 0.0,
        "virt_allocated_usdt": 0.0,
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,
        "tpv_ath": 0.0,
        "rebalance_cycles": 0
    })

    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state changed from {state.get('base_ticker')} to {base_ticker}. Resetting.")
        state.update({"virt_basis_price": 0.0, "virt_allocated_usdt": 0.0, "initial_tpv": 0.0, "reference_tpv": 0.0, "tpv_ath": 0.0, "base_ticker": base_ticker})
        await save_json(state_file_path, state)

    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]
    siphoning_reserve = state.get("siphoning_reserve", 0.0)
    initial_tpv = state.get("initial_tpv", 0.0)
    reference_tpv = state.get("reference_tpv", 0.0)
    tpv_ath = state.get("tpv_ath", 0.0)

    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", 0.0)
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    equity_trailing_stop_pct = portfolio_cfg.get("equity_trailing_stop_pct", 0.0)
    margin_warning = portfolio_cfg.get("margin_ratio_warning", 5.0)
    margin_critical = portfolio_cfg.get("margin_ratio_critical", 2.0)

    if paper_mode:
        default_paper_state = {
            "balance": max_capital_usdt if max_capital_usdt > 0 else 10000.0,
            "total_profit": 0.0,
            "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
            "last_price": 0.0,
            "base_ticker": base_ticker,
            f"{base_ticker}_LONG_entry": 0.0,
            f"{base_ticker}_SHORT_entry": 0.0,
            "v_basis": 0.0,
            "v_alloc": 0.0
        }
        paper_state = await load_json(paper_state_file_path, default_paper_state)
        if paper_state.get("base_ticker") != base_ticker:
            paper_state["last_price"] = 0.0
            paper_state["base_ticker"] = base_ticker
        if "positions" not in paper_state: paper_state["positions"] = {}
        for side in ["LONG", "SHORT"]:
            key = f"{base_ticker}_{side}"
            if key not in paper_state["positions"]: paper_state["positions"][key] = 0.0
            entry_key = f"{key}_entry"
            if entry_key not in paper_state: paper_state[entry_key] = 0.0
    else:
        paper_state = None

    notifier = TelegramNotifier()
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    
    if not paper_mode:
        try:
            is_hedge = await connector.get_hedge_mode()
            if not is_hedge:
                logger.error(f"CRITICAL: Hedge Mode is DISABLED for {base_ticker}.")
                return
            logger.info("Hedge Mode verified.")
        except Exception as e:
            logger.error(f"Failed to verify Hedge Mode: {e}")
            return

    await notifier.send_message(f"🚀 <b>Bot Started</b>: <code>{config_base}</code> ({base_ticker})\nMode: {'PAPER' if paper_mode else 'REAL'}")

    i = 0
    while True:
        try:
            prices = await connector.get_futures_prices([base_ticker])
            price = prices.get(base_ticker)
            if not price: raise Exception(f"Could not fetch {base_ticker} price")
            
            if paper_mode and paper_state["last_price"] > 0:
                price_diff = price - paper_state["last_price"]
                pnl = paper_state["positions"].get(f"{base_ticker}_LONG", 0.0) * price_diff + \
                      paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0) * (-price_diff)
                paper_state["balance"] += pnl
                paper_state["total_profit"] += pnl
            
            if paper_mode:
                paper_state["last_price"] = price
                real_equity = paper_state["balance"]
                virt_basis_price = paper_state.get("v_basis", 0.0)
                virt_allocated_usdt = paper_state.get("v_alloc", 0.0)
                positions = {k: v for k, v in paper_state.get("positions", {}).items()}
                l_entry = paper_state.get(f"{base_ticker}_LONG_entry", price)
                s_entry = paper_state.get(f"{base_ticker}_SHORT_entry", price)
            else:
                margin_info = await connector.get_margin_ratio()
                real_equity = margin_info.get("total_margin_balance", 0.0) - siphoning_reserve
                virt_basis_price = state.get("virt_basis_price", 0.0)
                virt_allocated_usdt = state.get("virt_allocated_usdt", 0.0)
                raw_positions = await connector.get_positions()
                positions = {k: v["qty"] for k, v in raw_positions.items()}
                l_entry = raw_positions.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                s_entry = raw_positions.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
            
            if max_capital_usdt > 0: real_equity = min(real_equity, max_capital_usdt)

            if virt_basis_price == 0 or initial_tpv == 0:
                if virt_basis_price == 0:
                    virt_basis_price = price
                    virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                if initial_tpv == 0:
                    temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                 base_ticker=base_ticker, siphoning_reserve=0.0, targets=targets,
                                                 long_entry_price=l_entry, short_entry_price=s_entry, initial_capital=real_equity)
                    initial_tpv = temp_calc.tpv
                    reference_tpv = initial_tpv
                    logger.info(f"Initialized TPV base: {initial_tpv:.2f}")

                state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt, "initial_tpv": initial_tpv, "reference_tpv": reference_tpv})
                if paper_mode: paper_state.update({"v_basis": virt_basis_price, "v_alloc": virt_allocated_usdt})
                await save_json(state_file_path, state)
                if paper_mode: await save_json(paper_state_file_path, paper_state)

            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                     long_entry_price=l_entry, short_entry_price=s_entry, initial_capital=initial_tpv)
            
            if not math.isclose(calc.siphoning_reserve, siphoning_reserve, abs_tol=1e-6):
                siphon_diff = calc.siphoning_reserve - siphoning_reserve
                siphoning_reserve = calc.siphoning_reserve
                state["siphoning_reserve"] = siphoning_reserve
                await save_json(state_file_path, state)
                if paper_mode:
                    paper_state["balance"] -= siphon_diff
                    await save_json(paper_state_file_path, paper_state)
                logger.info(f"💰 [SAFE] Change: {siphon_diff:+.4f} (Total: {siphoning_reserve:.2f})")

            if tpv_ath == 0 or calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv
                state["tpv_ath"] = tpv_ath
                await save_json(state_file_path, state)

            if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                if (1 - calc.total_tpv / tpv_ath) * 100 >= equity_trailing_stop_pct:
                    msg = f"Trailing Stop triggered for {base_ticker}."
                    logger.warning(msg)
                    await notifier.send_alert("STOP LOSS", msg)
                    break

            if i % 5 == 0:
                 res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                 logger.info(f"Heartbeat: TPV={calc.total_tpv:.2f}{res_str} | {base_ticker}={price:.6g} | L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%")
                 state["current_profit"] = calc.total_tpv - initial_tpv
                 await save_json(state_file_path, state)

            actions = calc.calculate_deviations(targets, threshold if calc.tpv >= reference_tpv else threshold * 2.0)
            valid_actions = [a for a in actions if a["type"] == "VIRTUAL_RESET" or abs(a.get("diff_usdt", 0)) >= portfolio_cfg.get("min_notional_usdt", 6.0)]
            
            if valid_actions:
                logger.info(f"Rebalance needed ({len(valid_actions)} actions).")
                cycles = state.get("rebalance_cycles", 0) + 1
                state["rebalance_cycles"] = cycles
                await save_json(state_file_path, state)
                await notifier.send_message(f"<b>🔄 Rebalance #{cycles}</b>: {base_ticker}")

                excess_to_siphon = max(0, calc.total_tpv - initial_tpv)
                execution_results = await executor.execute_actions(valid_actions, price, base_ticker, paper_mode=paper_mode, portfolio_cfg=portfolio_cfg, step_sizes=step_sizes)

                if execution_results:
                    for res in execution_results:
                        if res["type"] == "VIRTUAL_RESET":
                            virt_basis_price = price
                            virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                            if paper_mode: paper_state.update({"v_basis": virt_basis_price, "v_alloc": virt_allocated_usdt})
                            continue

                        pos_side, symbol, side = res["type"], res["symbol"], res["side"]
                        qty_rounded = executor.round_quantity(res["qty"], step_sizes.get(symbol.split('_')[0], 0.0))

                        if paper_mode and res["status"] == "SUCCESS":
                            commission = (qty_rounded * price) * 0.0004
                            paper_state["balance"] -= commission
                            pos_key = f"{base_ticker}_{pos_side}"
                            entry_key = f"{pos_key}_entry"
                            old_qty, old_entry = paper_state["positions"].get(pos_key, 0.0), paper_state.get(entry_key, price)

                            if side == ("BUY" if pos_side == "LONG" else "SELL"):
                                paper_state[entry_key] = (old_qty * old_entry + qty_rounded * price) / (old_qty + qty_rounded) if old_qty > 0 else price
                                paper_state["positions"][pos_key] += qty_rounded
                            else:
                                trade_pnl = ((price - old_entry) if pos_side == "LONG" else (old_entry - price)) * qty_rounded - commission
                                if trade_pnl > 0 and excess_to_siphon > 0:
                                    s_amt = min(trade_pnl, excess_to_siphon)
                                    siphoning_reserve += s_amt
                                    excess_to_siphon -= s_amt
                                    state["siphoning_reserve"] = siphoning_reserve
                                paper_state["positions"][pos_key] = max(0, paper_state["positions"][pos_key] - qty_rounded)
                                if paper_state["positions"][pos_key] == 0: paper_state[entry_key] = 0.0
                        
                        if res["status"] == "SUCCESS":
                            logger.info(f"Order Executed: {side} {symbol} @ {price:.6f}")

                    if paper_mode: await save_json(paper_state_file_path, paper_state)
                    await save_json(state_file_path, state)

                    # Update Calculator after trades for accurate summary
                    if paper_mode:
                        real_equity = paper_state["balance"]
                        positions = {k: v for k, v in paper_state["positions"].items()}
                        l_entry, s_entry = paper_state[f"{base_ticker}_LONG_entry"], paper_state[f"{base_ticker}_SHORT_entry"]
                    else:
                        raw_pos = await connector.get_positions()
                        positions = {k: v["qty"] for k, v in raw_pos.items()}
                        l_entry, s_entry = raw_pos.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0), raw_pos.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
                        real_equity = (await connector.get_margin_ratio()).get("total_margin_balance", 0.0) - siphoning_reserve

                    calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                                long_entry_price=l_entry, short_entry_price=s_entry, initial_capital=initial_tpv)

                    await notifier.send_message(f"<b>✅ Rebalance Complete</b>: L:{calc.share_long_pct}% S:{calc.share_short_pct}% TPV:{calc.total_tpv:.1f}")

            if i % 100 == 0:
                try:
                    m_info = await connector.get_margin_ratio()
                    await notifier.send_status(base_ticker, calc.total_tpv, calc.total_tpv - initial_tpv, state.get("rebalance_cycles", 0), siphoning_reserve, m_info.get("total_margin_balance"), await connector.get_bnb_balance())
                except: pass

        except Exception as e:
            logger.error(f"Error in cycle: {e}", exc_info=True)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(os.path.join(log_dir, f"rebalance_{base_ticker}.log"), encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(base_ticker)
    
    api_key, secret_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", "")), os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))
    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))
    asyncio.run(rebalance_loop(connector, PortfolioExecutor(connector, base_ticker=base_ticker), args.config, os.path.abspath(f"state_{base_ticker}.json"), os.path.abspath(f"paper_state_{base_ticker}.json"), logger, ticker_override=base_ticker))
