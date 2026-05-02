import asyncio
import json
import logging
import os
import time
import random
from typing import Dict, List, Any
from decimal import Decimal
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier
from storage import safe_load_json as load_json, safe_save_json as save_json

from pathlib import Path

# Global ProcessPoolExecutor for heavy math
process_executor = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))

def calculate_portfolio_task(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                             base_ticker, siphoning_reserve, targets, initial_capital, 
                             threshold, ignore_limits, long_entry_price=0.0, short_entry_price=0.0):
    """Heavy math task to be run in a separate process."""
    calc = PortfolioCalculator(
        positions=positions,
        spot_price=price,
        real_equity=real_equity,
        virt_basis_price=virt_basis_price,
        virt_allocated_usdt=virt_allocated_usdt,
        base_ticker=base_ticker,
        siphoning_reserve=siphoning_reserve,
        targets=targets,
        initial_capital=initial_capital,
        long_entry_price=long_entry_price,
        short_entry_price=short_entry_price
    )
    actions = calc.calculate_deviations(targets, threshold, ignore_limits=ignore_limits)
    
    # Extract serializable data for return
    return {
        "actions": actions,
        "share_long_pct": float(calc.share_long_pct),
        "share_short_pct": float(calc.share_short_pct),
        "share_virt_pct": float(calc.share_virt_pct),
        "share_cash_pct": float(calc.share_cash_pct),
        "tpv": float(calc.tpv),
        "total_tpv": float(calc.total_tpv),
        "siphoning_reserve": float(calc.siphoning_reserve),
        "virt_current_value": float(calc.virt_current_value)
    }

def sync_read_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def emit_signal(signal_type: str, ticker: str) -> None:
    """Создает пустой файл-флаг для супервайзера."""
    sig_path = Path("signals") / f"{signal_type}_{ticker}.flag"
    try:
        sig_path.touch(exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to emit signal {signal_type} for {ticker}: {e}")


async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None, paper_mode_override: bool = None):
    # Обычное чтение конфига без блокировок
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Aborting cycle: Failed to read config {config_path}: {e}")
        return

    if not config or "portfolios" not in config:
        logger.error(f"Aborting cycle: Invalid or missing config structure from {config_path}")
        return

    # Paper mode: override > config
    paper_mode = paper_mode_override if paper_mode_override is not None else config.get("paper_mode", False)
    
    # Приоритет тикера: override > config > default
    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

    # Initial parameter load
    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    global_threshold = portfolio_cfg["rebalance_threshold"]
    check_interval = portfolio_cfg.get("check_interval_sec", 15)
    ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
    threshold = ticker_thresholds.get(base_ticker, global_threshold)
    siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", portfolio_cfg.get("initial_capital", 0.0))

    # Состояние синтетической доли и сейфа
    state = await load_json(state_file_path, {
        "virt_basis_price": 0.0,
        "virt_allocated_usdt": 0.0,
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,  # Фиксированная база для гистерезиса
        "tpv_ath": 0.0,
        "rebalance_cycles": 0,
        "started_at": time.time()
    })

    # Если тикер сменился, сбрасываем базис виртуальной части и начальный TPV
    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state changed from {state.get('base_ticker')} to {base_ticker}. Resetting basis, initial TPV and ATH.")
        state["virt_basis_price"] = 0.0
        state["virt_allocated_usdt"] = 0.0
        state["initial_tpv"] = 0.0 
        state["reference_tpv"] = 0.0
        state["tpv_ath"] = 0.0
        state["base_ticker"] = base_ticker
        state["started_at"] = time.time()
        await save_json(state_file_path, state)

    virt_basis_price = float(state["virt_basis_price"])
    virt_allocated_usdt = float(state["virt_allocated_usdt"])
    siphoning_reserve = float(state.get("siphoning_reserve", 0.0))
    initial_tpv = float(state.get("initial_tpv", 0.0))
    reference_tpv = float(state.get("reference_tpv", 0.0))
    tpv_ath = float(state.get("tpv_ath", 0.0))
    cycles = state.get("rebalance_cycles", 0)

    # Инфо о бирже
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    equity_trailing_stop_pct = portfolio_cfg.get("equity_trailing_stop_pct", 0.0)
    margin_warning = portfolio_cfg.get("margin_ratio_warning", 5.0)
    margin_critical = portfolio_cfg.get("margin_ratio_critical", 2.0)

    if paper_mode:
        default_paper_state = {
            "balance": max_capital_usdt if max_capital_usdt > 0 else 10000.0,
            "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
            "last_price": 0.0,
            "base_ticker": base_ticker,
            "long_entry_price": 0.0,
            "short_entry_price": 0.0
        }
        paper_state = await load_json(paper_state_file_path, default_paper_state)

        if paper_state.get("base_ticker") != base_ticker:
            logger.info(f"Ticker in paper state changed from {paper_state.get('base_ticker')} to {base_ticker}. Resetting.")
            paper_state["last_price"] = 0.0
            paper_state["base_ticker"] = base_ticker

        if "positions" not in paper_state: paper_state["positions"] = {}
        if f"{base_ticker}_LONG" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_LONG"] = 0.0
        if f"{base_ticker}_SHORT" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_SHORT"] = 0.0
        if "long_entry_price" not in paper_state: paper_state["long_entry_price"] = 0.0
        if "short_entry_price" not in paper_state: paper_state["short_entry_price"] = 0.0
    else:
        paper_state = None

    # Инициализация уведомлений
    notifier = TelegramNotifier()
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    
    # Инициализация экзекутора
    max_ops = config.get("max_orders_per_second", 10)
    executor = PortfolioExecutor(connector, base_ticker=base_ticker, max_orders_per_second=max_ops)

    # Hedge Mode Guard
    if not paper_mode:
        try:
            is_hedge = await connector.get_hedge_mode()
            if not is_hedge:
                msg = f"CRITICAL: Hedge Mode is DISABLED on Binance for {base_ticker}. Please enable it to start the bot."
                logger.error(msg)
                asyncio.create_task(notifier.send_alert("STARTUP ERROR", msg))
                return
            logger.info("Hedge Mode verified.")
        except Exception as e:
            logger.error(f"Failed to verify Hedge Mode: {e}")
            asyncio.create_task(notifier.send_alert("STARTUP ERROR", f"Could not verify Hedge Mode: {e}"))
            return

    asyncio.create_task(notifier.send_message(f"🚀 <b>Bot Started</b>: <code>{config_base}</code> ({base_ticker})\nMode: {'PAPER' if paper_mode else 'REAL'}"))

    i = 0
    status_offset = random.randint(0, 99)
    try:
        while True:
            try:
                # Dynamic config reload
                try:
                    # Unblock the Event Loop
                    current_config = await asyncio.to_thread(sync_read_json, config_path)
                    portfolio_cfg = current_config["portfolios"][0]
                    targets = portfolio_cfg["targets"]
                    global_threshold = portfolio_cfg["rebalance_threshold"]
                    check_interval = portfolio_cfg.get("check_interval_sec", 15)
                    ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
                    threshold = ticker_thresholds.get(base_ticker, global_threshold)
                    siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
                    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
                    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", portfolio_cfg.get("initial_capital", 0.0))
                except Exception as e:
                    logger.error(f"Error reloading config: {e}. Using previous values.")

                # Use Mark Price for TPV and rebalance triggers as recommended by Audit
                prices = await connector.get_mark_prices([base_ticker])
                price = prices.get(base_ticker)
                if not price: raise Exception(f"Could not fetch {base_ticker} mark price")
                
                # Fetch current data for deviation calculation
                l_entry: float = 0.0
                s_entry: float = 0.0
                real_equity: float = 0.0
                positions: Dict[str, float] = {}

                if paper_mode:
                    paper_state["last_price"] = price
                    l_qty = abs(paper_state["positions"].get(f"{base_ticker}_LONG", 0.0))
                    s_qty = abs(paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0))
                    u_pnl = l_qty * (price - paper_state.get("long_entry_price", price)) + \
                            s_qty * (paper_state.get("short_entry_price", price) - price)

                    real_equity = paper_state["balance"] + u_pnl
                    positions = paper_state["positions"]
                    l_entry = paper_state.get("long_entry_price", price)
                    s_entry = paper_state.get("short_entry_price", price)
                    m_info = {}
                else:
                    m_info = await connector.get_margin_ratio()
                    # CRITICAL: Subtract reserve from margin balance before deviation calculation
                    real_equity = m_info.get("total_margin_balance", 0.0) - siphoning_reserve
                    raw_positions = await connector.get_positions()
                    positions = {k: v["qty"] for k, v in raw_positions.items()}
                    l_entry = raw_positions.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                    s_entry = raw_positions.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)

                if virt_basis_price == 0 or initial_tpv == 0:
                    if virt_basis_price == 0:
                        virt_basis_price = price
                        virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                    if initial_tpv == 0:
                        config_initial_cap = portfolio_cfg.get("initial_capital", real_equity)
                        
                        # Use executor for initial TPV calculation
                        loop = asyncio.get_running_loop()
                        initial_calc_res = await loop.run_in_executor(
                            process_executor, calculate_portfolio_task,
                            positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                            base_ticker, 0.0, targets, config_initial_cap, -1.0, True,
                            l_entry, s_entry
                        )
                        initial_tpv = initial_calc_res["tpv"]
                        reference_tpv = initial_tpv
                        logger.info(f"Initialized TPV base: {initial_tpv:.2f} (from {'config' if 'initial_capital' in portfolio_cfg else 'current equity'})")

                    state.update({
                        "virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt,
                        "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                    })

                # Offload heavy math to ProcessPoolExecutor
                loop = asyncio.get_running_loop()
                current_threshold = -1.0 if (abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0)) == 0) else threshold
                
                calc_res = await loop.run_in_executor(
                    process_executor, calculate_portfolio_task,
                    positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                    base_ticker, siphoning_reserve, targets, initial_tpv, 
                    current_threshold, (current_threshold < 0),
                    l_entry, s_entry
                )
                
                tpv_total = calc_res["total_tpv"]
                tpv_active = calc_res["tpv"]
                actions = calc_res["actions"]

                # EMERGENCY STOP: If TPV drops below 50% of initial capital
                if tpv_total < initial_tpv * 0.5:
                    msg = f"CRITICAL: TPV {tpv_total:.2f} is less than 50% of initial {initial_tpv:.2f}. EMERGENCY STOP!"
                    logger.critical(msg)
                    asyncio.create_task(notifier.send_alert("EMERGENCY STOP", msg))
                    emit_signal("stop", base_ticker)
                    await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker)
                    return

                if tpv_ath == 0 or tpv_total > tpv_ath:
                    tpv_ath = tpv_total
                    state["tpv_ath"] = tpv_ath

                if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                    drawdown_pct = (1 - tpv_total / tpv_ath) * 100
                    if drawdown_pct >= equity_trailing_stop_pct:
                        msg = f"Trailing Stop triggered: {drawdown_pct:.2f}% drop from ATH. Closing all positions for {base_ticker}."
                        logger.warning(f"!!! [STOP] {msg}")
                        asyncio.create_task(notifier.send_alert("STOP LOSS", msg))
                        for pos_key, qty in positions.items():
                            if qty == 0 or base_ticker not in pos_key: continue
                            side = "SELL" if qty > 0 else "BUY"
                            step_size = step_sizes.get(base_ticker, 0.0)
                            if paper_mode:
                                paper_state["positions"][pos_key] = 0.0
                            else:
                                await PortfolioExecutor(connector).execute_market_order(pos_key.split('_')[0], abs(qty), side, step_size, True, pos_key.split('_')[1] if '_' in pos_key else "BOTH")
                        if paper_mode: await save_json(paper_state_file_path, paper_state)
                        
                        # Эмитируем сигнал остановки для супервайзера
                        emit_signal("stop", base_ticker)
                        
                        # Сбрасываем ATH и начальные значения, чтобы при перезапуске бот не попал в цикл стоп-лоссов
                        state["tpv_ath"] = 0.0
                        state["initial_tpv"] = 0.0
                        state["reference_tpv"] = 0.0
                        state["virt_basis_price"] = 0.0
                        state["trailing_stop_triggered"] = True
                        await save_json(state_file_path, state)
                        
                        logger.info("Positions closed and state reset. Bot stopped.")
                        break

                if not paper_mode and (margin_warning > 0 or margin_critical > 0):
                    try:
                        m_ratio = m_info.get("margin_ratio", 0.0)
                        if m_ratio > 0:
                            if m_ratio < portfolio_cfg.get("margin_ratio_critical", 2.0):
                                msg = f"Margin ratio {m_ratio:.2f} < {portfolio_cfg.get('margin_ratio_critical', 2.0)}. Emergency stop!"
                                logger.error(msg)
                                asyncio.create_task(notifier.send_alert("CRITICAL MARGIN", msg))
                                emit_signal("stop", base_ticker)
                                await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker)
                                return
                            elif m_ratio < portfolio_cfg.get("margin_ratio_warning", 5.0):
                                msg = f"Low margin ratio: {m_ratio:.2f}"
                                logger.warning(msg)
                                asyncio.create_task(notifier.send_message(f"⚠️ <b>WARNING</b>: {msg} ({base_ticker})"))
                    except Exception as e:
                        logger.error(f"Failed to check margin: {e}")

                if i % 5 == 0:
                    res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                    logger.info(f"Heartbeat: Balance={real_equity:.2f}{res_str} | {base_ticker}={price:.6g} | L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}% C:{calc_res['share_cash_pct']:.1f}%")
                
                # Capture virtual parameters before potential reset to ensure virtual profit is siphoned
                old_virt_basis: float = virt_basis_price
                old_virt_alloc: float = virt_allocated_usdt

                # Логика ребалансировки
                valid_actions = []
                if actions:
                    # Внедряем Notional Value Guard для ВСЕХ ордеров (включая VIRTUAL_RESET)
                    min_notional = portfolio_cfg.get("min_notional_usdt", 6.0)
                    valid_actions = [a for a in actions if abs(a.get("diff_usdt", 0)) >= min_notional]
                    
                    if valid_actions:
                        logger.info(f"Rebalance needed ({len(valid_actions)} actions). Shares: L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}%")
                        
                        rebalance_msg = (
                            f"🔄 <b>Rebalance Starting</b>: <code>{base_ticker}</code>\n"
                            f"Current: L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}%\n"
                            f"Target Actions: {len(valid_actions)}"
                        )
                        asyncio.create_task(notifier.send_message(rebalance_msg))

                        # 1. Execute actions concurrently
                        exec_results = await executor.execute_actions(
                            valid_actions, price, paper_mode, portfolio_cfg, step_sizes, paper_state
                        )

                        # 2. Update core states (Virtual Basis, Paper Positions/Balance)
                        for res in exec_results:
                            status = res.get("status")
                            pos_side = res.get("type", "UNKNOWN")

                            if status == "ERROR":
                                logger.error(f"Action failed: {res.get('message')}")
                                continue

                            if status in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]:
                                if res.get("type") == "VIRTUAL_RESET":
                                    virt_basis_price = price
                                    # Target share reset based on active TPV
                                    virt_allocated_usdt = float(Decimal(str(tpv_active)) * Decimal(str(targets["VIRTUAL"]["share"])))
                                    state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})
                                    logger.info(f"🔄 Virtual share rebalanced (Reset to {targets['VIRTUAL']['share']*100:.1f}%)")
                                    continue

                                # Update execution results info
                                key = res.get("symbol", "UNKNOWN")
                                side = res.get("side", "UNKNOWN")
                                qty = res.get("qty", 0.0)
                                trade_pnl = res.get("trade_pnl", 0.0)
                                reduce_only = res.get("reduce_only", False)

                                if paper_mode:
                                    pos_key = f"{base_ticker}_{pos_side}"
                                    old_qty = paper_state["positions"].get(pos_key, 0.0)
                                    entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                                    # CRITICAL FIX: If entry price is 0.0, use current price to prevent fake collapse
                                    old_entry = paper_state.get(entry_key, price)
                                    if old_entry <= 0: old_entry = price

                                    if trade_pnl == 0.0 and reduce_only:
                                        if pos_side == "LONG":
                                            trade_pnl = qty * (price - old_entry)
                                        else:
                                            trade_pnl = qty * (old_entry - price)

                                    new_qty = (old_qty + qty) if (side == "BUY" and pos_side == "LONG") or (side == "SELL" and pos_side == "SHORT") else (old_qty - qty)
                                    
                                    if not reduce_only:
                                        curr_q = abs(old_qty)
                                        paper_state[entry_key] = (curr_q * old_entry + qty * price) / (curr_q + qty) if (curr_q + qty) > 0 else price
                                    
                                    paper_state["positions"][pos_key] = new_qty
                                    paper_state["balance"] += trade_pnl
                                    paper_state["balance"] -= res.get("commission", 0.0)
                                    
                                    trade_log = f"📝 PAPER: {side} {qty} {pos_key} @ {price:.6g}"
                                    if trade_pnl != 0: trade_log += f" | PnL: {trade_pnl:+.4f}"
                                    logger.info(trade_log)
                                    asyncio.create_task(notifier.send_message(f"<b>{trade_log}</b>"))
                                else:
                                    logger.info(f"✅ REAL: {side} {key} order filled. Status: {status}")
                            else:
                                logger.warning(f"❌ {side} {key} execution status: {status}. Message: {res.get('message')}")

                        cycles += 1
                        state["rebalance_cycles"] = cycles

                        if paper_mode:
                            await save_json(paper_state_file_path, paper_state)

                # 3. GLOBAL SAFE SIPHONING (Runs every cycle)
                if actions and len(valid_actions) > 0:
                    if paper_mode:
                        l_qty_p = abs(paper_state["positions"].get(f"{base_ticker}_LONG", 0.0))
                        s_qty_p = abs(paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0))
                        u_pnl = l_qty_p * (price - paper_state.get("long_entry_price", price)) + \
                                s_qty_p * (paper_state.get("short_entry_price", price) - price)
                        safe_real_equity = paper_state["balance"] + u_pnl
                        safe_positions = paper_state["positions"]
                    else:
                        m_info_new = await connector.get_margin_ratio()
                        safe_real_equity = m_info_new.get("total_margin_balance", 0.0) - siphoning_reserve
                        raw_positions_new = await connector.get_positions()
                        safe_positions = {k: v["qty"] for k, v in raw_positions_new.items()}
                        safe_l_entry = raw_positions_new.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                        safe_s_entry = raw_positions_new.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
                else:
                    safe_real_equity = real_equity
                    safe_positions = positions
                    safe_l_entry = l_entry
                    safe_s_entry = s_entry

                if paper_mode:
                    safe_l_entry = paper_state.get("long_entry_price", price)
                    safe_s_entry = paper_state.get("short_entry_price", price)

                # Calculate surplus using CURRENT virtual parameters (Basis price was reset ONLY if rebalance happened)
                # Re-calculate in separate process to get final TPV for siphoning
                safe_calc_res = await loop.run_in_executor(
                    process_executor, calculate_portfolio_task,
                    safe_positions, price, safe_real_equity,
                    virt_basis_price, virt_allocated_usdt,
                    base_ticker, siphoning_reserve, targets, initial_tpv,
                    -1.0, True, safe_l_entry, safe_s_entry
                )

                total_tpv_final = safe_calc_res["total_tpv"]
                # SURPLUS = Current Total Capital (including reserve) - Initial Targeted Capital
                total_surplus: float = total_tpv_final - initial_tpv
                siphoning_threshold_abs: float = initial_tpv * (siphoning_threshold_pct / 100)

                # Siphon only if total_surplus > existing reserve (meaning there is NEW profit)
                if total_surplus > siphoning_reserve + max(0.1, siphoning_threshold_abs):
                    new_profit = total_surplus - siphoning_reserve
                    siphon_amount: float = new_profit * (1 - reinvestment_ratio)
                    
                    if siphon_amount > 0.1:
                        siphoning_reserve += siphon_amount
                        if paper_mode:
                            paper_state["balance"] -= siphon_amount
                            await save_json(paper_state_file_path, paper_state)

                        state["siphoning_reserve"] = siphoning_reserve
                        logger.info(f"💰 SAFE ACTIVATED: Siphoned {siphon_amount:.4f} USDT. New Reserve: {siphoning_reserve:.2f}")
                        asyncio.create_task(notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT (Surplus)"))

                # Update reporting value in summary to account for new reserve
                final_reported_tpv = total_tpv_final

                if actions and len(valid_actions) > 0:
                    summary_msg = (
                        f"<b>✅ Rebalance #{cycles} Complete</b>: <code>{base_ticker}</code>\n"
                        f"New Shares: L:{safe_calc_res['share_long_pct']:.1f}% S:{safe_calc_res['share_short_pct']:.1f}% V:{safe_calc_res['share_virt_pct']:.1f}%\n"
                        f"TPV: <code>{total_tpv_final:.2f} USDT</code>"
                    )
                    logger.info(f"Rebalance #{cycles} complete. TPV: {total_tpv_final:.2f}")
                    asyncio.create_task(notifier.send_message(summary_msg))

                # Всегда обновляем стейт для агрегатора статусов в конце каждого цикла
                state.update({
                    "last_tpv": total_tpv_final,
                    "last_profit": total_tpv_final - initial_tpv,
                    "last_update": time.time(),
                    "rebalance_cycles": cycles
                })
                await save_json(state_file_path, state)

                if (i + status_offset) % 100 == 0:
                    logger.info(f"Heartbeat: TPV={total_tpv_final:.2f} | PnL={total_tpv_final - initial_tpv:+.2f} | Cycles={cycles}")

            except Exception as e:
                logger.error(f"Error in cycle: {e}")
                await asyncio.sleep(10)
            await asyncio.sleep(check_interval)
            i += 1
    finally:
        await notifier.close()

async def emergency_stop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    paper_mode = config.get("paper_mode", False)
    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
    
    logger.info(f"🛑 EMERGENCY STOP for {base_ticker} (Paper: {paper_mode})")
    
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    if paper_mode:
        paper_state = await load_json(paper_state_file_path, {})
        if paper_state and "positions" in paper_state:
            for pos_key, qty in paper_state["positions"].items():
                if qty != 0:
                    logger.info(f"Closing PAPER position {pos_key}: {qty}")
            paper_state["positions"] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
            paper_state["long_entry_price"] = 0.0
            paper_state["short_entry_price"] = 0.0
            await save_json(paper_state_file_path, paper_state)
    else:
        raw_positions = await connector.get_positions()
        for pos_key, data in raw_positions.items():
            if base_ticker in pos_key:
                qty = data["qty"]
                if qty != 0:
                    side = "SELL" if qty > 0 else "BUY"
                    step_size = step_sizes.get(base_ticker, 0.0)
                    logger.info(f"Closing REAL position {pos_key}: {qty}")
                    await PortfolioExecutor(connector).execute_market_order(base_ticker, abs(qty), side, step_size, True, pos_key.split('_')[1] if '_' in pos_key else "BOTH")

    # Сброс состояния
    state = await load_json(state_file_path, {})
    state["virt_basis_price"] = 0.0
    state["virt_allocated_usdt"] = 0.0
    state["initial_tpv"] = 0.0 
    state["reference_tpv"] = 0.0
    state["tpv_ath"] = 0.0
    await save_json(state_file_path, state)
    logger.info(f"✅ Emergency stop completed for {base_ticker}. All positions closed and state reset.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--stop", action="store_true", help="Close all positions and stop")
    parser.add_argument("--paper", action="store_true", help="Force paper mode for this instance")
    args = parser.parse_args()
    
    config_base = os.path.splitext(os.path.basename(args.config))[0]

    def get_initial_cfg():
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    cfg = get_initial_cfg()
    base_ticker = args.ticker if args.ticker else cfg.get("base_ticker", "BTCUSDT")
    
    # Paper mode logic: flag --paper OR global config paper_mode
    is_paper_instance = args.paper or cfg.get("paper_mode", False)
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rebalance_{base_ticker}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(base_ticker)
    
    instance_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"state_{base_ticker}.json"))
    instance_paper_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"paper_state_{base_ticker}.json"))
    
    api_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))
    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))

    if args.stop:
        asyncio.run(emergency_stop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker))
    else:
        logger.info(f"💾 State files: REAL={instance_state_file}, PAPER={instance_paper_state_file} | Mode: {'PAPER' if is_paper_instance else 'REAL'}")
        # Передаем признак paper_mode в rebalance_loop через конфиг-обертку или напрямую, 
        # но rebalance_loop читает конфиг из файла. Лучше пропатчить rebalance_loop чтобы он принимал paper_mode_override.
        asyncio.run(rebalance_loop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker, paper_mode_override=is_paper_instance))
