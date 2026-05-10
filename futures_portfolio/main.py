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

def calculate_portfolio_task(positions, price, real_equity, virt_qty, 
                             base_ticker, siphoning_reserve, targets, initial_capital, 
                             threshold, ignore_limits, long_entry_price=0.0, short_entry_price=0.0):
    """Heavy math task to be run in a separate process."""
    calc = PortfolioCalculator(
        positions=positions,
        spot_price=price,
        real_equity=real_equity,
        virt_qty=virt_qty,
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
        "virt_current_value": float(calc.notional_virt)
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
        "virt_qty": 0.0,
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,  # Фиксированная база для гистерезиса
        "tpv_ath": 0.0,
        "trailing_stop_violation_start": 0.0,
        "trailing_stop_paper_timeout_end": 0.0,
        "rebalance_cycles": 0,
        "started_at": time.time()
    })

    # Если тикер сменился, сбрасываем количество виртуальных монет и начальный TPV
    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state changed from {state.get('base_ticker')} to {base_ticker}. Resetting virt_qty, initial TPV and ATH.")
        state["virt_qty"] = 0.0
        state["initial_tpv"] = 0.0 
        state["reference_tpv"] = 0.0
        state["tpv_ath"] = 0.0
        state["trailing_stop_violation_start"] = 0.0
        state["trailing_stop_paper_timeout_end"] = 0.0
        state["base_ticker"] = base_ticker
        state["started_at"] = time.time()
        await save_json(state_file_path, state)

    virt_qty = float(state.get("virt_qty", 0.0))
    siphoning_reserve = float(state.get("siphoning_reserve", 0.0))
    initial_tpv = float(state.get("initial_tpv", 0.0))
    reference_tpv = float(state.get("reference_tpv", 0.0))
    tpv_ath = float(state.get("tpv_ath", 0.0))
    cycles = state.get("rebalance_cycles", 0)

    # Инфо о бирже
    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}
    
    equity_trailing_stop_pct = config.get("equity_trailing_stop_pct", 0.0)
    max_drawdown_limit = config.get("max_drawdown_limit", 0.5)
    margin_warning = portfolio_cfg.get("margin_ratio_warning", 5.0)
    margin_critical = portfolio_cfg.get("margin_ratio_critical", 2.0)

    # Initialize paper_state for shadow balance tracking (Used in both PAPER and REAL modes for isolation)
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
        paper_state["balance"] = max_capital_usdt if max_capital_usdt > 0 else 10000.0
        paper_state["positions"] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
        paper_state["long_entry_price"] = 0.0
        paper_state["short_entry_price"] = 0.0

    if "positions" not in paper_state: paper_state["positions"] = {}
    if f"{base_ticker}_LONG" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_LONG"] = 0.0
    if f"{base_ticker}_SHORT" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_SHORT"] = 0.0
    if "long_entry_price" not in paper_state: paper_state["long_entry_price"] = 0.0
    if "short_entry_price" not in paper_state: paper_state["short_entry_price"] = 0.0

    # Инициализация уведомлений
    notifier = TelegramNotifier()
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    
    # Инициализация экзекутора
    max_ops = config.get("max_orders_per_second", 10)
    executor = PortfolioExecutor(connector, base_ticker=base_ticker, max_orders_per_second=max_ops)

    # Hedge Mode, Leverage and Margin Type Guard
    if not paper_mode:
        try:
            is_hedge = await connector.get_hedge_mode()
            if not is_hedge:
                msg = f"CRITICAL: Hedge Mode is DISABLED on Binance for {base_ticker}. Please enable it to start the bot."
                logger.error(msg)
                asyncio.create_task(notifier.send_alert("STARTUP ERROR", msg))
                return
            logger.info("Hedge Mode verified.")

            # Force Leverage 5x and Isolated Margin
            try:
                await connector.set_leverage(base_ticker, 5)
                logger.info(f"Leverage set to 5x for {base_ticker}")
            except Exception as e:
                logger.warning(f"Could not set leverage for {base_ticker}: {e}")

            try:
                await connector.set_margin_type(base_ticker, "ISOLATED")
                logger.info(f"Margin Type set to ISOLATED for {base_ticker}")
            except Exception as e:
                logger.warning(f"Could not set margin type for {base_ticker}: {e}")

        except Exception as e:
            logger.error(f"Failed to verify exchange settings: {e}")
            asyncio.create_task(notifier.send_alert("STARTUP ERROR", f"Could not verify exchange settings: {e}"))
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
                    max_drawdown_limit = current_config.get("max_drawdown_limit", 0.5)
                    equity_trailing_stop_pct = current_config.get("equity_trailing_stop_pct", 0.0)
                    equity_trailing_stop_timeout_sec = current_config.get("equity_trailing_stop_timeout_sec", 0.0)
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

                # ALWAYS use shadow balance (paper_state) for real_equity calculation to support shared accounts
                # This ensures per-bot PnL isolation and prevents double-counting of account-wide profit
                if paper_mode:
                    paper_state["last_price"] = price
                    l_qty = abs(paper_state["positions"].get(f"{base_ticker}_LONG", 0.0))
                    s_qty = abs(paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0))
                    l_entry = paper_state.get("long_entry_price", price)
                    s_entry = paper_state.get("short_entry_price", price)
                    m_info = {}
                else:
                    # In REAL mode, we sync positions from exchange but keep balance in shadow
                    raw_positions = await connector.get_positions()
                    # Filter for this ticker only
                    ticker_positions = {k: v["qty"] for k, v in raw_positions.items() if base_ticker in k}
                    l_qty = abs(ticker_positions.get(f"{base_ticker}_LONG", 0.0))
                    s_qty = abs(ticker_positions.get(f"{base_ticker}_SHORT", 0.0))
                    l_entry = raw_positions.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                    s_entry = raw_positions.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
                    
                    # For logging and safety only
                    m_info = await connector.get_margin_ratio()
                    
                # Calculate isolated PnL and Equity
                u_pnl = l_qty * (price - l_entry) + s_qty * (s_entry - price)
                real_equity = paper_state["balance"] + u_pnl
                positions = paper_state["positions"] if paper_mode else {k: v for k, v in ticker_positions.items()}

                if virt_qty == 0 or initial_tpv == 0:
                    if initial_tpv == 0:
                        # For shared accounts, we MUST use assigned initial_capital as base
                        initial_tpv = portfolio_cfg.get("initial_capital", real_equity)
                        reference_tpv = initial_tpv
                        logger.info(f"Initialized TPV base: {initial_tpv:.2f} (Isolated Shadow Balance)")

                    if virt_qty == 0:
                        # Initialize Virtual Quantity as spot equivalent
                        v_share = targets["VIRTUAL"]["share"]
                        v_cost = initial_tpv * v_share
                        virt_qty = v_cost / price
                        
                        # CRITICAL: Deduct virtual cost from balance to maintain 100% TPV invariant
                        paper_state["balance"] -= v_cost
                        real_equity -= v_cost # Update local variable for immediate consistency
                        
                        logger.info(f"Initialized Virtual Quantity: {virt_qty:.6f} {base_ticker} (Cost: {v_cost:.2f} USDT deducted from Balance)")
                        await save_json(paper_state_file_path, paper_state)

                    state.update({
                        "virt_qty": virt_qty,
                        "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                    })

                # One-time Sanity Check for existing bots (Migration from buggy version)
                # If TPV is ~135% of initial and it's the first cycles, fix the double-counting
                if cycles <= 10 and (real_equity + virt_qty * price) > initial_tpv * 1.25:
                    v_cost = initial_tpv * targets["VIRTUAL"]["share"]
                    logger.warning(f"⚠️ Sanity Check: Detected double-counted Virtual leg for {base_ticker}. Adjusting balance by -{v_cost:.2f} USDT.")
                    paper_state["balance"] -= v_cost
                    real_equity -= v_cost
                    await save_json(paper_state_file_path, paper_state)

                # Offload heavy math to ProcessPoolExecutor
                loop = asyncio.get_running_loop()
                current_threshold = -1.0 if (abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0)) == 0) else threshold
                
                calc_res = await loop.run_in_executor(
                    process_executor, calculate_portfolio_task,
                    positions, price, real_equity, virt_qty, 
                    base_ticker, siphoning_reserve, targets, initial_tpv, 
                    current_threshold, (current_threshold < 0),
                    l_entry, s_entry
                )
                
                tpv_total = calc_res["total_tpv"]
                tpv_active = calc_res["tpv"]
                actions = calc_res["actions"]

                # Rotation window PnL tracking for Supervisor (tied to probation_period_days)
                rotation_window_days = current_config.get("probation_period_days", 0.041)
                rotation_sec = max(300, rotation_window_days * 86400) # e.g. 1 hour
                
                now = time.time()
                last_prob_update = state.get("last_probation_update", 0)
                if now - last_prob_update > rotation_sec:
                    old_tpv = state.get("tpv_probation_basis", tpv_total)
                    state["profit_probation"] = tpv_total - old_tpv
                    state["tpv_probation_basis"] = tpv_total
                    state["last_probation_update"] = now
                elif "profit_probation" not in state:
                    # Fallback for first run
                    state["profit_probation"] = 0.0
                    state["tpv_probation_basis"] = tpv_total
                    state["last_probation_update"] = now

                # EMERGENCY STOP: If total_tpv (including SAFE) drops below max_drawdown_limit % of initial_tpv
                drawdown_threshold = initial_tpv * (1 - max_drawdown_limit / 100)
                if initial_tpv > 0 and tpv_total < drawdown_threshold:
                    msg = f"CRITICAL: Total Equity {tpv_total:.2f} (including SAFE) is less than {drawdown_threshold:.2f} ({max_drawdown_limit}% drawdown limit). EMERGENCY STOP!"
                    logger.critical(msg)
                    asyncio.create_task(notifier.send_alert("EMERGENCY STOP", msg))
                    
                    # Проверяем прибыль относительно глобального начального капитала
                    global_initial = portfolio_cfg.get("initial_capital", 65.0)
                    if tpv_total < global_initial:
                        emit_signal("stop", base_ticker)
                        logger.info(f"Sent STOP signal. Total Equity {tpv_total:.2f} < Global Initial {global_initial:.2f}. Ticker blacklisted.")
                    else:
                        emit_signal("exit", base_ticker)
                        logger.info(f"Sent EXIT signal. Total Equity {tpv_total:.2f} >= Global Initial {global_initial:.2f}. Ticker goes to probation.")
                        
                        # Устанавливаем таймаут пробации для прибыльного Emergency Stop
                        probation_days = current_config.get("probation_period_days", 0.041)
                        state["trailing_stop_paper_timeout_end"] = now + probation_days * 86400
                        state["trailing_stop_triggered"] = True
                        await save_json(state_file_path, state)

                    await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker, paper_mode=paper_mode)
                    return

                if tpv_ath == 0 or tpv_total > tpv_ath:
                    tpv_ath = tpv_total
                    state["tpv_ath"] = tpv_ath

                if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                    drawdown_pct = (1 - tpv_total / tpv_ath) * 100
                    if drawdown_pct >= equity_trailing_stop_pct:
                        violation_start = state.get("trailing_stop_violation_start", 0.0)
                        if violation_start == 0:
                            violation_start = now
                            state["trailing_stop_violation_start"] = violation_start
                            logger.warning(f"Trailing Stop threshold breached ({drawdown_pct:.2f}%). Timeout: {equity_trailing_stop_timeout_sec}s")
                        
                        elapsed = now - violation_start
                        if elapsed >= equity_trailing_stop_timeout_sec:
                            msg = f"Trailing Stop triggered: {drawdown_pct:.2f}% drop from ATH for {elapsed:.1f}s. Closing all positions for {base_ticker}."
                            logger.warning(f"!!! [STOP] {msg}")
                            asyncio.create_task(notifier.send_alert("STOP LOSS", msg))
                            
                            # Realize PnL and close positions
                            for pos_key, qty in positions.items():
                                if qty == 0 or base_ticker not in pos_key: continue
                                side = "SELL" if qty > 0 else "BUY"
                                step_size = step_sizes.get(base_ticker, 0.0)
                                
                                # Calculate realized PnL for shadow balance
                                p_qty = abs(paper_state["positions"].get(pos_key, 0.0))
                                if p_qty > 0:
                                    if "LONG" in pos_key:
                                        pnl = p_qty * (price - paper_state.get("long_entry_price", price))
                                    else:
                                        pnl = p_qty * (paper_state.get("short_entry_price", price) - price)
                                    paper_state["balance"] += pnl
                                    paper_state["positions"][pos_key] = 0.0

                                if not paper_mode:
                                    # In REAL mode, close real position
                                    await PortfolioExecutor(connector).execute_market_order(pos_key.split('_')[0], abs(qty), side, step_size, True, pos_key.split('_')[1] if '_' in pos_key else "BOTH")
                                    
                            # Reset shadow balance to initial capital to avoid loop on restart
                            paper_state["balance"] = portfolio_cfg.get("initial_capital", 65.0)
                            paper_state["long_entry_price"] = 0.0
                            paper_state["short_entry_price"] = 0.0
                            await save_json(paper_state_file_path, paper_state)
                            
                            # Set paper probation timeout (from probation_period_days)
                            probation_days = current_config.get("probation_period_days", 0.041)
                            timeout_end = now + probation_days * 86400
                            state["trailing_stop_paper_timeout_end"] = timeout_end
                            logger.info(f"Setting post-stop paper probation until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeout_end))}")

                            # Эмитируем сигнал остановки для супервайзера
                            # Проверяем прибыль относительно глобального начального капитала (65 USDT)
                            global_initial = portfolio_cfg.get("initial_capital", 65.0)
                            if tpv_total < global_initial:
                                emit_signal("stop", base_ticker)
                                logger.info(f"Sent STOP signal. Total Equity {tpv_total:.2f} < Global Initial {global_initial:.2f}. Ticker blacklisted.")
                            else:
                                emit_signal("exit", base_ticker)
                                logger.info(f"Sent EXIT signal. Total Equity {tpv_total:.2f} >= Global Initial {global_initial:.2f}. Ticker remains available (Probation).")
                            
                            # Сбрасываем ATH и начальные значения
                            state["tpv_ath"] = 0.0
                            state["initial_tpv"] = 0.0
                            state["reference_tpv"] = 0.0
                            state["virt_qty"] = 0.0
                            state["trailing_stop_triggered"] = True
                            state["trailing_stop_violation_start"] = 0.0
                            await save_json(state_file_path, state)
                            
                            logger.info("Positions closed and state reset. Bot stopped.")
                            break
                        else:
                            if i % 5 == 0:
                                logger.info(f"Trailing Stop Pending: {drawdown_pct:.2f}% (Wait {equity_trailing_stop_timeout_sec - elapsed:.1f}s more)")
                    else:
                        if state.get("trailing_stop_violation_start", 0.0) > 0:
                            logger.info(f"Trailing Stop Recovered: drawdown {drawdown_pct:.2f}% is back below {equity_trailing_stop_pct}%")
                            state["trailing_stop_violation_start"] = 0.0

                if not paper_mode and (margin_warning > 0 or margin_critical > 0):
                    try:
                        m_ratio = m_info.get("margin_ratio", 0.0)
                        if m_ratio > 0:
                            if m_ratio < portfolio_cfg.get("margin_ratio_critical", 2.0):
                                msg = f"Margin ratio {m_ratio:.2f} < {portfolio_cfg.get('margin_ratio_critical', 2.0)}. Emergency stop!"
                                logger.error(msg)
                                asyncio.create_task(notifier.send_alert("CRITICAL MARGIN", msg))
                                emit_signal("stop", base_ticker)
                                await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker, paper_mode=paper_mode)
                                return
                            elif m_ratio < portfolio_cfg.get("margin_ratio_warning", 5.0):
                                msg = f"Low margin ratio: {m_ratio:.2f}"
                                logger.warning(msg)
                                asyncio.create_task(notifier.send_message(f"⚠️ <b>WARNING</b>: {msg} ({base_ticker})"))
                    except Exception as e:
                        logger.error(f"Failed to check margin: {e}")

                if i % 5 == 0:
                    res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                    # prioritizing TPV in logs to match user's config expectation
                    logger.info(f"Heartbeat: TPV={tpv_total:.2f}{res_str} | PnL={tpv_total - initial_tpv:+.2f} | {base_ticker}={price:.6g} | L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}% C:{calc_res['share_cash_pct']:.1f}% (RealEquity:{real_equity:.2f})")
                
                # Логика ребалансировки
                valid_actions = []
                if actions:
                    # Внедряем Notional Value Guard для ВСЕХ ордеров
                    # Проверяем и в портфеле, и в глобальном конфиге
                    min_notional = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt", 6.0))
                    valid_actions = [a for a in actions if abs(a.get("diff_usdt", 0)) >= min_notional]
                    
                    if valid_actions:
                        logger.info(f"Rebalance needed ({len(valid_actions)} actions). Shares: L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}%\nTPV: {tpv_active:.2f}")
                        
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

                        # 2. Update core states (Virtual Quantity, Paper Positions/Balance)
                        for res in exec_results:
                            status = res.get("status")
                            pos_side = res.get("type", "UNKNOWN")

                            if status == "ERROR":
                                logger.error(f"Action failed: {res.get('message')}")
                                continue

                            if status in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]:
                                if res.get("type") == "VIRTUAL_ORDER":
                                    # Update virtual quantity based on the USDT diff (sold/bought from Cash)
                                    diff_usdt = res.get("diff_usdt", 0.0)
                                    # diff_usdt is the amount ADDED to Virtual (from Cash)
                                    # So we subtract it from paper_state["balance"] and add to virt_qty
                                    paper_state["balance"] -= diff_usdt
                                    virt_qty += diff_usdt / price
                                    state["virt_qty"] = virt_qty
                                    
                                    logger.info(f"🔄 Virtual Fixed: {diff_usdt:+.4f} USDT moved between Cash and Virtual. New Qty: {virt_qty:.6f}")
                                    continue

                                # Update execution results info
                                key = res.get("symbol", "UNKNOWN")
                                side = res.get("side", "UNKNOWN")
                                qty = res.get("qty", 0.0)
                                trade_pnl = res.get("trade_pnl", 0.0)
                                reduce_only = res.get("reduce_only", False)

                                # Shadow accounting for BOTH paper and real modes to support per-bot isolation
                                pos_key = f"{base_ticker}_{pos_side}"
                                old_qty = paper_state["positions"].get(pos_key, 0.0)
                                entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
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
                                
                                mode_tag = "PAPER" if paper_mode else "REAL"
                                trade_log = f"📝 {mode_tag}: {side} {qty} {pos_key} @ {price:.6g}"
                                if trade_pnl != 0: trade_log += f" | PnL: {trade_pnl:+.4f}"
                                logger.info(trade_log)
                                asyncio.create_task(notifier.send_message(f"<b>{trade_log}</b>"))
                            else:
                                logger.warning(f"❌ {side} {key} execution status: {status}. Message: {res.get('message')}")

                        cycles += 1
                        state["rebalance_cycles"] = cycles

                        # Always save paper_state to track shadow balance
                        await save_json(paper_state_file_path, paper_state)

                # 3. GLOBAL SAFE SIPHONING (Runs every cycle)
                if actions and len(valid_actions) > 0:
                    # ALWAYS use shadow balance (paper_state) for siphoning calculation to support shared accounts
                    l_qty_p = abs(paper_state["positions"].get(f"{base_ticker}_LONG", 0.0))
                    s_qty_p = abs(paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0))
                    
                    if paper_mode:
                        u_pnl_p = l_qty_p * (price - paper_state.get("long_entry_price", price)) + \
                                  s_qty_p * (paper_state.get("short_entry_price", price) - price)
                        safe_l_entry = paper_state.get("long_entry_price", price)
                        safe_s_entry = paper_state.get("short_entry_price", price)
                    else:
                        # In real mode, use exchange entry prices for better accuracy
                        raw_positions_new = await connector.get_positions()
                        safe_l_entry = raw_positions_new.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                        safe_s_entry = raw_positions_new.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
                        u_pnl_p = l_qty_p * (price - safe_l_entry) + s_qty_p * (safe_s_entry - price)
                        
                    safe_real_equity = paper_state["balance"] + u_pnl_p
                    safe_positions = paper_state["positions"]
                else:
                    safe_real_equity = real_equity
                    safe_positions = positions
                    safe_l_entry = l_entry
                    safe_s_entry = s_entry

                if paper_mode:
                    safe_l_entry = paper_state.get("long_entry_price", price)
                    safe_s_entry = paper_state.get("short_entry_price", price)

                # Calculate surplus using CURRENT virtual parameters
                loop = asyncio.get_running_loop()
                safe_calc_res = await loop.run_in_executor(
                    process_executor, calculate_portfolio_task,
                    safe_positions, price, safe_real_equity,
                    virt_qty,
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
                        # ALWAYS subtract from shadow balance to track isolated per-bot equity
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

async def emergency_stop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None, paper_mode: bool = False, close_only: bool = False):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}

    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
    portfolio_cfg = config.get("portfolios", [{}])[0]
    initial_capital = portfolio_cfg.get("initial_capital", 65.0)

    logger.info(f"🛑 EMERGENCY STOP for {base_ticker} (Paper: {paper_mode}, CloseOnly: {close_only})")

    # --- PERSISTENCE PROTOCOL: Archive state before reset ---
    if not close_only:
        try:
            history_dir = Path("history")
            history_dir.mkdir(exist_ok=True)
            state = await load_json(state_file_path, {})
            paper_state = await load_json(paper_state_file_path, {})

            archive_data = {
                "ticker": base_ticker,
                "timestamp": time.time(),
                "state": state,
                "paper_state": paper_state,
                "final_profit": (paper_state.get("balance", 0) + state.get("virt_qty", 0) * paper_state.get("last_price", 0)) - initial_capital + state.get("siphoning_reserve", 0)
            }
            archive_path = history_dir / f"archive_{base_ticker}_{int(time.time())}.json"
            await save_json(str(archive_path), archive_data)
            logger.info(f"💾 State archived to {archive_path}")
        except Exception as e:
            logger.error(f"Failed to archive state: {e}")

    exchange_info = await connector.get_exchange_info()
    step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}

    # Always try to load paper_state to reset it (unless close_only)
    paper_state = await load_json(paper_state_file_path, {})

    if paper_mode:
        if paper_state and "positions" in paper_state:
            for pos_key, qty in paper_state["positions"].items():
                if qty != 0:
                    logger.info(f"Closing PAPER position {pos_key}: {qty}")

        if not close_only:
            # Reset paper state to fresh start
            paper_state.update({
                "balance": initial_capital,
                "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
                "long_entry_price": 0.0,
                "short_entry_price": 0.0,
                "last_price": 0.0
            })
            await save_json(paper_state_file_path, paper_state)
    else:
        # REAL mode: close on exchange
        raw_positions = await connector.get_positions()
        for pos_key, data in raw_positions.items():
            if base_ticker in pos_key:
                qty = data["qty"]
                if qty != 0:
                    side = "SELL" if qty > 0 else "BUY"
                    step_size = step_sizes.get(base_ticker, 0.0)
                    logger.info(f"Closing REAL position {pos_key}: {qty}")
                    await PortfolioExecutor(connector).execute_market_order(
                        symbol=base_ticker,
                        qty=abs(qty),
                        side=side,
                        step_size=step_size,
                        reduce_only=True,
                        position_side=pos_key.split('_')[1] if '_' in pos_key else "BOTH",
                        min_notional=0.0
                    )

        if not close_only:
            if paper_state:
                paper_state.update({
                    "balance": initial_capital,
                    "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
                    "long_entry_price": 0.0,
                    "short_entry_price": 0.0
                })
                await save_json(paper_state_file_path, paper_state)

    if not close_only:
        # Сброс основного состояния
        state = await load_json(state_file_path, {})
        state.update({
            "virt_qty": 0.0,
            "initial_tpv": 0.0,
            "reference_tpv": 0.0,
            "tpv_ath": 0.0,
            "trailing_stop_triggered": False,
            "trailing_stop_violation_start": 0.0
        })
        await save_json(state_file_path, state)
        logger.info(f"✅ Emergency stop completed for {base_ticker}. All positions closed and state reset.")
    else:
        logger.info(f"✅ Positions closed for {base_ticker}. State preserved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--stop", action="store_true", help="Close all positions and stop")
    parser.add_argument("--close-only", action="store_true", help="Only close positions on exchange, preserve bot state")
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
        asyncio.run(emergency_stop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker, paper_mode=is_paper_instance, close_only=args.close_only))
    else:
        logger.info(f"💾 State files: REAL={instance_state_file}, PAPER={instance_paper_state_file} | Mode: {'PAPER' if is_paper_instance else 'REAL'}")
        # Передаем признак paper_mode в rebalance_loop через конфиг-обертку или напрямую, 
        # но rebalance_loop читает конфиг из файла. Лучше пропатчить rebalance_loop чтобы он принимал paper_mode_override.
        asyncio.run(rebalance_loop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker, paper_mode_override=is_paper_instance))
