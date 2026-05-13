import asyncio
import json
import logging
import os
import time
import random
import shutil
from collections import deque
from typing import Dict, List, Any
from decimal import Decimal
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier
from storage import safe_load_json as load_json, safe_save_json as save_json

from pathlib import Path

# ProcessPoolExecutor removed to reduce latency

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
    # [SSOT] Absolute Scope Safety - Initialize all variables at function start
    i = 0
    status_offset = random.randint(0, 99)
    target_initial = 0.0
    max_spread = 0.0015 # 0.15%
    max_velocity = 0.01 # 1.0%
    velocity_window = 60
    
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
    
    # SSOT Capital: Use paper_initial_capital for PAPER, initial_capital for REAL
    if paper_mode:
        target_initial_cap = float(portfolio_cfg.get("paper_initial_capital", 100.0))
    else:
        target_initial_cap = float(portfolio_cfg.get("initial_capital", 86.0))
    
    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", target_initial_cap)

    # Состояние синтетической доли и сейфа
    state = await load_json(state_file_path, {
        "virt_qty": 0.0,
        "virt_entry_price": 0.0,
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "balance": target_initial_cap, # SSOT Balance
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,  # Фиксированная база для гистерезиса
        "tpv_ath": 0.0,
        "trailing_stop_violation_start": 0.0,
        "trailing_stop_paper_timeout_end": 0.0,
        "rebalance_cycles": 0,
        "last_rebalance_price": 0.0,
        "started_at": time.time()
    })

    # Initialize paper_state for shadow balance tracking (Used in both PAPER and REAL modes for isolation)
    default_paper_state = {
        "balance": target_initial_cap,
        "positions": {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0},
        "last_price": 0.0,
        "base_ticker": base_ticker,
        "long_entry_price": 0.0,
        "short_entry_price": 0.0
    }
    paper_state = await load_json(paper_state_file_path, default_paper_state)

    # Ensure balance exists (Shadow Balance Migration Guard)
    if "balance" not in paper_state:
        fallback_bal = state.get("initial_tpv", target_initial_cap)
        if fallback_bal <= 0: fallback_bal = target_initial_cap
        paper_state["balance"] = fallback_bal
        logger.warning(f"⚠️ 'balance' missing in {paper_state_file_path}. Initialized to {fallback_bal}")

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

    if paper_state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in paper state changed from {paper_state.get('base_ticker')} to {base_ticker}. Resetting.")
        paper_state["last_price"] = 0.0
        paper_state["base_ticker"] = base_ticker
        paper_state["balance"] = max_capital_usdt if max_capital_usdt > 0 else 10000.0
        paper_state["positions"] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
        paper_state["long_entry_price"] = 0.0
        paper_state["short_entry_price"] = 0.0
        await save_json(paper_state_file_path, paper_state)

    if "positions" not in paper_state: paper_state["positions"] = {}
    if f"{base_ticker}_LONG" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_LONG"] = 0.0
    if f"{base_ticker}_SHORT" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_SHORT"] = 0.0
    if "long_entry_price" not in paper_state: paper_state["long_entry_price"] = 0.0
    if "short_entry_price" not in paper_state: paper_state["short_entry_price"] = 0.0

    # [Architectural Safeguard] State Isolation Protocol
    # Если реальных позиций нет, это чистый старт (или рестарт после стопа).
    # Жестко затираем фантомные балансы, чтобы не сломать Trailing Stop.
    if not paper_mode:
        try:
            raw_positions = await connector.get_positions()
            ticker_positions = {k: v["qty"] for k, v in raw_positions.items() if base_ticker in k}
            total_position_size = sum(abs(float(v)) for v in ticker_positions.values())

            if total_position_size == 0:
                initial_cap = target_initial_cap

                # Проверяем оба стейта на наличие фантомного профита
                current_balance = float(paper_state.get('balance', initial_cap))
                if abs(current_balance - initial_cap) > 0.1 or float(state.get('virt_qty', 0)) > 0:
                    logger.warning(f"🧹 Phantom Buffer detected for {base_ticker}. Enforcing Clean Slate for baseline!")

                    # Сброс бумажного стейта (баланс и позиции)
                    paper_state['balance'] = initial_cap
                    paper_state['long_entry_price'] = 0.0
                    paper_state['short_entry_price'] = 0.0
                    paper_state['positions'] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
                    await save_json(paper_state_file_path, paper_state)

                    # Сброс основного стейта (ATH, V-нога, циклы)
                    state['tpv_ath'] = initial_cap
                    state['virt_qty'] = 0.0
                    state['rebalance_cycles'] = 0
                    state['initial_tpv'] = initial_cap
                    state['reference_tpv'] = initial_cap
                    await save_json(state_file_path, state)
        except Exception as e:
            logger.error(f"State Isolation Protocol failed: {e}")

    virt_qty = float(state.get("virt_qty", 0.0))
    virt_entry_price = float(state.get("virt_entry_price", 0.0))
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

    # [Safety] Absolute Scope Safety - Initialize all variables before the loop
    i = 0
    status_offset = random.randint(0, 99)
    target_initial = target_initial_cap
    velocity_cfg = portfolio_cfg.get("safety_guards", {})
    max_spread = velocity_cfg.get("max_spread_pct", 0.15) / 100
    max_velocity = velocity_cfg.get("max_price_velocity_pct", 1.0) / 100
    velocity_window = velocity_cfg.get("velocity_window_sec", 60)
    price_history = deque() # Будет хранить (timestamp, price)

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
                    
                    if i % 20 == 0:
                        logger.info(f"⚙️ Active Threshold for {base_ticker}: {threshold*100:.2f}%")

                    siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
                    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
                    
                    if paper_mode:
                        target_initial = float(portfolio_cfg.get("paper_initial_capital", 100.0))
                    else:
                        target_initial = float(portfolio_cfg.get("initial_capital", 86.0))
                        
                    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", target_initial)
                    max_drawdown_limit = current_config.get("max_drawdown_limit", 0.5)
                    equity_trailing_stop_pct = current_config.get("equity_trailing_stop_pct", 0.0)
                    equity_trailing_stop_timeout_sec = current_config.get("equity_trailing_stop_timeout_sec", 0.0)
                    
                    # Обновляем параметры защит
                    guards_cfg = portfolio_cfg.get("safety_guards", {})
                    max_spread = guards_cfg.get("max_spread_pct", 0.15) / 100
                    max_velocity = guards_cfg.get("max_price_velocity_pct", 1.0) / 100
                    velocity_window = guards_cfg.get("velocity_window_sec", 60)

                except Exception as e:
                    logger.error(f"Error reloading config: {e}. Using previous values.")

                # Dynamic initial_tpv update from config
                if initial_tpv != target_initial and target_initial > 0:
                    logger.info(f"🔄 Initial Capital changed in config: {initial_tpv} -> {target_initial}. Updating base.")
                    initial_tpv = target_initial
                    reference_tpv = initial_tpv
                    state["initial_tpv"] = initial_tpv
                    state["reference_tpv"] = reference_tpv

                # Use Mark Price for TPV and rebalance triggers as recommended by Audit
                prices = await connector.get_mark_prices([base_ticker])
                price = prices.get(base_ticker)
                if not price: raise Exception(f"Could not fetch {base_ticker} mark price")
                
                # -------------------------------------------------------------------------
                # [SAFETY] VELOCITY GUARD
                # -------------------------------------------------------------------------
                now = time.time()
                price_history.append((now, price))
                # Удаляем старые записи за пределами окна
                while price_history and (now - price_history[0][0]) > velocity_window:
                    price_history.popleft()
                
                if len(price_history) > 1:
                    old_t, old_p = price_history[0]
                    velocity = abs(price - old_p) / old_p
                    if velocity > max_velocity:
                        if i % 5 == 0:
                            logger.warning(f"🚀 Velocity Guard: {base_ticker} is moving too fast ({velocity*100:.2f}% in {int(now-old_t)}s). Blocking trades.")
                        await asyncio.sleep(check_interval)
                        i += 1
                        continue

                # -------------------------------------------------------------------------
                # [SAFETY] SPREAD GUARD (Only for REAL mode or detailed Paper simulation)
                # -------------------------------------------------------------------------
                if not paper_mode:
                    try:
                        depth = await connector.get_order_book(base_ticker, limit=5)
                        best_bid = float(depth['bids'][0][0]) if depth['bids'] else 0
                        best_ask = float(depth['asks'][0][0]) if depth['asks'] else 0
                        if best_bid > 0 and best_ask > 0:
                            spread = (best_ask - best_bid) / best_bid
                            if spread > max_spread:
                                if i % 5 == 0:
                                    logger.warning(f"⚠️ Spread Guard: {base_ticker} spread too wide ({spread*100:.3f}% > {max_spread*100:.3f}%). Blocking trades.")
                                await asyncio.sleep(check_interval)
                                i += 1
                                continue
                    except Exception as e:
                        logger.error(f"Failed to check order book for spread: {e}")

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
                # [SSOT] TPV = Cash + Virtual Value. Cash is paper_state["balance"].
                # In calculator.py, real_equity is treated as Wallet Balance.
                real_equity = paper_state["balance"]
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
                        
                        # SSOT Fix: Do not deduct virtual cost from balance.
                        # TPV now reflects real_equity + virt_pnl
                        
                        logger.info(f"Initialized Virtual Quantity: {virt_qty:.6f} {base_ticker} (Target Cost: {v_cost:.2f} USDT, Not deducted from Balance)")
                        state["virt_entry_price"] = price

                    state.update({
                        "virt_qty": virt_qty,
                        "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                    })

                # REMOVED: Migration Sanity Check (Caused TPV leakage)

                # Direct synchronous call to PortfolioCalculator to reduce latency
                current_threshold = -1.0 if (abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0)) == 0) else threshold
                
                calc = PortfolioCalculator(
                    positions=positions,
                    spot_price=price,
                    real_equity=real_equity,
                    virt_qty=virt_qty,
                    base_ticker=base_ticker,
                    siphoning_reserve=siphoning_reserve,
                    targets=targets,
                    initial_capital=initial_tpv,
                    long_entry_price=l_entry,
                    short_entry_price=s_entry,
                    virt_entry_price=virt_entry_price
                )
                calc_res = calc.calculate_rebalance(targets, current_threshold, (current_threshold < 0))
                
                tpv_total = calc_res["total_tpv"]
                tpv_active = calc_res["tpv"]
                actions = calc_res["actions"]

                now = time.time()

                # EMERGENCY STOP: If total_tpv (including SAFE) drops below max_drawdown_limit % of initial_tpv
                drawdown_threshold = initial_tpv * (1 - max_drawdown_limit / 100)
                if initial_tpv > 0 and tpv_total < drawdown_threshold:
                    msg = f"CRITICAL: Total Equity {tpv_total:.2f} (including SAFE) is less than {drawdown_threshold:.2f} ({max_drawdown_limit}% drawdown limit). EMERGENCY STOP!"
                    logger.critical(msg)
                    asyncio.create_task(notifier.send_alert("EMERGENCY STOP", msg))
                    
                    # Проверяем прибыль относительно глобального начального капитала
                    global_initial = portfolio_cfg.get("initial_capital", 60.0)
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
                            paper_state["balance"] = portfolio_cfg.get("initial_capital", 60.0)
                            paper_state["long_entry_price"] = 0.0
                            paper_state["short_entry_price"] = 0.0
                            await save_json(paper_state_file_path, paper_state)
                            
                            # Set paper probation timeout (from probation_period_days)
                            probation_days = current_config.get("probation_period_days", 0.041)
                            timeout_end = now + probation_days * 86400
                            state["trailing_stop_paper_timeout_end"] = timeout_end
                            logger.info(f"Setting post-stop paper probation until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeout_end))}")

                            # Эмитируем сигнал остановки для супервайзера
                            # Проверяем прибыль относительно глобального начального капитала
                            global_initial = portfolio_cfg.get("initial_capital", 60.0)
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
                    def get_dev(actual, target): return (actual - target) * 100

                    l_p, s_p, v_p = calc_res['share_long_pct'], calc_res['share_short_pct'], calc_res['share_virt_pct']
                    c_p = calc_res['share_cash_pct']
                    pnl_l, pnl_s, pnl_v = calc_res['pnl_l'], calc_res['pnl_s'], calc_res['pnl_v']

                    l_target = Decimal(str(targets['BASE_LONG']['share']))
                    s_target = Decimal(str(targets['BASE_SHORT']['share']))
                    v_target = Decimal(str(targets['VIRTUAL']['share']))

                    h_msg = (
                        f"Heartbeat: TPV={tpv_total:.2f}{res_str} | PnL={tpv_total - initial_tpv:+.2f} | {base_ticker}={price:.6g} | "
                        f"L:{l_p:.1f}% [{get_dev(Decimal(str(l_p))/100, l_target):+.1f}%] {{{pnl_l:+.2f}$}} | "
                        f"S:{s_p:.1f}% [{get_dev(Decimal(str(s_p))/100, s_target):+.1f}%] {{{pnl_s:+.2f}$}} | "
                        f"V:{v_p:.1f}% [{get_dev(Decimal(str(v_p))/100, v_target):+.1f}%] {{{pnl_v:+.2f}$}} | "
                        f"C:{c_p:.1f}%"
                    )
                    logger.info(h_msg)
                
                # Логика ребалансировки
                valid_actions = []
                if actions:
                    # Внедряем Notional Value Guard для ВСЕХ ордеров
                    # Проверяем и в портфеле, и в глобальном конфиге
                    min_notional = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt", 6.0))
                    valid_actions = [a for a in actions if abs(a.get("diff_usdt", 0)) >= min_notional]
                    
                    # -------------------------------------------------------------------------
                    # [V3.6.0] ANTI-CHURN PRICE FUSE (Enforce BLSH)
                    # -------------------------------------------------------------------------
                    last_reb_price = state.get("last_rebalance_price", 0.0)
                    fused_actions = []
                    
                    if last_reb_price == 0:
                        # Cold start: always allow first rebalance
                        fused_actions = valid_actions
                        logger.info(f"❄️ Cold Start: Allowing all actions to form portfolio baseline at {price:.6g}")
                    else:
                        for action in valid_actions:
                            side = action.get("side")
                            act_type = action.get("type", "REAL") # VIRTUAL_ORDER or position side
                            
                            if side == "BUY":
                                # Buy only if price dropped enough
                                limit_price = last_reb_price * (1 - threshold)
                                if price <= limit_price:
                                    fused_actions.append(action)
                                else:
                                    logger.warning(f"🚫 FUSE ({act_type}): Buy blocked. {price:.6g} > {limit_price:.6g} (Last: {last_reb_price:.6g})")
                            elif side == "SELL":
                                # Sell only if price rose enough
                                limit_price = last_reb_price * (1 + threshold)
                                if price >= limit_price:
                                    fused_actions.append(action)
                                else:
                                    logger.warning(f"🚫 FUSE ({act_type}): Sell blocked. {price:.6g} < {limit_price:.6g} (Last: {last_reb_price:.6g})")
                            else:
                                fused_actions.append(action) # Safety for unknown types
                    
                    if fused_actions:
                        # Log specific trigger reasons
                        for action in fused_actions:
                            act_type = action.get("type")
                            side = action.get("position_side", "BOTH")
                            symbol = action.get("symbol")
                            diff_usdt = action.get("diff_usdt", 0)

                            key_map = {"ORDER": "BASE_" + side, "VIRTUAL_ORDER": "VIRTUAL"}
                            key = key_map.get(act_type, symbol)

                            share_suffix = "virt" if key == "VIRTUAL" else key.split('_')[-1].lower()
                            current_share = Decimal(str(calc_res.get(f"share_{share_suffix}_pct", 0))) / 100
                            target_share = Decimal(str(targets.get(key, {}).get("share", 0)))
                            dev = (current_share - target_share) * 100

                            trigger_key = key.replace("BASE_", "")
                            logger.info(f"Rebalance triggered: {trigger_key} deviation {dev:+.2f}% exceeds threshold {threshold*100:.2f}%")

                        logger.info(f"Rebalance needed ({len(fused_actions)} fused actions). Shares: L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}%\nTPV: {tpv_active:.2f}")
                        
                        rebalance_msg = (
                            f"🔄 <b>Rebalance #{cycles + 1} Starting</b>: <code>{base_ticker}</code>\n"
                            f"Shares: L:{calc_res['share_long_pct']:.1f}% S:{calc_res['share_short_pct']:.1f}% V:{calc_res['share_virt_pct']:.1f}%\n"
                            f"TPV: <code>{tpv_total:.2f} USDT</code>"
                        )
                        asyncio.create_task(notifier.send_message(rebalance_msg))

                        # 1. Execute actions concurrently
                        exec_results = await executor.execute_actions(
                            fused_actions, price, paper_mode, portfolio_cfg, step_sizes, paper_state
                        )

                        any_success = False
                        # 2. Update core states (Virtual Quantity, Paper Positions/Balance)
                        for res in exec_results:
                            status = res.get("status")
                            pos_side = res.get("type", "UNKNOWN")

                            if status == "ERROR":
                                logger.error(f"Action failed: {res.get('message')}")
                                continue

                            if status in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]:
                                any_success = True
                                if res.get("type") == "VIRTUAL_ORDER":
                                    diff_usdt = Decimal(str(res.get("diff_usdt", 0.0)))
                                    dec_price = Decimal(str(price))
                                    virt_qty_before = Decimal(str(virt_qty))

                                    # 1. Списание/начисление кэша (Cash Accounting)
                                    paper_state["balance"] -= float(diff_usdt)

                                    # 2. Обновление количества
                                    new_v_qty = virt_qty_before + diff_usdt / dec_price

                                    # 3. Установка цены входа (только при открытии с нуля)
                                    if diff_usdt > 0 and virt_qty_before == 0:
                                        state["virt_entry_price"] = float(dec_price)

                                    # 4. Value-based Dust Guard: если позиция меньше 1.0 USDT — в ноль
                                    if abs(new_v_qty * dec_price) < Decimal('1.0'):
                                        # Возвращаем остатки в кэш перед обнулением
                                        paper_state["balance"] += float(new_v_qty * dec_price)
                                        new_v_qty = Decimal('0')
                                        state["virt_entry_price"] = 0.0
                                        logger.info(f"🧹 Dust cleaned: Position value < 1.0 USDT")

                                    virt_qty = float(new_v_qty)
                                    state["virt_qty"] = virt_qty
                                    
                                    logger.info(f"{'➕ VIRTUAL BUY' if diff_usdt > 0 else '➖ VIRTUAL SELL'}: {abs(float(diff_usdt)):.2f} USDT")
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
                                # asyncio.create_task(notifier.send_message(f"<b>{trade_log}</b>"))
                            else:
                                logger.warning(f"❌ {side} {key} execution status: {status}. Message: {res.get('message')}")

                        if any_success:
                            state["last_rebalance_price"] = price
                            logger.info(f"🎯 Baseline Updated: Last rebalance price set to {price:.6g}")

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
                        safe_l_entry = paper_state.get("long_entry_price", price)
                        safe_s_entry = paper_state.get("short_entry_price", price)
                    else:
                        # In real mode, use exchange entry prices for better accuracy
                        raw_positions_new = await connector.get_positions()
                        safe_l_entry = raw_positions_new.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                        safe_s_entry = raw_positions_new.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)
                        
                    safe_real_equity = paper_state["balance"]
                    safe_positions = paper_state["positions"]
                else:
                    safe_real_equity = paper_state["balance"]
                    safe_positions = positions
                    safe_l_entry = l_entry
                    safe_s_entry = s_entry

                if paper_mode:
                    safe_l_entry = paper_state.get("long_entry_price", price)
                    safe_s_entry = paper_state.get("short_entry_price", price)

                # Calculate surplus using CURRENT virtual parameters
                safe_calc = PortfolioCalculator(
                    positions=safe_positions,
                    spot_price=price,
                    real_equity=safe_real_equity,
                    virt_qty=virt_qty,
                    base_ticker=base_ticker,
                    siphoning_reserve=siphoning_reserve,
                    targets=targets,
                    initial_capital=initial_tpv,
                    long_entry_price=safe_l_entry,
                    short_entry_price=safe_s_entry,
                    virt_entry_price=virt_entry_price
                )
                safe_calc_res = safe_calc.calculate_rebalance(targets, -1.0, True)

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
                    "total_pnl_pct": safe_calc_res.get("total_pnl_pct", 0.0),
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
    initial_capital = portfolio_cfg.get("initial_capital", 60.0)

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
                "final_profit": state.get("last_profit", 0.0)
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
            "virt_entry_price": 0.0,
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
    parser.add_argument("--wipe", action="store_true", help="Wipe bot state (destructive stop)")
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
    
    # Strict Isolation: Different prefixes for Paper and Real modes
    prefix = "paper" if is_paper_instance else "real"
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rebalance_{prefix}_{base_ticker}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(base_ticker)
    
    instance_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{prefix}_state_{base_ticker}.json"))
    
    if is_paper_instance:
        # В бумажном режиме основной стейт и есть бумажный стейт
        instance_paper_state_file = instance_state_file
    else:
        # В реальном режиме бот ведет "теневой" бумажный баланс в отдельном файле, 
        # чтобы не конфликтовать с основным бумажным ботом (мастер-рейтингом)
        instance_paper_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"shadow_state_{base_ticker}.json"))
    
    api_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))

    if os.environ.get("MOCK_MODE") == "1":
        from connector import BinanceConnectorMock
        connector = BinanceConnectorMock()
    else:
        connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))

    if args.stop:
        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: По умолчанию НЕ удаляем стейт при стопе. 
        # Только если явно передан --wipe
        should_wipe = args.wipe
        asyncio.run(emergency_stop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker, paper_mode=is_paper_instance, close_only=(not should_wipe)))
    else:
        logger.info(f"💾 State files: REAL={instance_state_file}, PAPER={instance_paper_state_file} | Mode: {'PAPER' if is_paper_instance else 'REAL'}")
        # Передаем признак paper_mode в rebalance_loop через конфиг-обертку или напрямую, 
        # но rebalance_loop читает конфиг из файла. Лучше пропатчить rebalance_loop чтобы он принимал paper_mode_override.
        asyncio.run(rebalance_loop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker, paper_mode_override=is_paper_instance))
