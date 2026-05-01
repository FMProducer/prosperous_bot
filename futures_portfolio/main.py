import asyncio
import json
import logging
import os
import time
import random
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier

import aiofiles
from filelock import FileLock, Timeout
from pathlib import Path

def read_shared_config(path: str) -> Dict[str, Any]:
    """Чтение общего конфига без использования блокировок для исключения конкуренции."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Critical: Shared config read failed: {e}")
        return {}

def emit_signal(signal_type: str, ticker: str) -> None:
    """Создает пустой файл-флаг для супервайзера."""
    sig_path = Path("signals") / f"{signal_type}_{ticker}.flag"
    try:
        sig_path.touch(exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to emit signal {signal_type} for {ticker}: {e}")

async def load_json(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Lock-free read. Полагаемся на атомарность файловой системы."""
    try:
        if not os.path.exists(path):
            return default
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        return _STATE_CACHE.get(path, default)

# Глобальный кэш для защиты состояния
_STATE_CACHE: Dict[str, Dict[str, Any]] = {}

async def save_json(path: str, data: Dict[str, Any]) -> None:
    """Атомарная запись без блокировок. os.replace гарантирует консистентность на Windows."""
    _STATE_CACHE[path] = data
    try:
        tmp_path = f"{path}.tmp"
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))
        os.replace(tmp_path, path)
        if path in _STATE_CACHE: del _STATE_CACHE[path]
    except Exception as e:
        logging.warning(f"Failed to save {path}, cached in memory: {e}")

async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None, paper_mode_override: bool = None):
    # Defaults to satisfy linters
    limit_order_enabled = True
    min_notional_usdt = 6.0

    # Используем новое безопасное чтение конфига
    config = read_shared_config(config_path)
    if not config or "portfolios" not in config:
        logger.error(f"Aborting cycle: Invalid or missing config structure from {config_path}")
        return

    limit_order_enabled = config.get("limit_order_enabled", True)
    min_notional_usdt = config.get("min_notional_usdt", 6.0)

    # Paper mode: override > config
    paper_mode = paper_mode_override if paper_mode_override is not None else config.get("paper_mode", False)
    
    portfolio_cfg = config["portfolios"][0]
    targets = portfolio_cfg["targets"]
    global_threshold = portfolio_cfg["rebalance_threshold"]
    check_interval = portfolio_cfg["check_interval_sec"]
    
    # Приоритет тикера: override > config > default
    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
    
    # Ticker-specific threshold override
    ticker_thresholds = portfolio_cfg.get("ticker_thresholds", {})
    threshold = ticker_thresholds.get(base_ticker, global_threshold)
    
    logger.info(f"Using rebalance threshold: {threshold*100:.2f}% for {base_ticker}")
    
    siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
    reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
    
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

    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]
    siphoning_reserve = state.get("siphoning_reserve", 0.0)
    initial_tpv = state.get("initial_tpv", 0.0)
    reference_tpv = state.get("reference_tpv", 0.0)
    tpv_ath = state.get("tpv_ath", 0.0)
    cycles = state.get("rebalance_cycles", 0)

    # Параметры капитала и защиты
    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", portfolio_cfg.get("initial_capital", 0.0))
    
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
                prices = await connector.get_futures_prices([base_ticker])
                price = prices.get(base_ticker)
                if not price: raise Exception(f"Could not fetch {base_ticker} price")
                
                if paper_mode and paper_state["last_price"] > 0:
                    price_diff = price - paper_state["last_price"]
                    long_pnl = paper_state["positions"].get(f"{base_ticker}_LONG", 0.0) * price_diff
                    short_pnl = paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0) * (-price_diff)
                    paper_state["balance"] += (long_pnl + short_pnl)
                
                if paper_mode:
                    paper_state["last_price"] = price
                    real_equity = paper_state["balance"]
                    # Для Paper Mode имитируем структуру с ценами входа
                    raw_positions = paper_state.get("positions", {})
                    positions = {k: v for k, v in raw_positions.items()}
                    l_entry = paper_state.get("long_entry_price", price)
                    s_entry = paper_state.get("short_entry_price", price)
                else:
                    # В реальном режиме считаем активным капиталом всё, что за вычетом сейфа
                    total_free = await connector.get_free_balance()
                    real_equity = total_free - siphoning_reserve
                    raw_positions = await connector.get_positions()
                    # Извлекаем только QTY для калькулятора, цены входа передаем отдельно
                    positions = {k: v["qty"] for k, v in raw_positions.items()}
                    l_entry = raw_positions.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                    s_entry = raw_positions.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)

                if virt_basis_price == 0 or initial_tpv == 0:
                    if virt_basis_price == 0:
                        virt_basis_price = price
                        virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                    if initial_tpv == 0:
                        config_initial_cap = portfolio_cfg.get("initial_capital", real_equity)
                        temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                     base_ticker=base_ticker, siphoning_reserve=0.0, targets=targets,
                                                     long_entry_price=l_entry, short_entry_price=s_entry,
                                                     initial_capital=config_initial_cap)
                        initial_tpv = temp_calc.tpv
                        reference_tpv = initial_tpv
                        logger.info(f"Initialized TPV base: {initial_tpv:.2f} (from {'config' if 'initial_capital' in portfolio_cfg else 'current equity'})")

                    state.update({
                        "virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt,
                        "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                    })
                    await save_json(state_file_path, state)

                # Для калькулятора используем ЧИСТУЮ эквити (для расчета профита), 
                # а внутри калькулятора будет лимитированная tpv для ребаланса.
                calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                         base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                         long_entry_price=l_entry, short_entry_price=s_entry,
                                         initial_capital=initial_tpv)
                
                # Лимитированная версия эквити для логики стоп-лоссов и маржи
                limited_equity = min(real_equity, max_capital_usdt) if max_capital_usdt > 0 else real_equity
                
                if tpv_ath == 0 or calc.total_tpv > tpv_ath:
                    tpv_ath = calc.total_tpv
                    state["tpv_ath"] = tpv_ath
                    await save_json(state_file_path, state)

                if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                    drawdown_pct = (1 - calc.total_tpv / tpv_ath) * 100
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
                        m_info = await connector.get_margin_ratio()
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

                current_pos_sum = abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0))
                is_first_run = current_pos_sum == 0
                is_extreme = calc.share_long_pct > 100 or calc.share_short_pct > 100

                if i % 5 == 0:
                    res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                    logger.info(f"Heartbeat: Balance={real_equity:.2f}{res_str} | {base_ticker}={price:.6g} | L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%")

                current_threshold = -1.0 if (is_first_run or is_extreme) else threshold
                if not (is_first_run or is_extreme) and reference_tpv > 0 and calc.tpv < reference_tpv:
                    current_threshold *= 2.0

                # Логика ребалансировки
                actions = calc.calculate_deviations(targets, current_threshold, ignore_limits=(current_threshold < 0))
                
                if actions:
                    # Внедряем Notional Value Guard для физических ордеров
                    min_notional = portfolio_cfg.get("min_notional_usdt", 6.0)
                    valid_actions = [a for a in actions if a["type"] == "VIRTUAL_RESET" or abs(a.get("diff_usdt", 0)) >= min_notional]
                    
                    if not valid_actions:
                        if i % 10 == 0:
                            logger.info(f"Rebalance actions identified, but too small (<{min_notional} USDT). Skipping.")
                    else:
                        logger.info(f"Rebalance needed ({len(valid_actions)} actions). Shares: L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%")
                        
                        rebalance_msg = (
                            f"🔄 <b>Rebalance Starting</b>: <code>{base_ticker}</code>\n"
                            f"Current: L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%\n"
                            f"Target Actions: {len(valid_actions)}"
                        )
                        asyncio.create_task(notifier.send_message(rebalance_msg))

                        # Fetch step sizes for correct rounding
                        exchange_info = await connector.get_exchange_info()
                        step_sizes = {s["symbol"]: float(f["stepSize"]) for s in exchange_info["symbols"] for f in s["filters"] if f["filterType"] == "LOT_SIZE"}

                        # В начале цикла ребаланса фиксируем доступный излишек для сифонинга
                        current_initial_cap = portfolio_cfg.get("initial_capital", initial_tpv)
                        excess_to_siphon: float = max(0, (calc.total_tpv - siphoning_reserve) - current_initial_cap)

                        for action in valid_actions:
                            trade_pnl = 0.0 

                            if action["type"] == "VIRTUAL_RESET":
                                virt_basis_price = price
                                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                                state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})
                                await save_json(state_file_path, state)
                                logger.info(f"🔄 Virtual share rebalanced (Reset to {targets['VIRTUAL']['share']*100:.1f}%)")
                                continue

                            key = action["symbol"]
                            pos_side = key.split('_')[1]
                            diff_usdt = action["diff_usdt"]
                            order_qty = diff_usdt / price
                            side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
                            reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")
                            step_size = step_sizes.get(key.split('_')[0], 0.0)

                            if paper_mode:
                                qty_rounded = PortfolioExecutor(None).round_quantity(abs(order_qty), step_size)
                                if qty_rounded > 0:
                                    order_value = qty_rounded * price
                                    commission = order_value * 0.0004
                                    
                                    # PnL Calculation for paper mode
                                    pos_key = f"{base_ticker}_{pos_side}"
                                    old_qty = paper_state["positions"].get(pos_key, 0.0)
                                    entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                                    old_entry = paper_state.get(entry_key, price)

                                    if reduce_only:
                                        if pos_side == "LONG":
                                            trade_pnl = qty_rounded * (price - old_entry) - commission
                                        else:
                                            trade_pnl = qty_rounded * (old_entry - price) - commission
                                    
                                    new_qty = (old_qty + qty_rounded) if (side == "BUY" and pos_side == "LONG") or (side == "SELL" and pos_side == "SHORT") else (old_qty - qty_rounded)
                                    
                                    if not reduce_only:
                                        curr_q = abs(old_qty)
                                        paper_state[entry_key] = (curr_q * old_entry + qty_rounded * price) / (curr_q + qty_rounded) if (curr_q + qty_rounded) > 0 else price
                                    
                                    paper_state["positions"][pos_key] = new_qty
                                    paper_state["balance"] -= commission
                                    await save_json(paper_state_file_path, paper_state)
                                    
                                    trade_log = f"📝 PAPER: {side} {qty_rounded} {pos_key} @ {price:.6g}"
                                    if trade_pnl != 0: trade_log += f" | PnL: {trade_pnl:+.4f}"
                                    logger.info(trade_log)
                                    asyncio.create_task(notifier.send_message(f"<b>{trade_log}</b>"))
                            else:
                                # Real execution
                                logger.info(f"⚡ REAL: Executing {side} {abs(order_qty):.6g} {key}...")
                                executor = PortfolioExecutor(connector)
                                success = await executor.execute_rebalance(action, price, step_size, limit_order=limit_order_enabled)
                                if success:
                                    logger.info(f"✅ REAL: {side} {key} order filled.")
                                else:
                                    logger.warning(f"❌ REAL: {side} {key} order failed.")

                            if reduce_only and trade_pnl > 0 and excess_to_siphon > 0:
                                siphon_amount = min(trade_pnl * (1 - reinvestment_ratio), excess_to_siphon)
                                if siphon_amount > 0:
                                    siphoning_reserve += siphon_amount
                                    excess_to_siphon -= siphon_amount
                                    state["siphoning_reserve"] = siphoning_reserve
                                    await save_json(state_file_path, state)
                                    logger.info(f"💰 SIPHONED: +{siphon_amount:.4f} USDT (Reserve: {siphoning_reserve:.2f})")
                                    asyncio.create_task(notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT from {pos_side} {side}"))

                        cycles += 1
                        state["rebalance_cycles"] = cycles
                        await save_json(state_file_path, state)
                        
                        # Recalculate and notify completion
                        new_raw_positions = paper_state["positions"] if paper_mode else (await connector.get_positions())
                        new_positions = new_raw_positions if paper_mode else {k: v["qty"] for k, v in new_raw_positions.items()}
                        new_calc = PortfolioCalculator(new_positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                    base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                                    long_entry_price=l_entry, short_entry_price=s_entry,
                                                    initial_capital=initial_tpv)
                        
                        summary_msg = (
                            f"<b>✅ Rebalance #{cycles} Complete</b>: <code>{base_ticker}</code>\n"
                            f"New Shares: L:{new_calc.share_long_pct:.1f}% S:{new_calc.share_short_pct:.1f}% V:{new_calc.share_virt_pct:.1f}%\n"
                            f"TPV: <code>{new_calc.total_tpv:.2f} USDT</code>"
                        )
                        logger.info(f"Rebalance #{cycles} complete. TPV: {new_calc.total_tpv:.2f}")
                        asyncio.create_task(notifier.send_message(summary_msg))

                # Всегда обновляем стейт для агрегатора статусов в конце каждого цикла
                state.update({
                    "last_tpv": calc.total_tpv,
                    "last_profit": calc.total_tpv - initial_tpv,
                    "last_update": time.time(),
                    "rebalance_cycles": cycles
                })
                await save_json(state_file_path, state)

                if (i + status_offset) % 100 == 0:
                    logger.info(f"Heartbeat: TPV={calc.total_tpv:.2f} | PnL={calc.total_tpv - initial_tpv:+.2f} | Cycles={cycles}")

            except Exception as e:
                logger.error(f"Error in cycle: {e}")
                await asyncio.sleep(10)
            await asyncio.sleep(check_interval)
            i += 1
    finally:
        await notifier.close()

async def emergency_stop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None):
    config = await load_json(config_path, {})
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
    with open(args.config, "r", encoding="utf-8") as f: cfg = json.load(f)
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
