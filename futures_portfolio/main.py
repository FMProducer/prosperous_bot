import asyncio
import sys
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
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier
from storage import safe_load_json as load_json, safe_save_json as save_json, safe_load_json_sync

from pathlib import Path

# ProcessPoolExecutor removed to reduce latency

def sync_read_json(path: str) -> Dict:
    return safe_load_json_sync(path, {})


def emit_signal(signal_type: str, ticker: str, is_paper: bool) -> None:
    """Создает пустой файл-флаг для супервайзера."""
    sig_dir = Path("signals")
    sig_dir.mkdir(exist_ok=True)
    mode_tag = "paper" if is_paper else "real"
    sig_path = sig_dir / f"{signal_type}_{mode_tag}_{ticker}.flag"
    try:
        sig_path.touch(exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to emit signal {signal_type} for {ticker}: {e}")

def self_kill_pm2(ticker: str, is_paper: bool) -> None:
    """Удаляет себя из PM2 перед выходом, чтобы предотвратить autorestart."""
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    try:
        import subprocess
        import sys
        if sys.platform == "win32":
            cmd = f"pm2 delete {proc_name}"
            is_shell = True
        else:
            cmd = ["pm2", "delete", proc_name]
            is_shell = False

        subprocess.run(
            cmd,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=is_shell
        )
        logging.info(f"PM2 self-kill: deleted {proc_name}")
    except Exception as e:
        logging.warning(f"PM2 self-kill failed for {proc_name}: {e}")

async def _update_final_metrics_for_exit(state: dict, state_file_path: str, total_tpv_final: Decimal, initial_tpv: Decimal, safe_calc_res: dict, cycles: int, logger: logging.Logger) -> None:
    """Updates and saves the final profit metrics in the state file before a bot exits."""
    try:
        logger.info(f"🔄 Санитизация состояния перед выходом. Финальный TPV: {float(total_tpv_final):.4f}")

        # Полное ребазирование метрик под финальное значение TPV
        state.update({
            "last_tpv": float(total_tpv_final),
            "initial_tpv": float(total_tpv_final),
            "reference_tpv": float(total_tpv_final),
            "tpv_ath": float(total_tpv_final),
            "last_profit": float(total_tpv_final - initial_tpv),
            "total_pnl_pct": safe_calc_res.get("total_pnl_pct", 0.0),
            "last_update": time.time(),
            "rebalance_cycles": cycles,
            # Очистка триггеров скользящего стопа (но сохраняем сам факт срабатывания для супервайзера)
            "trailing_stop_violation_start": 0.0,
            "trailing_stop_paper_timeout_end": 0.0,
            "trailing_stop_triggered": True
        })

        await save_json(state_file_path, state)
        logger.info(f"✅ Final metrics saved before exit. Last Profit: {float(state['last_profit']):.2f}")
    except Exception as e:
        logger.error(f"Error saving final metrics before exit: {e}")

async def _handle_liquidation_recovery(connector, base_ticker, state, state_file_path,
                                        paper_state, paper_state_file_path,
                                        config_path, logger, notifier):
    """Handles liquidation: closes remaining position, stops bot, adds to blacklist."""
    try:
        logger.critical(f"🚨 LIQUIDATION RECOVERY for {base_ticker}: Closing remaining positions...")

        from decimal import Decimal
        executor = PortfolioExecutor(connector, base_ticker=base_ticker)

        # Get real positions
        real_pos_dict = await connector.get_position_risk()
        # Convert dict to list for uniform processing
        # Dict format: {"SYMBOL_LONG": {"liq_price": ..., "unrealized_pnl": ..., "positionAmt": ...}, ...}
        real_pos = []
        for key, val in real_pos_dict.items():
            entry = dict(val)
            entry["symbol"] = key.split("_")[0] if "_" in key else key
            entry["positionSide"] = key.split("_")[1] if "_" in key else "BOTH"
            entry["positionAmt"] = entry.get("qty", 0) or entry.get("positionAmt", 0)
            real_pos.append(entry)
        real_pos = [p for p in real_pos if base_ticker in p.get("symbol", "")]

        for p in real_pos:
            amt = float(p.get("positionAmt", 0))
            if abs(amt) < 1e-10:
                continue
            ps = p.get("positionSide", "")
            side = "SELL" if amt > 0 else "BUY"
            logger.critical(f"Closing {ps} position: {amt} @ market")

            try:
                symbol = p.get("symbol", base_ticker)
                await executor.execute_market_order(
                    symbol=symbol,
                    qty=Decimal(str(abs(amt))),
                    side=side,
                    step_size=Decimal("0.1"),
                    reduce_only=True,
                    position_side=ps if ps in ("LONG", "SHORT") else "BOTH",
                    min_notional=Decimal("0"),
                )
                logger.critical(f"✅ Closed {ps} position")
            except Exception as e:
                logger.error(f"Failed to close {ps}: {e}")

        # Update state
        state["positions"] = {}
        state["last_tpv"] = 0.0
        state["last_profit"] = -float(state.get("initial_tpv", 180.0))
        state["total_pnl_pct"] = -100.0
        state["trailing_stop_triggered"] = True
        await save_json(state_file_path, state)

        # Remove from live_swarm and add to toxic_blacklist
        try:
            import json as _json
            from pathlib import Path
            cfg_path = Path(config_path)
            if cfg_path.exists():
                cfg_data = _json.loads(cfg_path.read_text())
                live_swarm = cfg_data.get("live_swarm", [])
                if base_ticker in live_swarm:
                    live_swarm.remove(base_ticker)
                    cfg_data["live_swarm"] = live_swarm
                toxic = cfg_data.get("toxic_blacklist_real", {})
                cooldown_days = cfg_data.get("toxic_cooldown_days", 0.02)
                expiry = time.time() + cooldown_days * 86400
                toxic[base_ticker] = expiry
                cfg_data["toxic_blacklist_real"] = toxic
                cfg_path.write_text(_json.dumps(cfg_data, indent=2, ensure_ascii=False))
                logger.info(f"{base_ticker} removed from live_swarm, added to toxic_blacklist_real")
        except Exception as e:
            logger.error(f"Failed to update config: {e}")

        emit_signal("stop", base_ticker, is_paper=False)

        try:
            await notifier.send_alert("🚨 LIQUIDATION",
                f"{base_ticker}: Position liquidated! Bot stopped. Remaining positions closed.")
        except:
            pass

        logger.critical(f"🚨 LIQUIDATION RECOVERY complete for {base_ticker}")

    except Exception as e:
        logger.error(f"Liquidation recovery failed: {e}")


async def _handle_liquidation_guard(
    pos_key: str, dist: float, liq_price: float,
    liquidation_distance_warn: float, liquidation_distance_crit: float,
    is_paper: bool, raw_positions, paper_state, connector,
    base_ticker: str, step_sizes: dict, notifier, logger
):
    """
    Shared liquidation guard handler for both paper and real modes.
    Paper: simulates price-based close (adjusts shadow_state balance).
    Real: sends actual market close order via Binance.
    """
    if dist <= liquidation_distance_crit:
        logger.critical(
            f"🚨 LIQUIDATION CRITICAL [{'PAPER' if is_paper else 'REAL'}]: "
            f"{pos_key} distance {dist:.1f}% (liq_price={liq_price:.8f}). Emergency close!"
        )
        try:
            await notifier.send_alert(
                "🚨 LIQUIDATION IMMINENT",
                f"{pos_key}: distance {dist:.1f}% to liq @ {liq_price:.8f}. Closing!"
            )
        except Exception:
            pass

        if is_paper:
            # Paper mode: simulate closing BOTH sides at current price
            for _side in ("LONG", "SHORT"):
                _pos_key = f"{base_ticker}_{_side}"
                _entry_key = f"{_side.lower()}_entry_price"
                _old_entry = paper_state.get(_entry_key, 0.0)
                _old_qty = paper_state["positions"].get(_pos_key, 0.0)
                if _old_qty == 0:
                    continue
                # Calculate PnL from price movement
                if _side == "LONG":
                    _pnl = _old_qty * (liq_price - _old_entry)
                else:
                    _pnl = _old_qty * (_old_entry - liq_price)
                paper_state["balance"] = float(Decimal(str(paper_state["balance"])) + Decimal(str(_pnl)))
                paper_state["positions"][_pos_key] = 0.0
                paper_state[_entry_key] = 0.0
                paper_state[f"{_side.lower()}_liquidation_price"] = 0.0
                logger.info(f"✅ PAPER: Emergency closed {_pos_key} (PnL={_pnl:+.4f})")
            mode_tag = "PAPER"
        else:
            # Real mode: send actual close order for BOTH sides
            for _side in ("LONG", "SHORT"):
                _pos_key = f"{base_ticker}_{_side}"
                _close_side = "SELL" if _side == "LONG" else "BUY"
                _close_qty = abs(raw_positions.get(_pos_key, {}).get("qty", 0.0))
                if _close_qty > 0:
                    try:
                        await PortfolioExecutor(connector).execute_market_order(
                            symbol=base_ticker,
                            side=_close_side,
                            qty=_close_qty,
                            position_side=_side,
                            reduce_only=True
                        )
                        paper_state["positions"][_pos_key] = 0.0
                        logger.info(f"✅ Emergency closed {_pos_key} to avoid liquidation")
                    except Exception as close_err:
                        logger.error(f"Failed to emergency close {_pos_key}: {close_err}")
            mode_tag = "REAL"

        logger.info(
            f"✅ [{'PAPER' if is_paper else 'REAL'}] Emergency closed {pos_key} "
            f"(dist={dist:.1f}%, liq={liq_price:.8f})"
        )

        # After closing one side in critical zone → add to blacklist
        # This prevents the bot from continuing on a failing ticker
        try:
            import json as _json
            from pathlib import Path
            # Find config path from notifier or use default
            _cfg_path = Path("config.json")
            if _cfg_path.exists():
                cfg_data = _json.loads(_cfg_path.read_text())
                # Add to black_list (permanent) — ticker is too volatile for this strategy
                bl = cfg_data.get("black_list", [])
                if base_ticker not in bl:
                    bl.append(base_ticker)
                    cfg_data["black_list"] = bl
                # Also add to isolated toxic_blacklist with cooldown
                bl_key = "toxic_blacklist_paper" if is_paper else "toxic_blacklist_real"
                toxic = cfg_data.get(bl_key, {})
                cooldown_days = cfg_data.get("toxic_cooldown_days", 0.02)
                expiry = time.time() + cooldown_days * 86400
                toxic[base_ticker] = expiry
                cfg_data[bl_key] = toxic
                # Remove from live_swarm
                live_swarm = cfg_data.get("live_swarm", [])
                if base_ticker in live_swarm:
                    live_swarm.remove(base_ticker)
                    cfg_data["live_swarm"] = live_swarm
                _cfg_path.write_text(_json.dumps(cfg_data, indent=2, ensure_ascii=False))
                logger.critical(
                    f"🚫 {base_ticker} added to black_list + toxic_blacklist. "
                    f"Removed from live_swarm."
                )
        except Exception as e:
            logger.error(f"Failed to blacklist {base_ticker}: {e}")

    elif dist <= liquidation_distance_warn:
        logger.warning(
            f"⚠️ LIQUIDATION WARNING [{'PAPER' if is_paper else 'REAL'}]: "
            f"{pos_key} distance {dist:.1f}% (liq_price={liq_price:.8f})"
        )


async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None, paper_mode_override: bool = None):
    # [SSOT] Absolute Scope Safety - Initialize all variables at function start
    i = 0
    status_offset = random.randint(0, 99)
    target_initial = 0.0
    max_spread = 0.0015 # 0.15%
    max_velocity = 0.01 # 1.0%
    velocity_window = 60
    last_config_mtime = 0.0
    current_config = None
    
    # Обычное чтение конфига без блокировок
    config = safe_load_json_sync(config_path, {})
    if not config:
        logger.error(f"Aborting cycle: Failed to read config {config_path}")
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
    t_surplus = portfolio_cfg.get("rebalance_threshold_surplus", portfolio_cfg.get("rebalance_threshold", 0.02))
    t_deficit = portfolio_cfg.get("rebalance_threshold_deficit", t_surplus)

    check_interval = portfolio_cfg.get("check_interval_sec", 15)

    t_cfg = portfolio_cfg.get("ticker_thresholds", {}).get(base_ticker)
    if isinstance(t_cfg, dict):
        threshold_surplus = float(t_cfg.get("surplus", t_surplus))
        threshold_deficit = float(t_cfg.get("deficit", t_deficit))
    elif isinstance(t_cfg, (float, int)):
        threshold_surplus = threshold_deficit = float(t_cfg)
    else:
        threshold_surplus, threshold_deficit = float(t_surplus), float(t_deficit)

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
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "balance": target_initial_cap, # SSOT Balance
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,  # Фиксированная база для гистерезиса
        "tpv_ath": target_initial_cap,  # FIX: start ATH at initial_capital, not 0
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
        fallback_bal = float(state.get("initial_tpv", target_initial_cap))
        if fallback_bal <= 0: fallback_bal = target_initial_cap
        paper_state["balance"] = fallback_bal
        logger.warning(f"⚠️ 'balance' missing in {paper_state_file_path}. Initialized to {fallback_bal}")

    # Если тикер сменился, сбрасываем количество виртуальных монет и начальный TPV
    if state.get("base_ticker") != base_ticker:
        logger.info(f"Ticker in state changed from {state.get('base_ticker')} to {base_ticker}. Resetting virt_qty, initial TPV and ATH.")
        state["virt_qty"] = 0.0
        state["virt_debt"] = 0.0
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
                    # state['trailing_stop_triggered'] = False  # [RESTRICTION] Manual reset only by supervisor
                    state['trailing_stop_violation_start'] = 0.0
                    state['trailing_stop_paper_timeout_end'] = 0.0
                    logger.info(f"🔄 State reset (TS flag preserved): clean start detected (no open positions)")
                    await save_json(state_file_path, state)
        except Exception as e:
            logger.error(f"State Isolation Protocol failed: {e}")

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
    equity_trailing_stop_activation_pct = config.get("equity_trailing_stop_activation_pct", 0.0)
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

            # Set leverage from config (targets.BASE_LONG.leverage)
            try:
                _lev = int(targets.get("BASE_LONG", {}).get("leverage", 7))
                await connector.set_leverage(base_ticker, _lev)
                logger.info(f"Leverage set to {_lev}x for {base_ticker}")
            except Exception as e:
                logger.warning(f"Could not set leverage for {base_ticker}: {e}")

            try:
                await connector.set_margin_type(base_ticker, "CROSSED")
                logger.info(f"Margin Type set to CROSS for {base_ticker}")
            except Exception as e:
                logger.error(f"Could not set margin type for {base_ticker}: {e}")
                # CRITICAL: without correct margin type, liq_price from Binance is unreliable
                # Do NOT trade if margin type cannot be set
                asyncio.create_task(notifier.send_alert("STARTUP ERROR", f"Could not set margin type for {base_ticker}: {e}. Bot will NOT start."))
                return

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
    last_io_save = time.time()

    # Sanitize ATH on clean start (when supervisor has explicitly cleared the TS flag)
    if not state.get("trailing_stop_triggered", False):
        _pos_long = abs(paper_state["positions"].get(f"{base_ticker}_LONG", 0.0))
        _pos_short = abs(paper_state["positions"].get(f"{base_ticker}_SHORT", 0.0))
        if _pos_long < 1e-10 and _pos_short < 1e-10:
            _initial = Decimal(str(state.get("initial_tpv", target_initial_cap)))
            _current_ath = Decimal(str(state.get("tpv_ath", 0.0)))
            if _current_ath > _initial:
                logger.info(f"🔄 Clean start detected. Resetting tpv_ath from {_current_ath} to {_initial} to enable TS activation.")
                state["tpv_ath"] = float(_initial)
                state["trailing_stop_violation_start"] = 0.0
                # Synchronous save before entering the main loop
                await save_json(state_file_path, state)

    try:
        while True:
            state_dirty = False
            paper_state_dirty = False
            any_success = False
            try:
                # Failsafe: if restarted by PM2 with triggered state, exit immediately to trigger Reaper
                if state.get("trailing_stop_triggered", False):
                    logger.critical(f"🚨 Trailing Stop already triggered for {base_ticker}. Entering Zombie Mode.")
                    self_kill_pm2(base_ticker, paper_mode)
                    while True:
                        await asyncio.sleep(86400)

                # Dynamic config reload
                try:
                    current_mtime = os.path.getmtime(config_path)
                    if current_mtime != last_config_mtime:
                        # Unblock the Event Loop only if file changed
                        current_config = await asyncio.to_thread(sync_read_json, config_path)
                        last_config_mtime = current_mtime
                        portfolio_cfg = current_config["portfolios"][0]
                        targets = portfolio_cfg["targets"]
                        t_surplus = portfolio_cfg.get("rebalance_threshold_surplus", portfolio_cfg.get("rebalance_threshold", 0.02))
                        t_deficit = portfolio_cfg.get("rebalance_threshold_deficit", t_surplus)
                        check_interval = portfolio_cfg.get("check_interval_sec", 15)

                        t_cfg = portfolio_cfg.get("ticker_thresholds", {}).get(base_ticker)
                        if isinstance(t_cfg, dict):
                            threshold_surplus = float(t_cfg.get("surplus", t_surplus))
                            threshold_deficit = float(t_cfg.get("deficit", t_deficit))
                        elif isinstance(t_cfg, (float, int)):
                            threshold_surplus = threshold_deficit = float(t_cfg)
                        else:
                            threshold_surplus, threshold_deficit = float(t_surplus), float(t_deficit)

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
                        max_spread = float(guards_cfg.get("max_spread_pct", 0.15)) / 100.0
                        max_velocity = float(guards_cfg.get("max_price_velocity_pct", 1.0)) / 100.0
                        velocity_window = int(guards_cfg.get("velocity_window_sec", 60))
                        net_move_block_pct = float(guards_cfg.get("net_move_block_pct", 1.5)) / 100.0
                        net_move_window = int(guards_cfg.get("net_move_window_sec", 30))

                        logger.debug(f"⚙️ Config reloaded. Active Thresholds for {base_ticker}: Surplus {threshold_surplus*100:.2f}%, Deficit {threshold_deficit*100:.2f}%, Max Spread: {max_spread*100:.2f}%")

                    if i % 60 == 0:
                        logger.debug(f"Thresholds running at: Surplus {threshold_surplus*100:.2f}%, Deficit {threshold_deficit*100:.2f}%")

                except Exception as e:
                    logger.error(f"Error reloading config: {e}. Using previous values.")

                # Dynamic initial_tpv update from config
                # Для тикеров в toxic_blacklist ребазирование зависит от прибыльности:
                #   убыток (last_profit < 0) → ребаз на initial_capital из конфига
                #   прибыль (last_profit >= 0) → ребаз на last_tpv (сохраняет прибыль)
                if current_config is None:
                    current_config = await asyncio.to_thread(sync_read_json, config_path)
                _bl_key = "toxic_blacklist_paper" if paper_mode else "toxic_blacklist_real"
                _bl = current_config.get(_bl_key, {})
                _now = time.time()
                _in_blacklist = base_ticker in _bl and _bl.get(base_ticker, 0) > _now

                if _in_blacklist:
                    _last_profit = float(state.get("last_profit", 0.0))
                    if _last_profit < 0:
                        # Убыточный бот → полный сброс на начальный капитал из конфига
                        _rebase_target = target_initial
                        logger.info(f"🔄 Blacklist rebase (LOSS): {base_ticker} initial_tpv {initial_tpv} -> {_rebase_target}")
                    else:
                        # Прибыльный бот → сохраняем накопленную прибыль
                        _rebase_target = float(state.get("last_tpv", target_initial))
                        logger.info(f"🔄 Blacklist rebase (PROFIT): {base_ticker} initial_tpv {initial_tpv} -> {_rebase_target} (last_tpv)")
                    initial_tpv = _rebase_target
                    reference_tpv = initial_tpv
                    state["initial_tpv"] = initial_tpv
                    state["reference_tpv"] = reference_tpv
                    state_dirty = True
                elif initial_tpv != target_initial and target_initial > 0:
                    logger.info(f"🔄 Initial Capital changed in config: {initial_tpv} -> {target_initial}. Updating base.")
                    initial_tpv = target_initial
                    reference_tpv = initial_tpv
                    state["initial_tpv"] = initial_tpv
                    state["reference_tpv"] = reference_tpv
                    state_dirty = True

                # Use Mark Price for TPV and rebalance triggers as recommended by Audit
                prices = await connector.get_mark_prices([base_ticker])
                price = prices.get(base_ticker)
                if not price: raise Exception(f"Could not fetch {base_ticker} mark price")
                
                # -------------------------------------------------------------------------
                # [SAFETY] VELOCITY & TREND GUARD
                # -------------------------------------------------------------------------
                now = time.time()
                price_history.append((now, price))
                # Увеличиваем окно истории до 300 секунд для анализа тренда
                while price_history and (now - price_history[0][0]) > 300:
                    price_history.popleft()
                
                if len(price_history) > 1:
                    # 1. VELOCITY GUARD (% за окно 60с)
                    v_point = next((p for p in price_history if now - p[0] <= velocity_window), price_history[0])
                    old_t, old_p = v_point
                    velocity = abs(price - old_p) / old_p
                    if velocity > max_velocity:
                        if i % 5 == 0:
                            logger.warning(f"🚀 Velocity Guard: {base_ticker} is moving too fast ({velocity*100:.2f}% in {int(now-old_t)}s). Blocking trades.")
                        await asyncio.sleep(check_interval); i += 1; continue

                    # 2. TREND GUARD (Efficiency Filter)
                    if len(price_history) > 10:
                        p_list = [p[1] for p in price_history]
                        net_move = abs(p_list[-1] - p_list[0])
                        total_path = sum(abs(p_list[j] - p_list[j-1]) for j in range(1, len(p_list)))
                        trend_eff = (net_move / total_path) if total_path > 0 else 0
                        
                        trend_min_move = guards_cfg.get("trend_min_move_pct", 0.7) / 100.0
                        trend_eff_thresh = guards_cfg.get("trend_eff_threshold", 0.35)
                        
                        # Trend Guard: если прошли > min_move% и эффективность > threshold — это тренде
                        if (net_move / p_list[0] > trend_min_move) and trend_eff > trend_eff_thresh:
                            if i % 5 == 0:
                                logger.warning(f"🚫 Trend Guard: {base_ticker} toxic move (Eff: {trend_eff:.2f}, Move: {net_move/p_list[0]*100:.2f}%). Freezing.")
                            await asyncio.sleep(check_interval); i += 1; continue

                # -------------------------------------------------------------------------
                # [SAFETY] NET MOVE GUARD (NMG) — Pump/Dump Protection
                # -------------------------------------------------------------------------
                # If price moved > threshold in one direction within short window,
                # block ALL rebalance actions. Prevents closing positions at fake
                # profit/loss during fast unidirectional moves.
                nmg_triggered = False
                if len(price_history) > 1:
                    nmg_point = next((p for p in price_history if now - p[0] <= net_move_window), None)
                    if nmg_point:
                        nmg_old_t, nmg_old_p = nmg_point
                        net_move = abs(price - nmg_old_p) / nmg_old_p
                        if net_move > net_move_block_pct:
                            nmg_triggered = True
                            if i % 5 == 0:
                                logger.warning(f"🛡️ Net Move Guard: {base_ticker} moved {net_move*100:.2f}% in {int(now-nmg_old_t)}s (limit {net_move_block_pct*100:.1f}%). Blocking ALL actions.")
                            await asyncio.sleep(check_interval); i += 1; continue

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

                # -------------------------------------------------------------------------
                # [SAFETY] GUARD #4: LIQUIDATION DISTANCE MONITOR
                # -------------------------------------------------------------------------
                # Checks how close each position is to its liquidation price.
                # If distance < threshold → emergency close that specific position.
                # Works for BOTH paper (simulated liq price) and real (exchange liq price).
                liquidation_distance_warn = portfolio_cfg.get("liquidation_distance_warn_pct", 15.0)
                liquidation_distance_crit = portfolio_cfg.get("liquidation_distance_crit_pct", 8.0)

                if not paper_mode and raw_positions:
                    # REAL mode + CROSSED margin: Binance returns unreliable liq_price
                    # Skip exchange liq check — use margin ratio (m_info) instead
                    # Liquidation risk is managed by max_drawdown_limit + trailing stop
                    pass

                elif paper_mode:
                    # PAPER mode: use simulated liquidation price from shadow_state
                    # Recalculate liq_price every cycle (cross-margin aware)
                    try:
                        _free_margin_val = Decimal(str(portfolio_cfg.get("paper_account_free_margin", 0.0)))
                        _mmr_val = Decimal('0.004')
                        _lv_l = Decimal(str(targets.get("BASE_LONG", {}).get("leverage", 7)))
                        _lv_s = Decimal(str(targets.get("BASE_SHORT", {}).get("leverage", 7)))
                        
                        for pos_side in ("LONG", "SHORT"):
                            pos_key = f"{base_ticker}_{pos_side}"
                            qty = abs(paper_state["positions"].get(pos_key, 0.0))
                            if qty == 0:
                                continue
                            
                            # Recalculate liq price using cross-margin formula
                            entry_key = f"{pos_side.lower()}_entry_price"
                            entry = Decimal(str(paper_state.get(entry_key, price)))
                            _lv = _lv_l if pos_side == "LONG" else _lv_s
                            _qty = Decimal(str(qty)) + Decimal('1e-10')
                            
                            liq_price_key = f"{pos_side.lower()}_liquidation_price"
                            if pos_side == "LONG":
                                _iso = entry * (Decimal('1') - Decimal('1') / _lv + _mmr_val)
                                _cross = max(_iso - _free_margin_val / _qty, Decimal('0'))
                            else:
                                _iso = entry * (Decimal('1') + Decimal('1') / _lv - _mmr_val)
                                _cross = _iso + _free_margin_val / _qty
                            
                            liq_price = float(_cross.quantize(Decimal('1e-8')))
                            paper_state[liq_price_key] = liq_price
                            paper_state_dirty = True
                            
                            if liq_price <= 0:
                                continue
                            # Calculate distance from current price to simulated liq price
                            if pos_side == "LONG":
                                dist = (price - liq_price) / price * 100.0 if price > 0 else 100.0
                            else:
                                dist = (liq_price - price) / price * 100.0 if price > 0 else 100.0
                            dist = max(dist, 0.0)

                            await _handle_liquidation_guard(
                                pos_key=pos_key, dist=dist, liq_price=liq_price,
                                liquidation_distance_warn=liquidation_distance_warn,
                                liquidation_distance_crit=liquidation_distance_crit,
                                is_paper=True, raw_positions=None,
                                paper_state=paper_state, connector=None,
                                base_ticker=base_ticker, step_sizes=step_sizes,
                                notifier=notifier, logger=logger
                            )
                            if dist <= liquidation_distance_crit:
                                paper_state_dirty = True
                                # Stop bot after liquidation critical — ticker is blacklisted
                                emit_signal("stop", base_ticker, is_paper=True)
                                logger.critical(f"🛑 Stopping {base_ticker} after liquidation critical (PAPER). Ticker blacklisted.")
                    except Exception as e:
                        logger.error(f"Paper liquidation guard check failed: {e}")

                # Cross-margin free margin check for paper mode
                # Paper bots share a real account — ensure enough free margin exists
                if paper_mode:
                    try:
                        account_free_margin = portfolio_cfg.get("paper_account_free_margin", 0.0)
                        min_free_pct = portfolio_cfg.get("paper_min_free_margin_pct", 15.0)
                        if account_free_margin > 0:
                            # Each paper bot needs: notional / leverage margin per side
                            side_margin = initial_tpv / 2  # half_capital as margin needed per side
                            total_margin_needed = side_margin * 2  # both sides
                            # margin_ratio = free margin / margin needed for this bot
                            margin_ratio = account_free_margin / total_margin_needed if total_margin_needed > 0 else 999
                            min_ratio = min_free_pct / 100.0
                            if margin_ratio < min_ratio:
                                logger.warning(
                                    f"⚠️ CROSS MARGIN [PAPER]: Free margin ${account_free_margin:.2f} "
                                    f"insufficient for {base_ticker} (need ${total_margin_needed:.2f}, "
                                    f"ratio={margin_ratio:.2f}x < min {min_ratio:.2f}x). "
                                    f"Skipping rebalance."
                                )
                                # Skip this cycle — don't rebalance when margin is thin
                                continue
                    except Exception as e:
                        logger.error(f"Paper cross-margin check failed: {e}")

                # Calculate isolated PnL and Equity
                # [SSOT] TPV = Cash + Virtual Value. Cash is paper_state["balance"].
                # In calculator.py, real_equity is treated as Wallet Balance.
                positions = paper_state["positions"] if paper_mode else {k: v for k, v in ticker_positions.items()}

                if virt_qty == 0 or initial_tpv == 0:
                    if initial_tpv == 0:
                        # For shared accounts, we MUST use assigned initial_capital as base
                        initial_tpv = portfolio_cfg.get("initial_capital", paper_state["balance"])
                        reference_tpv = initial_tpv
                        logger.info(f"Initialized TPV base: {initial_tpv:.2f} (Isolated Shadow Balance)")

                    if virt_qty == 0:
                        target_v_share = Decimal(str(targets["VIRTUAL"]["share"]))
                        if target_v_share <= 0:
                            # VIRTUAL is disabled (share=0) — skip initialization spam
                            virt_qty = 0.0
                            state["virt_debt"] = 0.0
                        else:
                            # КОРРЕКТНАЯ ИНИЦИАЛИЗАЦИЯ:
                            # 1. Считаем сколько монет купить на целевую долю
                            initial_cap = Decimal(str(initial_tpv))
                            dec_price = Decimal(str(price))

                            virt_qty_dec = (target_v_share * initial_cap) / dec_price
                            virt_cost = float(virt_qty_dec * dec_price)

                            # 2. Фиксируем ДОЛГ виртуальной ноги (стоимость покупки)
                            # Мы НЕ вычитаем это из balance в paper_state, так как balance
                            # представляет полный Wallet Balance (как на бирже).
                            state["virt_debt"] = state.get("virt_debt", 0.0) + virt_cost
                            state_dirty = True

                            # 3. Фиксируем количество
                            virt_qty = float(virt_qty_dec)

                            logger.info(f"🚀 Initialized Virtual: {virt_qty} units (Cost: {virt_cost:.2f} USDT added to virt_debt)")

                    state.update({
                        "virt_qty": virt_qty,
                        "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                        "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                    })
                    state_dirty = True

                # [SSOT RECONCILIATION]
                # In both modes, we use the shadow balance (paper_state["balance"]) for TPV calculation.
                # This ensures that adding/removing funds from the Binance wallet does not affect the bot's TPV.
                virt_debt = Decimal(str(state.get("virt_debt", 0.0)))
                real_equity = float(Decimal(str(paper_state["balance"])) - virt_debt)

                if real_equity < 0:
                    real_equity = 0.0

                if not paper_mode:
                    # In REAL mode, we still fetch wallet_balance for margin safety checks, 
                    # but we NO LONGER overwrite paper_state["balance"] with it.
                    wallet_balance = float(m_info.get("total_wallet_balance", 0.0))

                # Direct synchronous call to PortfolioCalculator to reduce latency
                ignore_limits = (abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0)) == 0)
                
                # Fetch min_notional once to reuse
                active_min_notional = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt", 6.0))

                calc = PortfolioCalculator(
                    positions=positions,
                    spot_price=price,
                    real_equity=real_equity,
                    virt_qty=virt_qty,
                    virt_debt=float(virt_debt),
                    base_ticker=base_ticker,
                    siphoning_reserve=siphoning_reserve,
                    targets=targets,
                    initial_capital=initial_tpv,
                    long_entry_price=l_entry,
                    short_entry_price=s_entry,
                    last_rebalance_price=state.get("last_rebalance_price", 0.0),
                    min_notional=active_min_notional
                )
                # PnL Guard removed — in 50/50 hedge, one leg is always negative when price moves
                # Blocking surplus sell prevents profit-taking during normal hedge operation
                allow_surplus_sell = True

                calc_res = calc.calculate_rebalance(targets, threshold_surplus, threshold_deficit, calc.tpv, ignore_limits, allow_surplus_sell)
                
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
                        emit_signal("stop", base_ticker, paper_mode)
                        logger.info(f"Sent STOP signal. Total Equity {tpv_total:.2f} < Global Initial {global_initial:.2f}. Ticker blacklisted.")
                    else:
                        emit_signal("exit", base_ticker, paper_mode)
                        logger.info(f"Sent EXIT signal. Total Equity {tpv_total:.2f} >= Global Initial {global_initial:.2f}. Ticker goes to probation.")
                        
                        # Устанавливаем таймаут пробации для прибыльного Emergency Stop
                        probation_days = current_config.get("probation_period_days", 0.041)
                        state["trailing_stop_paper_timeout_end"] = now + probation_days * 86400
                        state["trailing_stop_triggered"] = True

                    # --- CRITICAL: Update final metrics BEFORE emergency stop and exit ---
                    await _update_final_metrics_for_exit(state, state_file_path, Decimal(str(tpv_total)), Decimal(str(initial_tpv)), calc_res, cycles, logger)
                    # --------------------------------------------------------------------
                    await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker, paper_mode=paper_mode, close_only=True)
                    self_kill_pm2(base_ticker, paper_mode)
                    while True:
                        await asyncio.sleep(86400)

                if tpv_ath == 0 or tpv_total > tpv_ath:
                    tpv_ath = tpv_total
                    state["tpv_ath"] = tpv_ath
                    state_dirty = True

                if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                    # Trailing stop activates only when ATH exceeds activation threshold
                    activation_threshold = initial_tpv * (1 + equity_trailing_stop_activation_pct / 100)
                    if tpv_ath < activation_threshold:
                        # Not yet activated — reset violation and skip
                        if state.get("trailing_stop_violation_start", 0.0) > 0:
                            state["trailing_stop_violation_start"] = 0.0
                            state_dirty = True
                    else:
                        drawdown_pct = (1 - tpv_total / tpv_ath) * 100
                        if drawdown_pct >= equity_trailing_stop_pct:
                            violation_start = state.get("trailing_stop_violation_start", 0.0)
                            if violation_start == 0:
                                violation_start = now
                                state["trailing_stop_violation_start"] = violation_start
                                state_dirty = True
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
                                        await PortfolioExecutor(connector).execute_market_order(
                                            symbol=pos_key.split('_')[0],
                                            qty=Decimal(str(abs(qty))),
                                            side=side,
                                            step_size=Decimal(str(step_size)),
                                            reduce_only=True,
                                            position_side=pos_key.split('_')[1] if '_' in pos_key else "BOTH",
                                            min_notional=Decimal('0')
                                        )

                                # Realize state in shadow balance
                                paper_state["long_entry_price"] = 0.0
                                paper_state["short_entry_price"] = 0.0
                                await save_json(paper_state_file_path, paper_state)

                                # Жестко фиксируем смерть по трейлингу для HEAL Guard
                                state["trailing_stop_triggered"] = True

                                # --- CRITICAL: Update final metrics BEFORE state resets and exit ---
                                await _update_final_metrics_for_exit(state, state_file_path, Decimal(str(tpv_total)), Decimal(str(initial_tpv)), calc_res, cycles, logger)
                                # --------------------------------------------------------------------

                                # Set paper probation timeout (from probation_period_days)
                                probation_days = current_config.get("probation_period_days", 0.041)
                                timeout_end = now + probation_days * 86400
                                # Trailing stop timeout end should be set AFTER _update_final_metrics_for_exit if we want it to persist,
                                # but _update_final_metrics_for_exit clears it.
                                # Actually, requirements say: "Fully reset trailing stop tracking fields ... to 0.0/False to ensure clean state generation."
                                # So supervisor should handle the timeout if needed.
                                logger.info(f"Post-stop paper probation end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeout_end))}")

                                # Эмитируем сигнал остановки для супервайзера
                                global_initial = portfolio_cfg.get("initial_capital", 60.0)
                                if tpv_total < global_initial:
                                    emit_signal("stop", base_ticker, paper_mode)
                                    logger.info(f"Sent STOP signal. Total Equity {tpv_total:.2f} < Global Initial {global_initial:.2f}. Ticker blacklisted.")
                                else:
                                    # Trailing stop triggered with PROFIT
                                    # Delegate toxic_blacklist logic entirely to supervisor via IPC signal
                                    emit_signal("exit", base_ticker, paper_mode)
                                    logger.info(f"Sent EXIT signal. Total Equity {tpv_total:.2f} >= Global Initial {global_initial:.2f}.")

                                # --- CRITICAL: Update final metrics BEFORE state resets and exit ---
                                await _update_final_metrics_for_exit(state, state_file_path, Decimal(str(tpv_total)), Decimal(str(initial_tpv)), calc_res, cycles, logger)
                                # --------------------------------------------------------------------

                                # TERMINAL ACTION: Execute emergency liquidation immediately to prevent race condition with supervisor
                                logger.critical(f"Initiating synchronous emergency liquidation for {base_ticker}...")
                                # Close positions on exchange, preserve internal state for supervisor review
                                await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker, paper_mode=paper_mode, close_only=True)
                                self_kill_pm2(base_ticker, paper_mode)
                                while True:
                                    await asyncio.sleep(86400)
                            else:
                                # Timeout not yet reached — still pending
                                if i % 5 == 0:
                                    logger.info(f"Trailing Stop Pending: {drawdown_pct:.2f}% (Wait {equity_trailing_stop_timeout_sec - elapsed:.1f}s more)")
                        else:
                            # Drawdown back below threshold — NO RECOVERY
                            # Trailing stop is ONE-SHOT: once triggered, never resets
                            if state.get("trailing_stop_violation_start", 0.0) > 0:
                                logger.info(f"Trailing Stop: drawdown {drawdown_pct:.2f}% but already triggered — NO recovery")

                # Check if trailing stop was previously triggered (one-shot)
                if state.get("trailing_stop_triggered", False):
                    logger.critical(f"🚨 Trailing Stop already triggered for {base_ticker}. Entering Zombie Mode.")
                    self_kill_pm2(base_ticker, paper_mode)
                    while True:
                        await asyncio.sleep(86400)

                if not paper_mode and (margin_warning > 0 or margin_critical > 0):
                    try:
                        m_ratio = m_info.get("margin_ratio", 0.0)
                        if m_ratio > 0:
                            if m_ratio < portfolio_cfg.get("margin_ratio_critical", 2.0):
                                msg = f"Margin ratio {m_ratio:.2f} < {portfolio_cfg.get('margin_ratio_critical', 2.0)}. Emergency stop!"
                                logger.error(msg)
                                asyncio.create_task(notifier.send_alert("CRITICAL MARGIN", msg))
                                emit_signal("stop", base_ticker, paper_mode)
                                # --- CRITICAL: Update final metrics BEFORE emergency stop and exit ---
                                await _update_final_metrics_for_exit(state, state_file_path, Decimal(str(tpv_total)), Decimal(str(initial_tpv)), calc_res, cycles, logger)
                                # --------------------------------------------------------------------
                                await emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=base_ticker, paper_mode=paper_mode, close_only=True)
                                while True:
                                    await asyncio.sleep(86400)
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
                    
                    # Values for absolute breakdown
                    v_l, v_s, v_v, v_c = calc_res['val_long'], calc_res['val_short'], calc_res['val_virt'], calc_res['val_cash']
                    total_pnl = calc_res.get("total_pnl", tpv_total - initial_tpv)

                    l_target = Decimal(str(targets['BASE_LONG']['share']))
                    s_target = Decimal(str(targets['BASE_SHORT']['share']))
                    v_target = Decimal(str(targets['VIRTUAL']['share']))

                    h_msg = (
                        f"Heartbeat: TPV={tpv_total:.2f}{res_str} | PnL={total_pnl:+.2f} | {base_ticker}={price:.6g} | "
                        f"L:{l_p:.1f}% [{get_dev(Decimal(str(l_p))/100, l_target):+.1f}%] {{{v_l:+.2f}$}} | "
                        f"S:{s_p:.1f}% [{get_dev(Decimal(str(s_p))/100, s_target):+.1f}%] {{{v_s:+.2f}$}} | "
                        f"V:{v_p:.1f}% [{get_dev(Decimal(str(v_p))/100, v_target):+.1f}%] {{{v_v:+.2f}$}} | "
                        f"C:{c_p:.1f}% {{{v_c:.2f}$}}"
                    )
                    logger.info(h_msg)
                
                # Логика ребалансировки
                valid_actions = []
                fused_actions = []
                if actions:
                    # Внедряем Notional Value Guard для ВСЕХ ордеров
                    # Проверяем и в портфеле, и в глобальном конфиге
                    min_notional = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt", 6.0))
                    valid_actions = [a for a in actions if abs(a.get("diff_usdt", 0)) >= min_notional]
                    
                    fused_actions = valid_actions
                    
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

                            active_thresh = threshold_surplus if dev > 0 else threshold_deficit
                            trigger_key = key.replace("BASE_", "")
                            logger.info(f"Rebalance triggered: {trigger_key} deviation {dev:+.2f}% exceeds limit {active_thresh*100:.2f}%")

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
                                # -------------------------------------------------------------------------
                                # [V3.8.3] LOGGING & ACCOUNTING FOR VIRTUAL SPOT POSITION (1x Leverage)
                                # -------------------------------------------------------------------------
                                if res.get("type") == "VIRTUAL_ORDER":
                                    diff_usdt = Decimal(str(res.get("diff_usdt", 0.0)))
                                    dec_price = Decimal(str(price))
                                    virt_qty_before = Decimal(str(virt_qty))

                                    # 1. Списание/начисление кэша (Cash Accounting)
                                    current_balance = Decimal(str(paper_state["balance"]))
                                    paper_state["balance"] = float((current_balance - diff_usdt).quantize(Decimal('1e-4')))

                                    # Фиксация затрат (долга) для Real-режима
                                    state["virt_debt"] = float((Decimal(str(state.get("virt_debt", 0.0))) + diff_usdt).quantize(Decimal('1e-4')))
                                    paper_state_dirty = True
                                    state_dirty = True

                                    # 2. Пересчет количества монет по цене "сделки"
                                    # Дельта объема в чистых контрактах (монетах) базового актива
                                    v_delta_qty = diff_usdt / dec_price
                                    new_v_qty = virt_qty_before + v_delta_qty

                                    # 3. Value-based Dust Guard: если позиция меньше 1.0 USDT — в ноль
                                    if abs(new_v_qty * dec_price) < Decimal('1.0'):
                                        current_balance = Decimal(str(paper_state["balance"]))
                                        paper_state["balance"] = float((current_balance + new_v_qty * dec_price).quantize(Decimal('1e-4')))
                                        v_delta_qty = -new_v_qty # Фиксируем закрытие остатка
                                        new_v_qty = Decimal('0')
                                        logger.info(f"🧹 Dust Guard: Virtual Spot position liquidated (value < 1.0 USDT)")

                                    virt_qty = float(new_v_qty.quantize(Decimal('1e-8')))
                                    state["virt_qty"] = virt_qty
                                    state_dirty = True

                                    # 4. ПОЛНОЕ ИНФОРМАТИВНОЕ ЛОГИРОВАНИЕ ДЛЯ ПОЛЬЗОВАТЕЛЯ
                                    v_side = "BUY" if diff_usdt > 0 else "SELL"
                                    mode_tag = "PAPER" if paper_mode else "REAL"
                                    v_notional = virt_qty * float(dec_price)

                                    # Выводим строгий лог, идентичный реальной бирже
                                    v_trade_log = (
                                        f"📝 {mode_tag}_VIRTUAL: {v_side} {abs(float(v_delta_qty)):.4f} {base_ticker} @ {float(dec_price):.6g} "
                                        f"| Flow: {float(diff_usdt):+.2f} USDT "
                                        f"| Total Held: {virt_qty:.4f} {base_ticker.split('USDT')[0]} ({v_notional:.2f} USDT)"
                                    )
                                    logger.info(v_trade_log)
                                    continue

                                key = res.get("symbol", "UNKNOWN")
                                side = res.get("side", "UNKNOWN")

                                qty = Decimal(str(res.get("qty", 0.0)))
                                if qty <= 0:
                                    logger.warning(f"⚠️ Skip state update for {key} because executed qty is {qty}")
                                    continue

                                trade_pnl = Decimal(str(res.get("trade_pnl", 0.0)))
                                commission = Decimal(str(res.get("commission", 0.0)))
                                reduce_only = res.get("reduce_only", False)

                                pos_key = f"{base_ticker}_{pos_side}"
                                old_qty = Decimal(str(paper_state["positions"].get(pos_key, 0.0)))
                                entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                                old_entry = Decimal(str(paper_state.get(entry_key, price)))
                                if old_entry <= 0: old_entry = Decimal(str(price))

                                # Математически точный расчет изменения позиции
                                if (side == "BUY" and pos_side == "LONG") or (side == "SELL" and pos_side == "SHORT"):
                                    new_qty = old_qty + qty
                                else:
                                    new_qty = old_qty - qty

                                # Расчет средней цены входа (только при увеличении позиции)
                                if not reduce_only and (old_qty + qty) > 0:
                                    dec_price = Decimal(str(res.get("price", price)))
                                    new_entry = (old_qty * old_entry + qty * dec_price) / (old_qty + qty)
                                    paper_state[entry_key] = float(new_entry.quantize(Decimal('1e-8')))

                                    # [LIQUIDATION GUARD] Calculate simulated liquidation price
                                    # For LONG: liq_price < entry (price must drop to liq)
                                    # For SHORT: liq_price > entry (price must rise to liq)
                                    # Isolated margin formula: liq = entry × (1 ± 1/leverage ∓ mmr)
                                    # Cross-margin: free margin pushes liq further away (safer)
                                    _leverage = Decimal(str(targets.get(f"BASE_{pos_side}", {}).get("leverage", 7)))
                                    _mmr = Decimal('0.004')
                                    _free_margin = Decimal(str(portfolio_cfg.get("paper_account_free_margin", 0.0)))
                                    _pos_qty = Decimal(str(abs(paper_state["positions"].get(pos_key, 0.0)))) + Decimal('1e-10')
                                    
                                    if pos_side == "LONG":
                                        # LONG liq = entry × (1 - 1/leverage + mmr) — below entry
                                        _iso_liq = new_entry * (Decimal('1') - Decimal('1') / _leverage + _mmr)
                                        # Free margin pushes liq DOWN (further from current price = safer)
                                        _cross_liq = max(_iso_liq - _free_margin / _pos_qty, Decimal('0'))
                                        paper_state["long_liquidation_price"] = float(_cross_liq.quantize(Decimal('1e-8')))
                                    else:
                                        # SHORT liq = entry × (1 + 1/leverage - mmr) — above entry
                                        _iso_liq = new_entry * (Decimal('1') + Decimal('1') / _leverage - _mmr)
                                        # Free margin pushes liq UP (further from current price = safer)
                                        _cross_liq = _iso_liq + _free_margin / _pos_qty
                                        paper_state["short_liquidation_price"] = float(_cross_liq.quantize(Decimal('1e-8')))

                                elif reduce_only and new_qty == 0:
                                    # Если позиция закрыта полностью, сбрасываем цену входа
                                    paper_state[entry_key] = 0.0
                                    paper_state["long_liquidation_price"] = 0.0
                                    paper_state["short_liquidation_price"] = 0.0

                                # Запись обратно во float-структуру JSON
                                paper_state["positions"][pos_key] = float(new_qty.quantize(Decimal('1e-8')))

                                current_balance = Decimal(str(paper_state["balance"]))
                                paper_state["balance"] = float((current_balance + trade_pnl - commission).quantize(Decimal('1e-4')))
                                paper_state_dirty = True

                                mode_tag = "PAPER" if paper_mode else "REAL"
                                trade_log = f"📝 {mode_tag}: {side} {float(qty)} {pos_key} @ {price:.6g}"
                                if trade_pnl != 0: trade_log += f" | PnL: {float(trade_pnl):+.4f}"
                                logger.info(trade_log)
                                # asyncio.create_task(notifier.send_message(f"<b>{trade_log}</b>"))
                            else:
                                logger.warning(f"❌ {side} {key} execution status: {status}. Message: {res.get('message')}")

                        if any_success:
                            state["last_rebalance_price"] = price
                            cycles += 1
                            state["rebalance_cycles"] = cycles
                            state_dirty = True
                            logger.info(f"🎯 Baseline Updated: Last rebalance price set to {price:.6g}")

                # 3. GLOBAL SAFE SIPHONING (Runs every cycle)
                # siphoning_reserve is local variable, but we should update state as well
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
                # Use reconstructed real_equity for safe_calc
                safe_virt_debt = Decimal(str(state.get("virt_debt", 0.0)))
                safe_real_equity_adj = float(Decimal(str(safe_real_equity)) - safe_virt_debt)
                if safe_real_equity_adj < 0: safe_real_equity_adj = 0.0

                safe_calc = PortfolioCalculator(
                    positions=safe_positions,
                    spot_price=price,
                    real_equity=safe_real_equity_adj,
                    virt_qty=virt_qty,
                    virt_debt=float(safe_virt_debt),
                    base_ticker=base_ticker,
                    siphoning_reserve=siphoning_reserve,
                    targets=targets,
                    initial_capital=initial_tpv,
                    long_entry_price=safe_l_entry,
                    short_entry_price=safe_s_entry,
                    last_rebalance_price=state.get("last_rebalance_price", 0.0),
                    min_notional=active_min_notional
                )
                safe_calc_res = safe_calc.calculate_rebalance(targets, 0.0, 0.0, safe_calc.tpv, True)

                total_tpv_final = safe_calc_res["total_tpv"]

                # ВСЕГДА обновляем last_tpv и last_profit для актуального PnL в summary
                state["last_tpv"] = total_tpv_final
                state["last_profit"] = total_tpv_final - initial_tpv
                state["total_pnl_pct"] = safe_calc_res.get("total_pnl_pct", 0.0)
                state["last_update"] = time.time()
                state_dirty = True

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
                        paper_state_dirty = True

                        state["siphoning_reserve"] = siphoning_reserve
                        state_dirty = True
                        logger.info(f"💰 SAFE ACTIVATED: Siphoned {siphon_amount:.4f} USDT. New Reserve: {siphoning_reserve:.2f}")
                        asyncio.create_task(notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT (Surplus)"))

                # Update reporting value in summary to account for new reserve
                final_reported_tpv = total_tpv_final

                if any_success:
                    summary_msg = (
                        f"<b>✅ Rebalance #{cycles} Complete</b>: <code>{base_ticker}</code>\n"
                        f"New Shares: L:{safe_calc_res['share_long_pct']:.1f}% S:{safe_calc_res['share_short_pct']:.1f}% V:{safe_calc_res['share_virt_pct']:.1f}%\n"
                        f"TPV: <code>{total_tpv_final:.2f} USDT</code>"
                    )
                    logger.info(f"Rebalance #{cycles} complete. TPV: {total_tpv_final:.2f}")
                    
                    # [V3.9.0] Silence Telegram for Rebalance #1 (Baseline formation) to avoid startup spam
                    if cycles > 1:
                        asyncio.create_task(notifier.send_message(summary_msg))
                    else:
                        logger.info(f"ℹ️ Rebalance #1 (Baseline) notification suppressed in Telegram.")

                # Обновляем cycles в стейте (если были действия)
                state.update({
                    "rebalance_cycles": cycles
                })
                state_dirty = True

                # [LIQUIDATION GUARD] Verify real positions after rebalance
                if not paper_mode and any_success:
                    try:
                        real_pos_dict = await connector.get_position_risk()
                        # Convert dict {"SYMBOL_LONG": {...}, "SYMBOL_SHORT": {...}} to list
                        real_pos = []
                        for key, val in real_pos_dict.items():
                            entry = dict(val)
                            entry["symbol"] = key.split("_")[0] if "_" in key else key
                            entry["positionSide"] = key.split("_")[1] if "_" in key else "BOTH"
                            real_pos.append(entry)
                        real_pos = [p for p in real_pos if base_ticker in p.get("symbol", "")]
                        has_long = False
                        has_short = False
                        for p in real_pos:
                            amt = float(p.get("positionAmt", 0))
                            if abs(amt) < 1e-10:
                                continue
                            ps = p.get("positionSide", "")
                            if "LONG" in ps or (ps == "BOTH" and amt > 0):
                                has_long = True
                            elif "SHORT" in ps or (ps == "BOTH" and amt < 0):
                                has_short = True

                        # Check if one side is missing (liquidated)
                        expected_long = float(state.get("positions", {}).get(f"{base_ticker}_LONG", 0))
                        expected_short = float(state.get("positions", {}).get(f"{base_ticker}_SHORT", 0))

                        if expected_long != 0 and not has_long:
                            logger.error(f"🚨 LIQUIDATION DETECTED: {base_ticker}_LONG is MISSING on exchange!")
                            await _handle_liquidation_recovery(connector, base_ticker, state, state_file_path,
                                                                paper_state, paper_state_file_path,
                                                                config_path, logger, notifier)
                            while True:
                                await asyncio.sleep(86400)
                        if expected_short != 0 and not has_short:
                            logger.error(f"🚨 LIQUIDATION DETECTED: {base_ticker}_SHORT is MISSING on exchange!")
                            await _handle_liquidation_recovery(connector, base_ticker, state, state_file_path,
                                                                paper_state, paper_state_file_path,
                                                                config_path, logger, notifier)
                            while True:
                                await asyncio.sleep(86400)
                    except Exception as e:
                        logger.warning(f"Liquidation guard check failed: {e}")

                if state_dirty:
                    await save_json(state_file_path, state)
                if paper_state_dirty:
                    await save_json(paper_state_file_path, paper_state)

                if (i + status_offset) % 100 == 0:
                    total_pnl_final = safe_calc_res.get("total_pnl", total_tpv_final - initial_tpv)
                    logger.info(f"Heartbeat: TPV={total_tpv_final:.2f} | PnL={total_pnl_final:+.2f} | {base_ticker}={price:.6g} | Cycles={cycles}")

            except Exception as e:
                logger.error(f"Error in cycle: {e}")
                await asyncio.sleep(10)
            await asyncio.sleep(check_interval)
            i += 1
    finally:
        await notifier.close()

async def emergency_stop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None, paper_mode: bool = False, close_only: bool = False):
    config = safe_load_json_sync(config_path, {})

    base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")
    portfolio_cfg = config.get("portfolios", [{}])[0]
    initial_capital = portfolio_cfg.get("initial_capital", 60.0)

    logger.info(f"🛑 EMERGENCY STOP for {base_ticker} (Paper: {paper_mode}, CloseOnly: {close_only})")

    # --- PERSISTENCE PROTOCOL: Archive state before reset ---
    # --- PERSISTENCE PROTOCOL: Archive state before reset ---
    try:
        history_dir = Path("history")
        history_dir.mkdir(exist_ok=True)
        state = await load_json(state_file_path, {})
        paper_state = await load_json(paper_state_file_path, {})

        # Ensure final profit is calculated from the *current* state before archiving
        current_total_tpv_final = float(state.get("last_tpv", 0.0))
        current_initial_tpv = float(state.get("initial_tpv", 0.0))
        final_calculated_profit = current_total_tpv_final - current_initial_tpv

        archive_data = {
            "ticker": base_ticker,
            "timestamp": time.time(),
            "state": state,
            "paper_state": paper_state,
            "final_profit": final_calculated_profit,
            "final_total_tpv": current_total_tpv_final,
            "final_pnl_pct": state.get("total_pnl_pct", 0.0)
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
                        qty=Decimal(str(abs(qty))),
                        side=side,
                        step_size=Decimal(str(step_size)),
                        reduce_only=True,
                        position_side=pos_key.split('_')[1] if '_' in pos_key else "BOTH",
                        min_notional=Decimal('0')
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
            "virt_debt": 0.0,
            "initial_tpv": 0.0,
            "reference_tpv": 0.0,
            "tpv_ath": 0.0,
            # "trailing_stop_triggered": False, # [RESTRICTION] Manual reset only by supervisor
            "trailing_stop_violation_start": 0.0
        })
        await save_json(state_file_path, state)
        logger.info(f"✅ Emergency stop completed for {base_ticker}. All positions closed and state reset (TS flag preserved).")
    else:
        # При close_only (например, при ротации супервайзером) выполняем санитарию ATH и базы,
        # чтобы при следующем запуске бот не стартанул с глубокой просадки относительно старого ATH.
        try:
            state = await load_json(state_file_path, {})
            if state:
                current_tpv = float(state.get("last_tpv", 0.0))
                if current_tpv > 0:
                    logger.info(f"🔄 Санитизация состояния при плановом стопе ({base_ticker}). Ребазирование на {current_tpv:.4f}")
                    state.update({
                        "initial_tpv": current_tpv,
                        "reference_tpv": current_tpv,
                        "tpv_ath": current_tpv,
                        "trailing_stop_violation_start": 0.0
                    })
                    await save_json(state_file_path, state)
        except Exception as e:
            logger.error(f"Failed to sanitize state during emergency_stop: {e}")
        logger.info(f"✅ Positions closed for {base_ticker}. State preserved and sanitized.")

if __name__ == "__main__":
    import argparse
    import sys
    import io

    # Force UTF-8 for Windows streams
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--stop", action="store_true", help="Close all positions and stop")
    parser.add_argument("--close-only", action="store_true", help="Only close positions on exchange, preserve bot state")
    parser.add_argument("--wipe", action="store_true", help="Wipe bot state (destructive stop)")
    parser.add_argument("--paper", action="store_true", help="Force paper mode for this instance")
    parser.add_argument("--real", action="store_true", help="Force real mode (live) for this instance")
    args = parser.parse_args()
    
    config_base = os.path.splitext(os.path.basename(args.config))[0]

    def get_initial_cfg():
        return safe_load_json_sync(args.config, {})

    cfg = get_initial_cfg()
    base_ticker = args.ticker if args.ticker else cfg.get("base_ticker", "BTCUSDT")
    
    # Paper mode logic:
    # 1. If --real flag is present, force REAL mode.
    # 2. If --paper flag is present, force PAPER mode.
    # 3. Otherwise, fall back to global config paper_mode.
    if args.real:
        is_paper_instance = False
    elif args.paper:
        is_paper_instance = True
    else:
        is_paper_instance = cfg.get("paper_mode", False)

    # Configure individual logger
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_prefix = "paper" if is_paper_instance else "real"
    log_filename = f"{log_prefix}_{base_ticker}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(f"Bot_{base_ticker}_{log_prefix}")
    logger.setLevel(logging.INFO)
    # Clear handlers if any (prevent double logging on reload if it ever happens)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    
    # File Handler (UTF-8)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    if is_paper_instance:
        # PAPER mode uses paper_state_*.json for primary state and paper_shadow_*.json for simulated wallet
        instance_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"paper_state_{base_ticker}.json"))
        instance_paper_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"paper_shadow_{base_ticker}.json"))
    else:
        # REAL mode uses real_state_*.json for primary state and shadow_state_*.json for shadow balance
        instance_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"real_state_{base_ticker}.json"))
        instance_paper_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"shadow_state_{base_ticker}.json"))
    
    api_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))

    # [Hardware Block] Zombie Mode to prevent PM2 autorestart loop and API spam
    pre_state = safe_load_json_sync(instance_state_file, {})
    if pre_state.get("trailing_stop_triggered", False) and not args.stop and not args.wipe:
        logger.critical(f"🚨 [ZOMBIE MODE] Trailing Stop triggered for {base_ticker}. Entering infinite sleep to block PM2 autorestart.")
        import time
        while True:
            time.sleep(86400)

    if os.environ.get("MOCK_MODE") == "1":
        from connector import BinanceConnectorMock
        connector = BinanceConnectorMock()
    else:
        @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10),
               before_sleep=lambda retry_state: logger.warning(f"Connection failed, retrying... (attempt {retry_state.attempt_number})"))
        async def init_connector():
            conn = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))
            await conn.verify_connection()
            return conn

        try:
            connector = asyncio.run(init_connector())
            logger.info("BinanceConnector initialized and verified successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize BinanceConnector after 5 attempts: {e}")
            exit(1)

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