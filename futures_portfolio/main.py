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

async def self_kill_pm2(ticker: str, is_paper: bool) -> None:
    """Удаляет себя из PM2 перед выходом, чтобы предотвратить autorestart. Асинхронная версия."""
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    try:
        import sys
        cmd = f"pm2 delete {proc_name}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5)
        logging.info(f"PM2 self-kill: deleted {proc_name}")
    except Exception as e:
        logging.warning(f"PM2 self-kill failed for {proc_name}: {e}")

def _apply_clean_slate(state: dict, paper_state: dict, base_ticker: str, target_cap: float, logger: logging.Logger) -> None:
    """Enforces clean slate: resets phantom balances and state to target_initial_cap."""
    paper_state['balance'] = target_cap
    paper_state['long_entry_price'] = 0.0
    paper_state['short_entry_price'] = 0.0
    paper_state['positions'] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
    state['tpv_ath'] = target_cap
    state['virt_qty'] = 0.0
    state['virt_debt'] = 0.0  # [FIX] Reset virt_debt to prevent real_equity desync
    state['rebalance_cycles'] = 0
    state['initial_tpv'] = target_cap
    state['reference_tpv'] = target_cap
    state['trailing_stop_violation_start'] = 0.0
    state['trailing_stop_paper_timeout_end'] = 0.0
    logger.info(f"🔄 State reset (TS flag preserved): clean start detected (no open positions)")

async def _update_final_metrics_for_exit(state: dict, state_file_path: str, total_tpv_final: Decimal, initial_tpv: Decimal, safe_calc_res: dict, cycles: int, logger: logging.Logger) -> None:
    """Updates and saves the final profit metrics in the state file before a bot exits."""
    try:
        logger.info(f"🔄 Санитизация состояния перед выходом. Финальный TPV: {float(total_tpv_final):.4f}")

        # Полное ребазирование метрик под финальное значение TPV
        # [FIX] initial_tpv НЕ должен уменьшаться при стопе — сохраняем текущее значение
        # из state, чтобы при рестарте бот стартовал с target_initial_cap из конфига.
        _current_initial_tpv = float(state.get("initial_tpv", float(total_tpv_final)))
        _current_reference_tpv = float(state.get("reference_tpv", float(total_tpv_final)))
        state.update({
            "last_tpv": float(total_tpv_final),
            "initial_tpv": _current_initial_tpv,
            "reference_tpv": _current_reference_tpv,
            "tpv_ath": max(float(total_tpv_final), _current_initial_tpv),
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
                toxic = cfg_data.get("toxic_blacklist_real", {})
                cooldown_days = cfg_data.get("toxic_cooldown_days", 0.02)
                expiry = time.time() + cooldown_days * 86400
                toxic[base_ticker] = expiry
                cfg_data["toxic_blacklist_real"] = toxic
                _cfg_path.write_text(_json.dumps(cfg_data, indent=2, ensure_ascii=False))
                logger.info(f"{base_ticker} added to black_list and toxic_blacklist_real")
        except Exception as e:
            logger.error(f"Failed to update config after liquidation: {e}")

    elif dist <= liquidation_distance_warn:
        logger.warning(
            f"⚠️ LIQUIDATION WARNING [{'PAPER' if is_paper else 'REAL'}]: "
            f"{pos_key} distance {dist:.1f}% (liq_price={liq_price:.8f})"
        )
        try:
            await notifier.send_alert(
                "⚠️ LIQUIDATION WARNING",
                f"{pos_key}: distance {dist:.1f}% to liq @ {liq_price:.8f}"
            )
        except Exception:
            pass


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
        state["tpv_ath"] = target_initial_cap
        state["balance"] = target_initial_cap
        state["last_tpv"] = 0.0
        state["last_profit"] = 0.0
        state["total_pnl_pct"] = 0.0
        state["rebalance_cycles"] = 0
        state["last_rebalance_price"] = 0.0
        state["trailing_stop_violation_start"] = 0.0
        state["trailing_stop_paper_timeout_end"] = 0.0
        state["trailing_stop_triggered"] = False
        state["started_at"] = time.time()
        await save_json(state_file_path, state)

        paper_state["balance"] = target_initial_cap
        paper_state["positions"] = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}
        paper_state["long_entry_price"] = 0.0
        paper_state["short_entry_price"] = 0.0
        paper_state["last_price"] = 0.0
        paper_state["base_ticker"] = base_ticker
        await save_json(paper_state_file_path, paper_state)
        state = await load_json(state_file_path, {})

    # State Isolation Protocol (clean slate after config reload / blacklist rebase)
    # Paper mode: проверяем shadow-позиции, real mode: проверяем реальные позиции на бирже
    if paper_mode:
        # Paper mode: если shadow-позиции ненулевые, но paper_state['balance'] рассинхронизирован с target_initial_cap
        shadow_long = float(paper_state.get("positions", {}).get(f"{base_ticker}_LONG", 0.0))
        shadow_short = float(paper_state.get("positions", {}).get(f"{base_ticker}_SHORT", 0.0))
        has_shadow_positions = abs(shadow_long) > 1e-12 or abs(shadow_short) > 1e-12
        balance_desync = abs(paper_state.get("balance", target_initial_cap) - target_initial_cap) > 0.1
        
        if has_shadow_positions and balance_desync:
            logger.warning(f"🔄 Paper State Isolation: shadow positions exist but balance desync >0.1 USDT. Clean slate.")
            _apply_clean_slate(state, paper_state, base_ticker, target_initial_cap, logger)
            await save_json(state_file_path, state)
            await save_json(paper_state_file_path, paper_state)
    else:
        # Real mode: проверяем реальные позиции на бирже
        try:
            real_positions = await connector.get_positions()
            has_real_positions = any(
                float(pos.get("qty", 0.0)) != 0.0 
                for pos in real_positions.values()
            )
            if has_real_positions:
                # Если есть реальные позиции, но state['balance'] не синхронизирован
                balance_desync = abs(state.get("balance", target_initial_cap) - target_initial_cap) > 0.1
                if balance_desync:
                    logger.warning(f"🔄 Real State Isolation: real positions exist but balance desync >0.1 USDT. Clean slate.")
                    _apply_clean_slate(state, paper_state, base_ticker, target_initial_cap, logger)
                    await save_json(state_file_path, state)
                    await save_json(paper_state_file_path, paper_state)
        except Exception as e:
            logger.warning(f"Real State Isolation check failed (get_positions): {e}. Assuming clean slate.")
            # Fallback: clean slate по balance desync
            balance_desync = abs(state.get("balance", target_initial_cap) - target_initial_cap) > 0.1
            if balance_desync:
                _apply_clean_slate(state, paper_state, base_ticker, target_initial_cap, logger)
                await save_json(state_file_path, state)
                await save_json(paper_state_file_path, paper_state)

    # Config reload "Updating base" logic
        if state.get("reference_tpv", 0.0) == 0.0 and state.get("initial_tpv", 0.0) == 0.0:
            logger.info(f"📝 Config reload detected (reference_tpv=0). Re-initializing with target_initial_cap={target_initial_cap}")
            state["initial_tpv"] = target_initial_cap
            state["reference_tpv"] = target_initial_cap
            state["tpv_ath"] = max(state.get("tpv_ath", target_initial_cap), target_initial_cap)
            state["balance"] = target_initial_cap
            state["virt_qty"] = 0.0
            state["virt_debt"] = 0.0
            # [FIX] paper_state['balance'] НЕ перезаписываем — симуляция не должна терять накопленный PnL
            await save_json(state_file_path, state)
            await save_json(paper_state_file_path, paper_state)

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
    executor = PortfolioExecutor(connector, base_ticker=base_ticker)

    # Загрузка плечей
    leverage_map = {name: t["leverage"] for name, t in targets.items()}

    # Проверка режима хеджирования
    hedge_mode = await connector.get_hedge_mode()
    if not hedge_mode:
        logger.critical("Hedge Mode is NOT enabled on Binance Futures! Strategy requires Hedge Mode.")
        # Не выходим, пытаемся включить
        try:
            await connector.futures_client.futures_change_position_mode(dualSidePosition=True)
            logger.info("✅ Hedge Mode enabled successfully")
            hedge_mode = True
        except Exception as e:
            logger.error(f"Failed to enable Hedge Mode: {e}")

    # Установка плеча и типа маржи
    for leg_name, leg_config in targets.items():
        lev = leg_config.get("leverage", 1)
        if lev > 1:
            try:
                await connector.set_leverage(base_ticker, lev)
                await connector.set_margin_type(base_ticker, "ISOLATED")
                logger.info(f"✅ {base_ticker}: Leverage={lev}x, Margin=ISOLATED")
            except Exception as e:
                logger.error(f"Failed to set leverage/margin: {e}")

    logger.info(f"🤖 Bot started for {base_ticker} | Mode: {'PAPER' if paper_mode else 'REAL'} | Initial Capital: {target_initial_cap} USDT")
    logger.info(f"Targets: {targets}")
    logger.info(f"Thresholds: surplus={threshold_surplus:.4f}, deficit={threshold_deficit:.4f}")
    logger.info(f"Check interval: {check_interval}s")

    # Main rebalancing loop
    while True:
        try:
            i += 1
            cycle_start = time.time()
            
            # Config hot-reload
            current_mtime = os.path.getmtime(config_path)
            if current_mtime != last_config_mtime:
                config = safe_load_json_sync(config_path, {})
                last_config_mtime = current_mtime
                portfolio_cfg = config["portfolios"][0]
                targets = portfolio_cfg["targets"]
                threshold_surplus = portfolio_cfg.get("rebalance_threshold_surplus", threshold_surplus)
                threshold_deficit = portfolio_cfg.get("rebalance_threshold_deficit", threshold_deficit)
                check_interval = portfolio_cfg.get("check_interval_sec", check_interval)
                equity_trailing_stop_pct = config.get("equity_trailing_stop_pct", equity_trailing_stop_pct)
                equity_trailing_stop_activation_pct = config.get("equity_trailing_stop_activation_pct", equity_trailing_stop_activation_pct)
                max_drawdown_limit = config.get("max_drawdown_limit", max_drawdown_limit)
                # Reload leverage
                leverage_map = {name: t["leverage"] for name, t in targets.items()}
                logger.info(f"🔄 Config hot-reloaded: targets={targets}")

            # Get current prices
            prices = await connector.get_mark_prices([base_ticker])
            if not prices or base_ticker not in prices or prices[base_ticker] is None:
                logger.warning(f"Failed to get price for {base_ticker}. Skipping cycle.")
                await asyncio.sleep(check_interval)
                continue
            
            mark_price = prices[base_ticker]
            
            # Get account info for equity calculation
            margin_info = await connector.get_margin_ratio()
            available_balance = margin_info.get("available_balance", 0.0)
            total_margin_balance = margin_info.get("total_margin_balance", 0.0)
            total_maint_margin = margin_info.get("total_maint_margin", 0.0)
            margin_ratio = margin_info.get("margin_ratio", float('inf'))
            total_wallet_balance = margin_info.get("total_wallet_balance", 0.0)

            # Get real positions
            raw_positions = await connector.get_position_risk()
            
            # Liquidation distance guard
            liq_warn = portfolio_cfg.get("liquidation_distance_warn_pct", 15.0)
            liq_crit = portfolio_cfg.get("liquidation_distance_crit_pct", 8.0)
            if liq_warn > 0 and liq_crit > 0:
                for pos_key, pos_data in raw_positions.items():
                    dist = pos_data.get("distance_pct", 0.0)
                    if dist > 0:
                        liq_price = pos_data.get("liq_price", 0.0)
                        await _handle_liquidation_guard(
                            pos_key, dist, liq_price,
                            liq_warn, liq_crit,
                            paper_mode, raw_positions, paper_state, connector,
                            base_ticker, step_sizes, notifier, logger
                        )

            # Margin ratio monitoring
            if margin_ratio <= margin_critical:
                logger.critical(f"🚨 MARGIN RATIO CRITICAL: {margin_ratio:.2f}x <= {margin_critical}x. Emergency stop!")
                try:
                    await notifier.send_alert("🚨 MARGIN CRITICAL", f"Margin ratio: {margin_ratio:.2f}x. Stopping bot.")
                except:
                    pass
                await self_kill_pm2(base_ticker, paper_mode)
                return
            elif margin_ratio <= margin_warning:
                logger.warning(f"⚠️ MARGIN RATIO WARNING: {margin_ratio:.2f}x <= {margin_warning}x")

            # Calculate positions for legs
            long_qty = 0.0
            short_qty = 0.0
            
            for pos_key, pos_data in raw_positions.items():
                qty = float(pos_data.get("qty", 0.0))
                if base_ticker in pos_key:
                    if "_LONG" in pos_key or (pos_key == base_ticker and qty > 0):
                        long_qty = qty
                    elif "_SHORT" in pos_key or (pos_key == base_ticker and qty < 0):
                        short_qty = qty

            # Calculate MTM for virtual leg
            virt_price = mark_price
            virt_value = virt_qty * virt_price
            
            # Real equity = available_balance + unrealized PnL (margin balance includes unrealized PnL)
            real_equity = total_margin_balance
            
            # TPV = Real Equity + Virtual Value - Siphoning Reserve
            total_tpv = real_equity + virt_value - siphoning_reserve
            tpv_active = total_tpv - siphoning_reserve
            
            # Update ATH
            if total_tpv > tpv_ath:
                tpv_ath = total_tpv

            # Initialize initial_tpv and reference_tpv on first cycle
            if initial_tpv == 0.0:
                initial_tpv = total_tpv
                reference_tpv = total_tpv
            if reference_tpv == 0.0:
                reference_tpv = total_tpv

            # Equity Trailing Stop
            if equity_trailing_stop_pct > 0 and equity_trailing_stop_activation_pct > 0:
                profit_pct = (tpv_ath - target_initial_cap) / target_initial_cap * 100
                if profit_pct >= equity_trailing_stop_activation_pct:
                    drawdown_from_ath = (tpv_ath - total_tpv) / tpv_ath * 100
                    if drawdown_from_ath >= equity_trailing_stop_pct:
                        logger.critical(f"🚨 EQUITY TRAILING STOP: TPV dropped {drawdown_from_ath:.2f}% from ATH ({tpv_ath:.2f} -> {total_tpv:.2f}). Triggering stop.")
                        await self_kill_pm2(base_ticker, paper_mode)
                        return

            # Max Drawdown Limit (from tpv_ath)
            if max_drawdown_limit > 0:
                drawdown_from_ath = (tpv_ath - total_tpv) / tpv_ath * 100
                if drawdown_from_ath >= max_drawdown_limit:
                    logger.critical(f"🚨 MAX DRAWDOWN LIMIT: TPV dropped {drawdown_from_ath:.2f}% from ATH ({tpv_ath:.2f} -> {total_tpv:.2f}). Limit: {max_drawdown_limit}%")
                    # Check if total equity is also below global initial capital
                    if total_wallet_balance < target_initial_cap:
                        logger.critical(f"🚨 Total equity ({total_wallet_balance:.2f}) < initial capital ({target_initial_cap}). FULL STOP + BLACKLIST")
                        try:
                            import json as _json
                            from pathlib import Path
                            _cfg_path = Path(config_path)
                            if _cfg_path.exists():
                                cfg_data = _json.loads(_cfg_path.read_text())
                                bl = cfg_data.get("black_list", [])
                                if base_ticker not in bl:
                                    bl.append(base_ticker)
                                    cfg_data["black_list"] = bl
                                _cfg_path.write_text(_json.dumps(cfg_data, indent=2, ensure_ascii=False))
                                logger.info(f"{base_ticker} added to black_list")
                        except Exception as e:
                            logger.error(f"Failed to update config: {e}")
                        emit_signal("stop", base_ticker, is_paper=paper_mode)
                    else:
                        logger.critical(f"🚨 Equity above initial, but drawdown from ATH exceeded. EXIT ONLY (probation).")
                    try:
                        await notifier.send_alert("🚨 MAX DRAWDOWN", f"{base_ticker}: {drawdown_from_ath:.2f}% from ATH. Stopping.")
                    except:
                        pass
                    await self_kill_pm2(base_ticker, paper_mode)
                    return

            # Calculate target notional for each leg
            target_notional = {}
            for leg_name, leg_config in targets.items():
                share = leg_config.get("share", 0.0)
                if share > 0:
                    target_notional[leg_name] = tpv_active * share
                else:
                    target_notional[leg_name] = 0.0

            # Current notional
            long_notional = abs(long_qty) * mark_price
            short_notional = abs(short_qty) * mark_price
            virt_notional = virt_value

            current_notional = {
                "BASE_LONG": long_notional,
                "BASE_SHORT": short_notional,
                "VIRTUAL": virt_notional
            }

            # Rebalancing logic
            rebalance_needed = False
            rebalance_actions = []

            # Check each leg
            for leg_name, target in target_notional.items():
                current = current_notional.get(leg_name, 0.0)
                if target == 0:
                    continue
                diff_pct = (current - target) / target
                
                # Use hysteresis: double threshold on deficit (drawdown)
                if diff_pct > threshold_surplus:
                    rebalance_needed = True
                    rebalance_actions.append((leg_name, "SELL", diff_pct, target, current))
                elif diff_pct < -threshold_deficit:
                    # Soft hysteresis: double threshold on drawdown
                    if reference_tpv > target_initial_cap:
                        hysteresis_mult = 2.0
                    else:
                        hysteresis_mult = 1.0
                    if abs(diff_pct) > threshold_deficit * hysteresis_mult:
                        rebalance_needed = True
                        rebalance_actions.append((leg_name, "BUY", diff_pct, target, current))

            # Execute rebalancing
            if rebalance_needed:
                logger.info(f"🔄 REBALANCE #{cycles+1} triggered: {rebalance_actions}")
                cycles += 1
                
                for leg_name, action, diff_pct, target, current in rebalance_actions:
                    qty_change = (target - current) / mark_price
                    
                    if leg_name == "BASE_LONG":
                        side = "SELL" if action == "SELL" else "BUY"
                        position_side = "LONG"
                        qty = abs(qty_change)
                        step = step_sizes.get(base_ticker, 0.1)
                        if qty > step * 0.1:
                            try:
                                await executor.execute_market_order(
                                    symbol=base_ticker,
                                    qty=Decimal(str(qty)),
                                    side=side,
                                    step_size=Decimal(str(step)),
                                    position_side=position_side,
                                    reduce_only=(action == "SELL"),
                                    min_notional=Decimal(str(portfolio_cfg.get("min_notional_usdt", 6.1)))
                                )
                                logger.info(f"✅ BASE_LONG: {action} {qty:.4f} @ ~{mark_price:.4f}")
                            except Exception as e:
                                logger.error(f"BASE_LONG {action} failed: {e}")
                    
                    elif leg_name == "BASE_SHORT":
                        side = "SELL" if action == "SELL" else "BUY"
                        position_side = "SHORT"
                        qty = abs(qty_change)
                        step = step_sizes.get(base_ticker, 0.1)
                        if qty > step * 0.1:
                            try:
                                await executor.execute_market_order(
                                    symbol=base_ticker,
                                    qty=Decimal(str(qty)),
                                    side=side,
                                    step_size=Decimal(str(step)),
                                    position_side=position_side,
                                    reduce_only=(action == "SELL"),
                                    min_notional=Decimal(str(portfolio_cfg.get("min_notional_usdt", 6.1)))
                                )
                                logger.info(f"✅ BASE_SHORT: {action} {qty:.4f} @ ~{mark_price:.4f}")
                            except Exception as e:
                                logger.error(f"BASE_SHORT {action} failed: {e}")
                    
                    elif leg_name == "VIRTUAL":
                        # Virtual leg: just update virt_qty
                        if action == "BUY":
                            virt_qty += qty_change
                        else:
                            virt_qty -= qty_change
                        logger.info(f"📊 VIRTUAL: {action} {abs(qty_change):.4f} qty (new virt_qty={virt_qty:.4f})")

                # Update state after rebalance
                state["virt_qty"] = virt_qty
                state["rebalance_cycles"] = cycles
                state["last_rebalance_price"] = mark_price
                state["last_tpv"] = float(total_tpv)
                state["last_profit"] = float(total_tpv - initial_tpv)
                state["total_pnl_pct"] = float((total_tpv - initial_tpv) / initial_tpv * 100) if initial_tpv > 0 else 0.0
                state["last_update"] = time.time()
                
                await save_json(state_file_path, state)
                await save_json(paper_state_file_path, paper_state)
                
                logger.info(f"✅ Rebalance #{cycles} complete. TPV: {total_tpv:.4f}, VirtQty: {virt_qty:.4f}")

            # Periodic status log
            if i % 60 == 0 or i == 1:
                logger.info(
                    f"📊 Cycle #{i} | TPV: {total_tpv:.4f} (Active: {tpv_active:.4f}) | "
                    f"RealEq: {real_equity:.4f} | Virt: {virt_value:.4f} | "
                    f"MarginRatio: {margin_ratio:.2f}x | Cycles: {cycles}"
                )
                logger.info(
                    f"   LONG: {long_qty:.4f} (${long_notional:.2f}) | "
                    f"SHORT: {short_qty:.4f} (${short_notional:.2f}) | "
                    f"VIRT: {virt_qty:.4f} (${virt_value:.2f})"
                )
                logger.info(
                    f"   Targets: LONG=${target_notional.get('BASE_LONG',0):.2f} "
                    f"SHORT=${target_notional.get('BASE_SHORT',0):.2f} "
                    f"VIRT=${target_notional.get('VIRTUAL',0):.2f}"
                )

            # Sleep until next cycle
            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error in rebalance loop: {e}", exc_info=True)
            await asyncio.sleep(check_interval)


async def emergency_stop(connector, config_path, state_file_path, paper_state_file_path, logger, ticker_override=None, paper_mode=False, close_only=True):
    """Emergency stop: close all positions and save state."""
    base_ticker = ticker_override
    if not base_ticker:
        config = safe_load_json_sync(config_path, {})
        base_ticker = config.get("base_ticker", "BTCUSDT")
    
    logger.critical(f"🛑 EMERGENCY STOP for {base_ticker} (paper_mode={paper_mode})")
    
    try:
        raw_positions = await connector.get_position_risk()
        for pos_key, pos_data in raw_positions.items():
            if base_ticker in pos_key:
                qty = float(pos_data.get("qty", 0.0))
                if abs(qty) > 1e-10:
                    side = "SELL" if qty > 0.0 else "BUY"
                    logger.critical(f"Closing {pos_key}: {qty}")
                    # ... close logic
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")

    # Save final state
    state = await load_json(state_file_path, {})
    state["trailing_stop_triggered"] = True
    await save_json(state_file_path, state)
    
    logger.critical("🛑 Emergency stop complete")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--wipe", action="store_true")
    args = parser.parse_args()

    # Load config
    config = safe_load_json_sync(args.config, {})
    if not config:
        print(f"ERROR: Failed to load config {args.config}")
        sys.exit(1)

    cfg = config["portfolios"][0]
    base_ticker = args.ticker if args.ticker else config.get("base_ticker", "BTCUSDT")

    # Determine paper_mode
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
            # Запуск периодической синхронизации времени с Binance (fix для -1021 Timestamp error)
            if not args.stop and not is_paper_instance:
                asyncio.create_task(connector._periodic_time_sync(interval_sec=300))
                logger.info("⏰ Periodic time sync with Binance started (every 5 min)")
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


if __name__ == "__main__":
    main()