import asyncio
import json
import logging
import os
import time
from typing import Dict, List
from dotenv import load_dotenv

# Load .env file
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
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tmp_path, path)
                return
            except PermissionError:
                if i == retries - 1:
                    break
                time.sleep(0.5)
    await asyncio.to_thread(_write)

async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger, ticker_override: str = None):
    config = await load_json(config_path, {})
    paper_mode = config.get("paper_mode", False)
    
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
        "rebalance_cycles": 0
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
        await save_json(state_file_path, state)

    virt_basis_price = state["virt_basis_price"]
    virt_allocated_usdt = state["virt_allocated_usdt"]
    siphoning_reserve = state.get("siphoning_reserve", 0.0)
    initial_tpv = state.get("initial_tpv", 0.0)
    reference_tpv = state.get("reference_tpv", 0.0)
    tpv_ath = state.get("tpv_ath", 0.0)

    # Параметры капитала и защиты
    max_capital_usdt = portfolio_cfg.get("max_capital_usdt", 0.0)
    
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
                await notifier.send_alert("STARTUP ERROR", msg)
                return
            logger.info("Hedge Mode verified.")
        except Exception as e:
            logger.error(f"Failed to verify Hedge Mode: {e}")
            await notifier.send_alert("STARTUP ERROR", f"Could not verify Hedge Mode: {e}")
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
            
            # Ограничение капитала, если задано
            if max_capital_usdt > 0:
                real_equity = min(real_equity, max_capital_usdt)

            if virt_basis_price == 0 or initial_tpv == 0:
                if virt_basis_price == 0:
                    virt_basis_price = price
                    virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                if initial_tpv == 0:
                    temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                 base_ticker=base_ticker, siphoning_reserve=0.0, targets=targets,
                                                 long_entry_price=l_entry, short_entry_price=s_entry,
                                                 initial_capital=real_equity)
                    initial_tpv = temp_calc.tpv
                    reference_tpv = initial_tpv
                    logger.info(f"Initialized TPV base: {initial_tpv:.2f}")

                state.update({
                    "virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt,
                    "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                    "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                })
                await save_json(state_file_path, state)

            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                     long_entry_price=l_entry, short_entry_price=s_entry,
                                     initial_capital=initial_tpv)
            
            if tpv_ath == 0 or calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv
                state["tpv_ath"] = tpv_ath
                await save_json(state_file_path, state)

            if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                drawdown_pct = (1 - calc.total_tpv / tpv_ath) * 100
                if drawdown_pct >= equity_trailing_stop_pct:
                    msg = f"Trailing Stop triggered: {drawdown_pct:.2f}% drop from ATH. Closing all positions for {base_ticker}."
                    logger.warning(f"!!! [STOP] {msg}")
                    await notifier.send_alert("STOP LOSS", msg)
                    for pos_key, qty in positions.items():
                        if qty == 0 or base_ticker not in pos_key: continue
                        side = "SELL" if qty > 0 else "BUY"
                        step_size = step_sizes.get(base_ticker, 0.0)
                        if paper_mode:
                            paper_state["positions"][pos_key] = 0.0
                        else:
                            await PortfolioExecutor(connector).execute_market_order(pos_key.split('_')[0], abs(qty), side, step_size, True, pos_key.split('_')[1] if '_' in pos_key else "BOTH")
                    if paper_mode: await save_json(paper_state_file_path, paper_state)
                    
                    # Сбрасываем ATH и начальные значения, чтобы при перезапуске бот не попал в цикл стоп-лоссов
                    state["tpv_ath"] = 0.0
                    state["initial_tpv"] = 0.0
                    state["reference_tpv"] = 0.0
                    state["virt_basis_price"] = 0.0
                    await save_json(state_file_path, state)
                    
                    logger.info("Positions closed and state reset. Bot stopped.")
                    break

            if not paper_mode and (margin_warning > 0 or margin_critical > 0):
                margin_info = await connector.get_margin_ratio()
                margin_ratio = margin_info.get("margin_ratio", float('inf'))
                if margin_critical > 0 and margin_ratio < margin_critical:
                    msg = f"Margin ratio {margin_ratio:.2f} < {margin_critical:.2f}. Emergency stop!"
                    logger.error(f"!!! [CRITICAL] {msg}")
                    await notifier.send_alert("CRITICAL MARGIN", msg)
                    break
                elif margin_warning > 0 and margin_ratio < margin_warning:
                    msg = f"Low margin ratio: {margin_ratio:.2f}"
                    logger.warning(f"!!! [WARNING] {msg}")
                    await notifier.send_message(f"⚠️ <b>WARNING</b>: {msg} ({base_ticker})")

            # Реинвестирование из свободной маржи (если reinvestment_ratio > 0)
            if reinvestment_ratio > 0:
                free_margin = real_equity - (calc.total_tpv - calc.virt_current_value)  # Свободная маржа
                if free_margin > 100:  # Минимум 100 USDT для реинвестирования
                    to_reinvest = free_margin * reinvestment_ratio
                    initial_tpv += to_reinvest
                    logger.info(f"💰 Reinvest from free margin: +{to_reinvest:.2f} USDT (ratio: {reinvestment_ratio})")
                    state.update({"initial_tpv": initial_tpv})
                    await save_json(state_file_path, state)

            current_pos_sum = abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0))
            is_first_run = current_pos_sum == 0
            is_extreme = calc.share_long_pct > 100 or calc.share_short_pct > 100

            if i % 5 == 0:
                 res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                 logger.info(f"Heartbeat: Balance={real_equity:.2f}{res_str} | {base_ticker}={price:.6g} | L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%")

            current_threshold = -1.0 if (is_first_run or is_extreme) else threshold
            if not (is_first_run or is_extreme) and reference_tpv > 0 and calc.tpv < reference_tpv:
                current_threshold *= 2.0

            actions: List[Dict] = calc.calculate_deviations(targets, current_threshold, ignore_limits=(current_threshold < 0))
            if actions:
                # Внедряем Notional Value Guard для физических ордеров
                min_notional = portfolio_cfg.get("min_notional_usdt", 6.0)
                
                # Фильтруем: оставляем VIRTUAL_RESET и физические ордера >= min_notional
                valid_actions = [a for a in actions if a["type"] == "VIRTUAL_RESET" or abs(a.get("diff_usdt", 0)) >= min_notional]
                
                if not valid_actions:
                    if i % 10 == 0:
                        logger.info(f"Rebalance actions identified, but they are too small (< {min_notional} USDT). Skipping.")
                    continue

                logger.info(f"Rebalance needed ({len(valid_actions)} actions). Shares: L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%")

                # Увеличиваем счетчик циклов и сохраняем
                cycles = state.get("rebalance_cycles", 0) + 1
                state["rebalance_cycles"] = cycles
                await save_json(state_file_path, state)

                # Отправляем уведомление о начале ребаланса
                rebalance_msg = (
                    f"<b>🔄 Rebalance #{cycles}</b>: <code>{base_ticker}</code>\n"
                    f"Shares: L:{calc.share_long_pct:.1f}% S:{calc.share_short_pct:.1f}% V:{calc.share_virt_pct:.1f}%\n"
                    f"TPV: <code>{calc.total_tpv:.2f} USDT</code>"
                )
                await notifier.send_message(rebalance_msg)

                limit_enabled, limit_offset, limit_timeout = PortfolioExecutor(connector).get_limit_order_params(portfolio_cfg)
                limit_stats = {"attempted": 0, "filled": 0, "fallback": 0, "total_profit_usdt": 0.0, "total_improvement_pct": 0.0}

                # В начале цикла ребаланса фиксируем доступный излишек для сифонинга
                # ПРАВИЛО: Сифоним только если реальный баланс (без PnL) выше начального
                excess_to_siphon: float = max(0, real_equity - initial_tpv)

                for action in valid_actions:
                    trade_pnl = 0.0 # Всегда инициализируем в начале обработки действия
                    
                    if action["type"] == "VIRTUAL_RESET":
                        # Ребалансировка виртуальной части - просто сброс базиса
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
                            paper_state["balance"] -= commission
                            
                            trade_pnl = 0.0 # Инициализация для логгера
                            
                            # Расчет новой средневзвешенной цены входа
                            pos_key = f"{base_ticker}_{pos_side}"
                            old_qty = paper_state["positions"].get(pos_key, 0.0)
                            entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                            old_entry = paper_state.get(entry_key, price)

                            is_buy = side == ("BUY" if pos_side == "LONG" else "SELL")
                            if is_buy:
                                # Докупка - пересчитываем среднюю цену входа
                                if old_qty > 0:
                                    new_entry = (old_qty * old_entry + qty_rounded * price) / (old_qty + qty_rounded)
                                else:
                                    new_entry = price
                                paper_state["positions"][pos_key] += qty_rounded
                                paper_state[entry_key] = new_entry
                            else:
                                # Продажа (уменьшение позиции) - цена входа не меняется
                                # 💰 СИФОНИНГ: Прибыль от каждой сделки -> в сейф
                                trade_pnl = 0.0
                                if pos_side == "LONG":
                                    # Long: прибыль = (цена_продажи - цена_входа) * количество
                                    trade_pnl = (price - old_entry) * qty_rounded - commission
                                else:
                                    # Short: прибыль = (цена_входа - цена_покупки) * количество
                                    trade_pnl = (old_entry - price) * qty_rounded - commission

                                if trade_pnl > 0 and excess_to_siphon > 0:
                                    siphon_amount: float = min(trade_pnl, excess_to_siphon)
                                    siphoning_reserve += siphon_amount
                                    excess_to_siphon -= siphon_amount
                                    paper_state["balance"] -= siphon_amount
                                    logger.info(f"💰 [SAFE] P&L siphoned: +{siphon_amount:.4f} USDT (Total reserve: {siphoning_reserve:.2f})")
                                    state.update({"siphoning_reserve": siphoning_reserve})
                                    await save_json(state_file_path, state)

                                paper_state["positions"][pos_key] -= qty_rounded
                                paper_state["positions"][pos_key] = max(0, paper_state["positions"][pos_key])
                                if paper_state["positions"][pos_key] == 0:
                                    paper_state[entry_key] = 0.0

                            logger.info(f"[PAPER] Order: {side} {qty_rounded:.6f} {key} @ {price:.6f} (Fee: {commission:.4f} USDT, Entry: {old_entry:.6f}, PnL: {trade_pnl:+.4f})")
                            await save_json(paper_state_file_path, paper_state)

                            # Уведомление о сделке
                            trade_msg = (
                                f"<b>📊 {base_ticker} Trade</b>\n"
                                f"Side: <code>{side}</code> {pos_side}\n"
                                f"Qty: <code>{qty_rounded:.6f}</code> @ {price:.6f}\n"
                                f"Fee: <code>{commission:.4f}</code> USDT"
                            )
                            await notifier.send_message(trade_msg)
                    else:
                        if limit_enabled:
                            limit_stats["attempted"] += 1
                            result = await PortfolioExecutor(connector).execute_limit_with_fallback(
                                symbol=key.split('_')[0], qty=abs(order_qty), side=side, step_size=step_size,
                                reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                                timeout_sec=limit_timeout, min_notional=min_notional
                            )
                            if result["status"] == "SUCCESS_LIMIT":
                                limit_stats["filled"] += 1
                                limit_stats["total_profit_usdt"] += result.get("profit_usdt", 0)
                                limit_stats["total_improvement_pct"] += result.get("price_improvement_pct", 0)
                                logger.info(f"✅ Limit FILLED: {result['filled_qty']} @ {result['avg_price']:.6f} | gain={result['price_improvement_pct']:+.3f}% ({result['profit_usdt']:+.2f} USDT)")

                                # 💰 СИФОНИНГ: Прибыль от каждой сделки (reduce_only) -> в сейф
                                if reduce_only:
                                    exec_price = result.get("avg_price", price)
                                    filled_qty = result.get("filled_qty", 0)
                                    commission = result.get("commission", filled_qty * exec_price * 0.0004)

                                    trade_pnl = 0.0
                                    if pos_side == "LONG":
                                        trade_pnl = (exec_price - l_entry) * filled_qty - commission
                                    else:
                                        trade_pnl = (s_entry - exec_price) * filled_qty - commission

                                    if trade_pnl > 0 and excess_to_siphon > 0:
                                        siphon_amount: float = min(trade_pnl, excess_to_siphon)
                                        siphoning_reserve += siphon_amount
                                        excess_to_siphon -= siphon_amount
                                        logger.info(f"💰 [SAFE] Real P&L siphoned: +{siphon_amount:.4f} USDT (Total reserve: {siphoning_reserve:.2f})")
                                        state.update({"siphoning_reserve": siphoning_reserve})
                                        await save_json(state_file_path, state)
                                        await notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT from {pos_side} {side}")

                            elif result["status"] == "SUCCESS_FALLBACK":
                                limit_stats["fallback"] += 1
                                if result.get("limit_filled_qty", 0) > 0:
                                    limit_stats["filled"] += 1
                                    limit_stats["total_improvement_pct"] += result.get("limit_price_improvement_pct", 0)
                                    logger.info(f"⚡ Limit+Fallback: {result['limit_filled_qty']} @ {result['limit_avg_price']:.6f} (gain={result.get('limit_price_improvement_pct', 0):+.3f}%) + market")

                                    # 💰 СИФОНИНГ: Прибыль от limit части (reduce_only) -> в сейф
                                    if reduce_only:
                                        exec_price = result.get("limit_avg_price", price)
                                        filled_qty = result.get("limit_filled_qty", 0)
                                        commission = filled_qty * exec_price * 0.0004

                                        trade_pnl = 0.0
                                        if pos_side == "LONG":
                                            trade_pnl = (exec_price - l_entry) * filled_qty - commission
                                        else:
                                            trade_pnl = (s_entry - exec_price) * filled_qty - commission

                                        if trade_pnl > 0 and excess_to_siphon > 0:
                                            siphon_amount: float = min(trade_pnl, excess_to_siphon)
                                            siphoning_reserve += siphon_amount
                                            excess_to_siphon -= siphon_amount
                                            logger.info(f"💰 [SAFE] Fallback P&L siphoned: +{siphon_amount:.4f} USDT (Total reserve: {siphoning_reserve:.2f})")
                                            state.update({"siphoning_reserve": siphoning_reserve})
                                            await save_json(state_file_path, state)
                                            await notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT from {pos_side} {side}")
                                else:
                                    logger.info(f"⚡ Fallback (market only): timeout")
                            elif result["status"] in ("ERROR_FALLBACK", "ERROR"):
                                logger.warning(f"❌ Order with issues: {result.get('error', result.get('message', 'unknown'))}")
                        else:
                            # Market order execution
                            await PortfolioExecutor(connector).execute_market_order(key, abs(order_qty), side, step_size, reduce_only, pos_side)

                            # 💰 СИФОНИНГ: Для market ордеров (reduce_only) -> в сейф (оценочно)
                            if reduce_only:
                                exec_price = price  # Используем текущую цену как оценку
                                filled_qty = abs(order_qty)
                                commission = filled_qty * exec_price * 0.0004

                                trade_pnl = 0.0
                                if pos_side == "LONG":
                                    trade_pnl = (exec_price - l_entry) * filled_qty - commission
                                else:
                                    trade_pnl = (s_entry - exec_price) * filled_qty - commission

                                if trade_pnl > 0 and excess_to_siphon > 0:
                                    siphon_amount: float = min(trade_pnl, excess_to_siphon)
                                    siphoning_reserve += siphon_amount
                                    excess_to_siphon -= siphon_amount
                                    logger.info(f"💰 [SAFE] Market P&L siphoned: +{siphon_amount:.4f} USDT (Total reserve: {siphoning_reserve:.2f})")
                                    state.update({"siphoning_reserve": siphoning_reserve})
                                    await save_json(state_file_path, state)
                                    await notifier.send_message(f"💰 <b>SAFE</b>: +{siphon_amount:.4f} USDT from {pos_side} MARKET")

                if limit_enabled and limit_stats["attempted"] > 0:
                    avg_improvement = limit_stats["total_improvement_pct"] / limit_stats["filled"] if limit_stats["filled"] > 0 else 0
                    logger.info(f"📊 Limit Stats: attempted={limit_stats['attempted']}, filled={limit_stats['filled']}, fallback={limit_stats['fallback']}, avg_gain={avg_improvement:+.3f}%, total_profit={limit_stats['total_profit_usdt']:+.2f} USDT")

                # Пересчитываем позиции и доли после выполнения сделок
                if paper_mode:
                    raw_positions = paper_state.get("positions", {})
                    positions = {k: v for k, v in raw_positions.items()}
                    l_entry = paper_state.get("long_entry_price", price)
                    s_entry = paper_state.get("short_entry_price", price)
                else:
                    raw_positions = await connector.get_positions()
                    positions = {k: v["qty"] for k, v in raw_positions.items()}
                    l_entry = raw_positions.get(f"{base_ticker}_LONG", {}).get("entry_price", 0.0)
                    s_entry = raw_positions.get(f"{base_ticker}_SHORT", {}).get("entry_price", 0.0)

                new_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                 base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets,
                                                 long_entry_price=l_entry, short_entry_price=s_entry,
                                                 initial_capital=initial_tpv)

                # Сводное уведомление после ребаланса
                summary_msg = (
                    f"<b>✅ Rebalance #{cycles} Complete</b>: <code>{base_ticker}</code>\n"
                    f"New Shares: L:{new_calc.share_long_pct:.1f}% S:{new_calc.share_short_pct:.1f}% V:{new_calc.share_virt_pct:.1f}%\n"
                    f"TPV: <code>{new_calc.total_tpv:.2f} USDT</code>"
                )
                if paper_mode:
                    summary_msg += f"\nBalance: <code>{paper_state['balance']:.2f}</code> USDT"
                await notifier.send_message(summary_msg)

                virt_basis_price = price
                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})
                await save_json(state_file_path, state)
            
            if i % 100 == 0:
                total_balance = bnb_balance = None
                try:
                    m_info = await connector.get_margin_ratio()
                    total_balance = m_info.get("total_margin_balance")
                    bnb_balance = await connector.get_bnb_balance()
                except: pass
                cycles = state.get("rebalance_cycles", 0)
                await notifier.send_status(base_ticker, calc.total_tpv, calc.total_tpv - state.get("initial_tpv", calc.total_tpv), cycles, siphoning_reserve, total_balance, bnb_balance)

        except Exception as e:
            logger.error(f"Error in cycle: {e}")
            await asyncio.sleep(10)
        await asyncio.sleep(check_interval)
        i += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    config_base = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, "r", encoding="utf-8") as f: cfg = json.load(f)
    base_ticker = args.ticker if args.ticker else cfg.get("base_ticker", "BTCUSDT")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rebalance_{base_ticker}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(base_ticker)
    instance_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"state_{base_ticker}.json"))
    instance_paper_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"paper_state_{base_ticker}.json"))
    logger.info(f"💾 State files: REAL={instance_state_file}, PAPER={instance_paper_state_file}")
    
    api_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))
    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))
    asyncio.run(rebalance_loop(connector, args.config, instance_state_file, instance_paper_state_file, logger, ticker_override=base_ticker))
