import asyncio
import json
import logging
import os
from typing import Dict, List
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from connector import BinanceConnector
from calculator import PortfolioCalculator
from executor import PortfolioExecutor
from notifier import TelegramNotifier

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

async def rebalance_loop(connector: BinanceConnector, config_path: str, state_file_path: str, paper_state_file_path: str, logger: logging.Logger):
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
    state = load_json(state_file_path, {
        "virt_basis_price": 0.0,
        "virt_allocated_usdt": 0.0,
        "base_ticker": base_ticker,
        "siphoning_reserve": 0.0,
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,  # Фиксированная база для гистерезиса
        "tpv_ath": 0.0
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
        save_json(state_file_path, state)

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
            "base_ticker": base_ticker
        }
        paper_state = load_json(paper_state_file_path, default_paper_state)
        
        if paper_state.get("base_ticker") != base_ticker:
            logger.info(f"Ticker in paper state changed from {paper_state.get('base_ticker')} to {base_ticker}. Resetting.")
            paper_state["last_price"] = 0.0
            paper_state["base_ticker"] = base_ticker
            
        if "positions" not in paper_state: paper_state["positions"] = {}
        if f"{base_ticker}_LONG" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_LONG"] = 0.0
        if f"{base_ticker}_SHORT" not in paper_state["positions"]: paper_state["positions"][f"{base_ticker}_SHORT"] = 0.0
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
            # В случае ошибки сети или API лучше перестраховаться и не запускаться на реале, 
            # либо попробовать позже. Но для старта - лучше упасть.
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
                positions = paper_state["positions"]
            else:
                real_equity = await connector.get_free_balance()
                positions = await connector.get_positions()
            
            # Ограничение капитала, если задано
            if max_capital_usdt > 0:
                real_equity = min(real_equity, max_capital_usdt)

            if virt_basis_price == 0 or initial_tpv == 0:
                if virt_basis_price == 0:
                    virt_basis_price = price
                    virt_allocated_usdt = real_equity * targets["VIRTUAL"]["share"]
                if initial_tpv == 0:
                    temp_calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt,
                                                 base_ticker=base_ticker, siphoning_reserve=0.0)
                    initial_tpv = temp_calc.tpv
                    reference_tpv = initial_tpv
                    logger.info(f"Initialized TPV base: {initial_tpv:.2f}")

                state.update({
                    "virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt,
                    "base_ticker": base_ticker, "siphoning_reserve": siphoning_reserve,
                    "initial_tpv": initial_tpv, "reference_tpv": reference_tpv
                })
                save_json(state_file_path, state)

            calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets)
            
            if tpv_ath == 0 or calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv
                state["tpv_ath"] = tpv_ath
                save_json(state_file_path, state)

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
                    if paper_mode: save_json(paper_state_file_path, paper_state)
                    logger.info("Positions closed. Bot stopped.")
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

            if siphoning_threshold_pct > 0:
                if calc.tpv > initial_tpv * (1 + siphoning_threshold_pct / 100):
                    profit = calc.tpv - initial_tpv
                    to_reinvest = profit * reinvestment_ratio
                    to_reserve = profit - to_reinvest
                    siphoning_reserve += to_reserve
                    initial_tpv += to_reinvest
                    msg = f"Reserve updated: +{to_reserve:.2f} USDT"
                    logger.info(f"!!! [SAFE] {msg}")
                    await notifier.send_message(f"💰 <b>SAFE</b>: {msg} ({base_ticker})")
                    calc = PortfolioCalculator(positions, price, real_equity, virt_basis_price, virt_allocated_usdt, 
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)
                    state.update({"siphoning_reserve": siphoning_reserve, "initial_tpv": initial_tpv})
                    save_json(state_file_path, state)

            current_pos_sum = abs(positions.get(f"{base_ticker}_LONG", 0)) + abs(positions.get(f"{base_ticker}_SHORT", 0))
            is_first_run = current_pos_sum == 0
            is_extreme = calc.share_long_pct > 100 or calc.share_short_pct > 100

            if i % 5 == 0:
                 res_str = f" | SAFE:{siphoning_reserve:.2f}" if siphoning_reserve > 0 else ""
                 logger.info(f"Heartbeat: TPV={calc.total_tpv:.2f}{res_str} | {base_ticker}={price:.6g} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}%")

            current_threshold = -1.0 if (is_first_run or is_extreme) else threshold
            if not (is_first_run or is_extreme) and reference_tpv > 0 and calc.tpv < reference_tpv:
                current_threshold *= 2.0

            deviations = calc.calculate_deviations(targets, current_threshold, ignore_limits=(current_threshold < 0))
            if deviations:
                # Внедряем Notional Value Guard (проверка минимальной стоимости ордера)
                min_notional = portfolio_cfg.get("min_notional_usdt", 6.0)
                filtered_deviations = [d for d in deviations if abs(d["diff_usdt"]) >= min_notional]
                
                if not filtered_deviations:
                    if i % 10 == 0:
                        logger.info(f"Rebalance needed, but all orders are too small (< {min_notional} USDT). Skipping.")
                    continue

                logger.info(f"Rebalance needed. Shares: L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}%")
                filtered_deviations.sort(key=lambda x: x["diff_usdt"])

                # Извлекаем параметры лимитных ордеров из конфига
                limit_enabled, limit_offset, limit_timeout = PortfolioExecutor(connector).get_limit_order_params(portfolio_cfg)

                # Статистика для сбора данных
                limit_stats = {"attempted": 0, "filled": 0, "fallback": 0, "total_profit_usdt": 0.0, "total_improvement_pct": 0.0}

                for dev in filtered_deviations:
                    key = dev["symbol"]
                    pos_side = key.split('_')[1]
                    order_qty = dev["diff_usdt"] / price
                    side = ("BUY" if dev["diff_usdt"] > 0 else "SELL") if pos_side == "LONG" else ("SELL" if dev["diff_usdt"] > 0 else "BUY")
                    reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")
                    step_size = step_sizes.get(key.split('_')[0], 0.0)

                    if paper_mode:
                        qty_rounded = PortfolioExecutor(None).round_quantity(abs(order_qty), step_size)
                        if qty_rounded > 0:
                            paper_state["positions"][f"{base_ticker}_{pos_side}"] += (qty_rounded if side == ("BUY" if pos_side == "LONG" else "SELL") else -qty_rounded)
                            paper_state["positions"][f"{base_ticker}_{pos_side}"] = max(0, paper_state["positions"][f"{base_ticker}_{pos_side}"])
                            logger.info(f"[PAPER] Order: {side} {qty_rounded:.6f} {key}")
                            save_json(paper_state_file_path, paper_state)
                    else:
                        # Используем Limit + Fallback если включено в конфиге
                        if limit_enabled:
                            limit_stats["attempted"] += 1
                            result = await PortfolioExecutor(connector).execute_limit_with_fallback(
                                symbol=key.split('_')[0],
                                qty=abs(order_qty),
                                side=side,
                                step_size=step_size,
                                reduce_only=reduce_only,
                                position_side=pos_side,
                                offset_pct=limit_offset,
                                timeout_sec=limit_timeout
                            )
                            # Логирование результата со статистикой
                            if result["status"] == "SUCCESS_LIMIT":
                                limit_stats["filled"] += 1
                                limit_stats["total_profit_usdt"] += result.get("profit_usdt", 0)
                                limit_stats["total_improvement_pct"] += result.get("price_improvement_pct", 0)
                                logger.info(f"✅ Limit FILLED: {result['filled_qty']} @ {result['avg_price']:.6f} | gain={result['price_improvement_pct']:+.3f}% ({result['profit_usdt']:+.2f} USDT)")
                            elif result["status"] == "SUCCESS_FALLBACK":
                                limit_stats["fallback"] += 1
                                if result.get("limit_filled_qty", 0) > 0:
                                    limit_stats["filled"] += 1
                                    limit_stats["total_improvement_pct"] += result.get("limit_price_improvement_pct", 0)
                                    logger.info(f"⚡ Limit+Fallback: {result['limit_filled_qty']} @ {result['limit_avg_price']:.6f} (gain={result.get('limit_price_improvement_pct', 0):+.3f}%) + market")
                                else:
                                    logger.info(f"⚡ Fallback (market only): timeout")
                            elif result["status"] in ("ERROR_FALLBACK", "ERROR"):
                                logger.warning(f"❌ Order with issues: {result.get('error', result.get('message', 'unknown'))}")
                        else:
                            # Старая логика — только market
                            await PortfolioExecutor(connector).execute_market_order(key, abs(order_qty), side, step_size, reduce_only, pos_side)

                # Итоговая статистика по ребалансировке
                if limit_enabled and limit_stats["attempted"] > 0:
                    avg_improvement = limit_stats["total_improvement_pct"] / limit_stats["filled"] if limit_stats["filled"] > 0 else 0
                    logger.info(f"📊 Limit Stats: attempted={limit_stats['attempted']}, filled={limit_stats['filled']}, fallback={limit_stats['fallback']}, avg_gain={avg_improvement:+.3f}%, total_profit={limit_stats['total_profit_usdt']:+.2f} USDT")
                
                virt_basis_price = price
                virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                state.update({"virt_basis_price": virt_basis_price, "virt_allocated_usdt": virt_allocated_usdt})
                save_json(state_file_path, state)
            
            if i % 100 == 0:
                total_balance = None
                bnb_balance = None
                try:
                    # Пытаемся получить реальный баланс аккаунта даже в бумажном режиме
                    m_info = await connector.get_margin_ratio()
                    total_balance = m_info.get("total_margin_balance")
                    bnb_balance = await connector.get_bnb_balance()
                except Exception as e:
                    logger.warning(f"Failed to fetch account info for status: {e}")
                
                await notifier.send_status(
                    config_base, 
                    calc.total_tpv, 
                    calc.total_tpv - state.get("initial_tpv", calc.total_tpv), 
                    i,
                    total_balance,
                    bnb_balance
                )

        except Exception as e:
            err_msg = f"Error in cycle: {e}"
            logger.error(err_msg)
            import traceback
            logger.error(traceback.format_exc())
            if i % 10 == 0: # Не спамим ошибками связи каждую секунду
                await notifier.send_message(f"❌ <b>Error</b>: {err_msg} ({base_ticker})")

        await asyncio.sleep(check_interval)
        i += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None, help="Override base_ticker from config")
    args = parser.parse_args()

    config_base = os.path.splitext(os.path.basename(args.config))[0]
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Приоритет: аргумент командной строки > конфиг
    base_ticker = args.ticker if args.ticker else cfg.get("base_ticker", "BTCUSDT")
    
    # Индивидуальные логи для каждого тикера
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rebalance_{base_ticker}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger(config_base)

    # Индивидуальные файлы состояния
    instance_state_file = os.path.join(os.path.dirname(__file__), f"state_{config_base}.json")
    instance_paper_state_file = os.path.join(os.path.dirname(__file__), f"paper_state_{config_base}.json")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = os.environ.get("BINANCE_API_KEY", cfg.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", cfg.get("secret_key", ""))

    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))
    asyncio.run(rebalance_loop(connector, args.config, instance_state_file, instance_paper_state_file, logger))
