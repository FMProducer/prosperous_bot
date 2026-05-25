import asyncio
import json
import logging
import os
import time
import random
import sys
import shutil
from typing import Dict, List, Set, Any
from dotenv import load_dotenv
from pathlib import Path

# Загрузка окружения
load_dotenv()

from rank_tickers import main as run_scanner
from storage import safe_load_json, safe_save_json
from connector import BinanceConnector

# Настройка логирования
BASE_PATH = Path(__file__).resolve().parent
log_dir = BASE_PATH / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

def setup_logger():
    l = logging.getLogger("Supervisor")
    l.setLevel(logging.INFO)
    # Clear existing handlers if any
    if l.handlers:
        l.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    
    # File Handler
    fh = logging.FileHandler(log_dir / "supervisor.log", encoding="utf-8")
    fh.setFormatter(formatter)
    l.addHandler(fh)
    
    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    l.addHandler(sh)
    
    # Prevent propagation to root logger
    l.propagate = False
    return l

logger = setup_logger()

CONFIG_PATH = str(BASE_PATH / "config.json")

async def get_running_bots_info() -> Dict[str, dict]:
    try:
        proc = await asyncio.create_subprocess_shell(
            "pm2 jlist",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if not stdout: return {}
        data = json.loads(stdout.decode('utf-8', errors='replace'))
        bots = {}
        for app in data:
            name = app['name']
            if name.startswith(('paper-', 'real-')):
                args = app.get('pm2_env', {}).get('args', [])
                ticker = next((args[i+1] for i, a in enumerate(args) if a == '--ticker'), None)
                is_paper = name.startswith('paper-')
                if ticker:
                    bots[f"{'p' if is_paper else 'r'}_{ticker}"] = {"name": name, "paper": is_paper}
        return bots
    except Exception as e:
        logger.error(f"Failed to get PM2 list: {e}")
        return {}

async def stop_bot(ticker: str, is_paper: bool = False):
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    try:
        await (await asyncio.create_subprocess_shell(f"pm2 delete {proc_name}")).wait()
    except: pass

async def enforce_swarm_consistency(connector: BinanceConnector, config: dict):
    logger.info("🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.")
    
    live_swarm_tickers = set(config.get("live_swarm", []))
    active_positions = await connector.get_positions() # Получаем все открытые позиции с биржи

    to_close_tickers = set()

    # Если live_swarm пуст, закрываем все позиции
    if not live_swarm_tickers:
        if active_positions:
            logger.warning("⚠️ live_swarm is empty. All open positions on exchange will be closed!")
            for pos_key in active_positions.keys():
                ticker = pos_key.split('_')[0]
                to_close_tickers.add(ticker)
    else:
        # Если live_swarm не пуст, закрываем позиции по тикерам, которых нет в live_swarm
        for pos_key in active_positions.keys():
            ticker = pos_key.split('_')[0]
            if ticker not in live_swarm_tickers:
                logger.warning(f"⚠️ Unauthorized position found for {ticker} (not in live_swarm). It will be closed!")
                to_close_tickers.add(ticker)

    for ticker in to_close_tickers:
        logger.info(f"🧹 Closing all positions for unauthorized ticker: {ticker}")
        cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
        await (await asyncio.create_subprocess_shell(cmd_stop)).wait()
        logger.info(f"✅ Positions for {ticker} closed successfully.")

async def reset_real_state(ticker: str, config: dict):
    """Архивирует текущее состояние реального бота и сбрасывает его перед новым запуском."""
    base_state_path = BASE_PATH / f"real_state_{ticker}.json"
    shadow_state_path = BASE_PATH / f"shadow_state_{ticker}.json"
    history_dir = BASE_PATH / "history"
    history_dir.mkdir(exist_ok=True)
    
    # Архивируем, если файлы существуют
    for path in [base_state_path, shadow_state_path]:
        if path.exists():
            archive_path = history_dir / f"archive_{ticker}_{int(time.time())}_{path.name}"
            shutil.copy(path, archive_path)
            logger.info(f"💾 Archived state for {ticker}: {path.name} -> {archive_path.name}")
            os.remove(path)
            logger.info(f"🗑️ Deleted stale state file: {path.name}")

    # Создаем базовые пустые файлы для чистого старта
    portfolio_cfg = config.get("portfolios", [{}])[0]
    initial_capital = portfolio_cfg.get("initial_capital", 100.0)
    
    await safe_save_json(str(base_state_path), {
        "virt_qty": 0.0,
        "base_ticker": ticker,
        "siphoning_reserve": 0.0,
        "balance": initial_capital,
        "initial_tpv": 0.0,
        "reference_tpv": 0.0,
        "tpv_ath": 0.0,
        "rebalance_cycles": 0,
        "last_rebalance_price": 0.0,
        "started_at": time.time()
    })
    
    await safe_save_json(str(shadow_state_path), {
        "balance": initial_capital,
        "positions": {f"{ticker}_LONG": 0.0, f"{ticker}_SHORT": 0.0},
        "last_price": 0.0,
        "base_ticker": ticker,
        "long_entry_price": 0.0,
        "short_entry_price": 0.0
    })
    logger.info(f"✨ Reset real state files for {ticker} to clean initial values.")

async def start_bot(ticker: str, is_paper: bool = True, config: dict = None):
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    mode_flag = "--paper" if is_paper else "--real"
    
    # Перед запуском реального бота сбрасываем состояние
    if not is_paper and config:
        await reset_real_state(ticker, config)

    cmd = f'pm2 start main.py --name "{proc_name}" --cwd "{BASE_PATH}" --update-env --interpreter "{sys.executable}" --instances 1 -- --config config.json --ticker {ticker} {mode_flag}'
    await (await asyncio.create_subprocess_shell(cmd)).wait()

async def get_bot_efficiency(ticker: str, config: dict) -> dict:
    """
    Строгий расчет эффективности на основе непрерывного трека инкубатора.
    Обеспечивает доктрину параллельного слежения без рассинхронизации.
    """
    state = await safe_load_json(str(BASE_PATH / f"paper_state_{ticker}.json"), {})
    min_cycles = config.get("min_cycles_for_rank", 10)

    if not state:
        return {
            "profit": 0.0,
            "cycles": 0,
            "eff": 0.0,
            "trailing_stop_paper_timeout_end": 0.0
        }

    profit_usdt = state.get("last_profit", 0.0)
    cycles = state.get("rebalance_cycles", 0)

    return {
        "profit": profit_usdt,
        "cycles": cycles,
        "eff": profit_usdt / max(cycles, min_cycles),
        "trailing_stop_paper_timeout_end": state.get("trailing_stop_paper_timeout_end", 0.0)
    }

async def manage_swarm():
    logger.info("--- Starting Pure Live Supervisor (Strict Scanner Sync) ---")
    config = await safe_load_json(CONFIG_PATH, {})
    if not config: return

    # Инициализация коннектора для операций на бирже
    api_key = os.environ.get("BINANCE_API_KEY", config.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", config.get("secret_key", ""))
    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=config.get("testnet", True))
    await connector.verify_connection()

    # !!! [SAFETY GATE] !!!
    # 0. Принудительная проверка согласованности роя с биржевыми позициями
    await enforce_swarm_consistency(connector, config)

    # 1. Запуск сканера (ЖЕСТКО 20M)
    logger.info("🔍 Running ticker scanner (Min Vol: 20M)...")
    try:
        scanner_results = await asyncio.wait_for(run_scanner(quiet=True, min_volume=20_000_000), timeout=60)
        if not scanner_results:
            logger.error("❌ Scanner returned nothing. Aborting cycle to prevent config wipe.")
            return
    except Exception as e:
        logger.error(f"❌ Scanner failed: {e}")
        return

    # 2. Формирование НОВОГО списка Инкубатора (Strictly from Scanner)
    final_incubator = []
    limit_bots = config.get("max_bots", 20)
    
    # Persistent Toxic Blacklist Logic
    now = time.time()
    toxic_blacklist = config.get("toxic_blacklist", {})
    # Prune expired
    toxic_blacklist = {s: exp for s, exp in toxic_blacklist.items() if exp > now}
    
    cooldown_days = config.get("toxic_cooldown_days", config.get("scanner_period_days", 1.0))
    cooldown_sec = cooldown_days * 86400

    for r in scanner_results:
        symbol = r['symbol']
        if r.get('is_toxic'):
            expiry = now + cooldown_sec
            toxic_blacklist[symbol] = expiry
            logger.info(f"🚫 {symbol} marked toxic. Blacklisted until {time.ctime(expiry)}")

    config["toxic_blacklist"] = toxic_blacklist

    for r in scanner_results:
        symbol = r['symbol']
        if len(final_incubator) >= limit_bots: break
        
        if symbol in toxic_blacklist:
            logger.info(f"⏳ {symbol} is in Toxic Quarantine. Skipping.")
            continue
            
        final_incubator.append(symbol)
        logger.info(f"➕ Added to Incubator: {symbol} (Cycles: {r.get('cycles')})")

    # 3. Выбор Чемпионов для REAL
    # Собираем данные по эффективности для тех, кто прошел фильтры
    perf_map = {}
    for ticker in final_incubator:
        perf_map[ticker] = await get_bot_efficiency(ticker, config)
    
    # Сортировка по эффективности
    replacement_threshold = config.get("replacement_efficiency_threshold_pct", 20.0)
    running_bots = await get_running_bots_info()
    current_real_tickers = [k.replace("r_", "") for k in running_bots.keys() if k.startswith("r_")]
    real_whitelist = set(config.get("real_whitelist", []))

    ready_pool = []
    all_evaluated_tickers = set(final_incubator).union(current_real_tickers) # Consider all candidates and existing real bots
    use_whitelist = config.get("use_real_whitelist", True)

    for ticker in all_evaluated_tickers:
        # Whitelist check: only apply to new candidates. Existing real bots are evaluated regardless of whitelist.
        if use_whitelist and ticker not in real_whitelist and ticker not in current_real_tickers:
            logger.info(f"🚫 Skipping {ticker}: Not in real_whitelist and not a currently running real bot.")
            continue

        p = perf_map.get(ticker) # Get performance data for incubator candidates. Could be None for existing real bots without paper history.

        # Explicitly handle existing real bots that have no corresponding paper history (i.e., not in perf_map)
        is_running_real = ticker in current_real_tickers
        if is_running_real and p is None:
            # This is a real bot currently running but has no valid paper history (e.g., failed scanner/incubator)
            # Assign a very low effective score to ensure it's replaced by any profitable paper bot.
            p = {
                "profit": 0.0,
                "cycles": 0,
                "eff": 0.0,
                "trailing_stop_paper_timeout_end": 0.0,
            }
            p['sort_eff'] = -1_000_000.0 # Extremely low score to ensure replacement
            perf_map[ticker] = p # Add this low-priority entry to perf_map so sort can use it
            ready_pool.append(ticker)
            logger.info(f"⚖️ Scored {ticker}: Running REAL bot with no valid paper history. Assigning lowest priority for replacement.")
            continue # Move to the next ticker

        if p is None:
            logger.warning(f"Skipping {ticker}: No performance data found. This should not happen for an incubator candidate.")
            continue

        # Check for trailing stop timeout from paper mode, which would prevent promotion
        if time.time() < p.get('trailing_stop_paper_timeout_end', 0.0):
            logger.info(f"⏳ Skipping {ticker}: Still in trailing stop paper timeout (paper end: {time.ctime(p['trailing_stop_paper_timeout_end'])}).")
            continue

        # Sticky logic (Drawdown Protection)
        is_sticky = False
        if is_running_real:
            real_state = await safe_load_json(str(BASE_PATH / f"real_state_{ticker}.json"), {})
            if real_state.get("total_pnl_pct", 0.0) < 0:
                is_sticky = True

        # Only profitable, or has enough history (for evaluation), or is sticky (for existing real bots)
        min_cycles_required = config.get("min_cycles_for_rank", 10)
        has_enough_history = p.get('cycles', 0) >= min_cycles_required
        profit_condition = p.get('profit', 0.0) > 0

        # Bot is considered for REAL if it's profitable AND has enough paper history OR is an existing real bot, OR it's sticky.
        if (profit_condition and (has_enough_history or is_running_real)) or is_sticky:
            pure_eff = p.get('eff', 0.0)
            sort_eff = pure_eff

            if is_sticky:
                sort_eff += 1_000_000.0 # High bonus for sticky bots
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} + Sticky (Total:{sort_eff:.2f})")
            elif is_running_real:
                # Give existing real bots a bonus to prevent excessive churn if they are performing
                sort_eff *= (1 + replacement_threshold / 100.0)
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} * ReplacementBonus (Total:{sort_eff:.2f})")
            else:
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} (Total:{sort_eff:.2f})") # New incubator candidate

            p['sort_eff'] = sort_eff
            perf_map[ticker] = p # Ensure updated p with sort_eff is in perf_map for sorting
            ready_pool.append(ticker)
        else:
            logger.info(f"❌ Skipping {ticker}: Not profitable ({p.get('profit', 0.0):.4f}) or insufficient history ({p.get('cycles', 0)}/{min_cycles_required}) and not sticky (running real: {is_running_real}).")

    ready_pool.sort(key=lambda x: perf_map.get(x, {}).get('sort_eff', -2_000_000.0), reverse=True) # Ensure low-score bots are at the end
    max_real_slots = max(0, config.get("max_bots", 10) - config.get("paper_mode_bots", 9))
    target_real_bots = ready_pool[:max_real_slots]
    # -------------------------------------------------------------------------
    # 3.5. [Authoritative Cleanup] Закрываем позиции на бирже для тех, кто не в live_swarm,  
    # даже если PM2 процесс уже не найден (застрявшие позиции).
    active_positions = await connector.get_positions()
    for pos_key in active_positions.keys():
        # pos_key имеет вид "SYMBOL_SIDE"
        ticker = pos_key.split('_')[0]
        if ticker not in target_real_bots:
            qty = active_positions[pos_key].get("qty", 0.0)
            if float(qty) != 0:
                logger.warning(f"🧹 Authoritative Cleanup: Found stray position for {ticker}. Closing it!")
                cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
                await (await asyncio.create_subprocess_shell(cmd_stop)).wait()

    # 4. Исполнение в PM2: Доктрина Параллельного Слежения (Защищенная версия)
    # -------------------------------------------------------------------------

    # Создаем мутабельное множество текущих запущенных ключей для исключения рассинхрона
    active_running_keys = set(running_bots.keys())

    # Сначала гасим тех, кто вылетел
    for key, info in list(running_bots.items()):
        ticker = key.split("_")[1]

        if key.startswith("p_") and ticker not in final_incubator:
            logger.info(f"🛑 Stopping Incubator (Out of Scanner): {ticker}")
            await stop_bot(ticker, is_paper=True)
            active_running_keys.discard(key)

        if key.startswith("r_") and ticker not in target_real_bots:
            logger.info(f"🛑 Stopping Combat REAL process: {ticker} (Rolling back to pure paper tracking)")
            await stop_bot(ticker, is_paper=False)
            active_running_keys.discard(key)

            # Экстренно закрываем позиции на бирже для этого тикера
            cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
            await (await asyncio.create_subprocess_shell(cmd_stop)).wait()

    # Пауза для стабильности дескрипторов PM2
    await asyncio.sleep(1.5)

    # ЗАПУСК: Инкубатор (PAPER) работает ВСЕГДА для всех тикеров из сканера
    for ticker in final_incubator:
        p_key = f"p_{ticker}"
        if p_key not in active_running_keys:
            logger.info(f"🚀 [A] Launching Continuous Incubator (PAPER): {ticker}")
            await start_bot(ticker, is_paper=True)
            active_running_keys.add(p_key) # Жестко фиксируем запуск локально

    # ЗАПУСК: Боевые боты (REAL) включаются ПАРАЛЛЕЛЬНО к бумажным
    for ticker in target_real_bots:
        r_key = f"r_{ticker}"
        if r_key not in active_running_keys:
            logger.info(f"🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): {ticker}")
            await start_bot(ticker, is_paper=False, config=config)
            active_running_keys.add(r_key) # Жестко фиксируем запуск локально

    # 5. Сохранение конфига
    config["tickers"] = sorted(final_incubator)
    config["live_swarm"] = sorted(target_real_bots)
    config["base_ticker"] = config["tickers"][0] if config["tickers"] else "BTCUSDT"
    await safe_save_json(CONFIG_PATH, config)
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
