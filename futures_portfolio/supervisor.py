import asyncio
import json
import logging
import os
import time
import random
import sys
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

async def start_bot(ticker: str, is_paper: bool = True):
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    mode_flag = "--paper" if is_paper else "--real"
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
    for ticker in final_incubator:
        if ticker not in real_whitelist: continue
        p = perf_map[ticker]
        if time.time() < p['trailing_stop_paper_timeout_end']: continue
        
        # Sticky logic (Drawdown Protection)
        is_running_real = ticker in current_real_tickers
        is_sticky = False
        if is_running_real:
            real_state = await safe_load_json(str(BASE_PATH / f"real_state_{ticker}.json"), {})
            if real_state.get("total_pnl_pct", 0.0) < 0:
                is_sticky = True
        
        # Только прибыльные или Sticky
        min_cycles_required = config.get("min_cycles_for_rank", 10)
        has_enough_history = p['cycles'] >= min_cycles_required

        # Бот допускается к оценке REAL, если он прошел карантин по циклам,
        # ЛИБО если он уже торгует в реале (is_running_real), чтобы не дергать процессы зря.
        if (p['profit'] > 0 and (has_enough_history or is_running_real)) or is_sticky:
            pure_eff = p['eff']
            sort_eff = pure_eff
            
            if is_sticky:
                sort_eff += 1000000.0
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} + Sticky (Total:{sort_eff:.2f})")
            elif is_running_real:
                sort_eff *= (1 + replacement_threshold / 100.0)
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} * ReplacementBonus (Total:{sort_eff:.2f})")
            else:
                logger.info(f"⚖️ Scored {ticker}: Pure:{pure_eff:.4f} (Total:{sort_eff:.2f})")
            
            p['sort_eff'] = sort_eff
            ready_pool.append(ticker)

    ready_pool.sort(key=lambda x: perf_map[x].get('sort_eff', 0.0), reverse=True)
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
            await start_bot(ticker, is_paper=False)
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
