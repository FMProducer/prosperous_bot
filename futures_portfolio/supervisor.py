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

# Настройка логирования
BASE_PATH = Path(__file__).resolve().parent
log_dir = BASE_PATH / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "supervisor.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Supervisor")

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
    paper_flag = "--paper" if is_paper else ""
    cmd = f'pm2 start main.py --name {proc_name} --cwd "{BASE_PATH}" --update-env --interpreter "{sys.executable}" -- --config config.json --ticker {ticker} {paper_flag}'
    await (await asyncio.create_subprocess_shell(cmd)).wait()

async def get_bot_efficiency(ticker: str, config: dict) -> dict:
    state = await safe_load_json(str(BASE_PATH / f"paper_state_{ticker}.json"), {})
    min_cycles = config.get("min_cycles_for_rank", 20)
    
    # [FIX] Гарантируем наличие всех ключей даже если файл отсутствует
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
    toxic_tickers = {r['symbol'] for r in scanner_results if r.get('is_toxic')}
    
    for r in scanner_results:
        symbol = r['symbol']
        if len(final_incubator) >= limit_bots: break
        if r.get('is_toxic'):
            logger.info(f"🚫 {symbol} is [X] Toxic. Skipping.")
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
        if p['profit'] > 0 or is_sticky:
            sort_eff = p['eff']
            if is_sticky: sort_eff += 1000000.0
            elif is_running_real: sort_eff *= (1 + replacement_threshold / 100.0)
            
            p['sort_eff'] = sort_eff
            ready_pool.append(ticker)

    ready_pool.sort(key=lambda x: perf_map[x].get('sort_eff', 0.0), reverse=True)
    max_real_slots = max(0, config.get("max_bots", 10) - config.get("paper_mode_bots", 9))
    target_real_bots = ready_pool[:max_real_slots]

    # 4. Исполнение в PM2
    # Останавливаем тех, кого нет в новом списке
    for key, info in running_bots.items():
        ticker = key.split("_")[1]
        if key.startswith("p_") and ticker not in final_incubator:
            logger.info(f"🛑 Stopping Incubator: {ticker}")
            await stop_bot(ticker, is_paper=True)
        if key.startswith("r_") and ticker not in target_real_bots:
            logger.info(f"🛑 Stopping Combat: {ticker}")
            await stop_bot(ticker, is_paper=False)
            await (await asyncio.create_subprocess_shell(f'"{sys.executable}" main.py --ticker {ticker} --stop')).wait()

    # Запускаем новых
    for ticker in final_incubator:
        if f"p_{ticker}" not in running_bots: await start_bot(ticker, is_paper=True)
    for ticker in target_real_bots:
        if f"r_{ticker}" not in running_bots: await start_bot(ticker, is_paper=False)

    # 5. Сохранение конфига
    config["tickers"] = sorted(final_incubator)
    config["live_swarm"] = sorted(target_real_bots)
    config["base_ticker"] = config["tickers"][0] if config["tickers"] else "BTCUSDT"
    await safe_save_json(CONFIG_PATH, config)
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
