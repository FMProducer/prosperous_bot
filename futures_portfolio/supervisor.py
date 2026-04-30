import asyncio
import json
import logging
import os
import time
import subprocess
from typing import Dict, List, Set
from dotenv import load_dotenv

# Загрузка окружения
load_dotenv()

from rank_tickers import main as run_scanner
from backtest_rebalance import run_backtest

# Настройка логирования
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "supervisor.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Supervisor")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")
DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"

async def safe_load_json(path: str, default: dict) -> dict:
    try:
        if not os.path.exists(path): return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

async def safe_save_json(path: str, data: dict):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)
    except Exception as e:
        logger.error(f"Save failed: {e}")

async def get_running_bots_info() -> Dict[str, dict]:
    """Получает детальную информацию о запущенных ботах из PM2"""
    try:
        proc = await asyncio.create_subprocess_shell(
            "pm2 jlist",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        bots = {}
        for app in data:
            if app['name'].startswith('bot-'):
                args = app.get('pm2_env', {}).get('args', [])
                ticker = None
                is_paper = "--paper" in args
                for i, arg in enumerate(args):
                    if arg == '--ticker' and i + 1 < len(args):
                        ticker = args[i+1]
                if ticker:
                    bots[ticker] = {"name": app['name'], "paper": is_paper, "status": app['pm2_env']['status']}
        return bots
    except Exception as e:
        logger.error(f"Failed to get PM2 list: {e}")
        return {}

async def stop_bot(ticker: str):
    logger.info(f"🛑 Stopping bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    try:
        cmd_stop = f"python main.py --config config.json --ticker {ticker} --stop"
        proc = await asyncio.create_subprocess_shell(cmd_stop)
        await asyncio.wait_for(proc.wait(), timeout=30)
    except: pass
    
    try:
        await (await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")).wait()
    except: pass

async def start_bot(ticker: str, paper: bool = False):
    mode_str = "PAPER" if paper else "REAL"
    logger.info(f"🚀 Launching {mode_str} bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    paper_flag = "--paper" if paper else ""
    await (await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")).wait()
    
    cmd = f"pm2 start main.py --name bot-{short_name} --cwd {CURRENT_DIR} --update-env --interpreter python -- --config config.json --ticker {ticker} {paper_flag}"
    proc = await asyncio.create_subprocess_shell(cmd)
    await proc.wait()

async def manage_swarm():
    logger.info("--- Starting Multi-Slot Supervisor Cycle ---")
    
    config = await safe_load_json(CONFIG_PATH, {})
    if not config: return
    
    max_bots = config.get("max_bots", 10)
    paper_mode_bots = config.get("paper_mode_bots", 9)
    max_real_slots = max(0, max_bots - paper_mode_bots)
    
    probation_hours = config.get("probation_period_days", 0.125) * 24
    backtest_days = config.get("backtest_period_days", 1.0)
    max_dd_limit = config.get("max_drawdown_limit", 15.0)
    
    # 1. Сбор реальности
    running_info = await get_running_bots_info()
    actual_running_tickers = set(running_info.keys())
    
    # 2. Проверка стоп-лоссов
    new_black_list = set(config.get("black_list", []))
    for ticker in actual_running_tickers:
        state = await safe_load_json(os.path.join(CURRENT_DIR, f"state_{ticker}.json"), {})
        if state.get("trailing_stop_triggered"):
            logger.warning(f"⚠️ {ticker} HIT STOP-LOSS!")
            await stop_bot(ticker)
            new_black_list.add(ticker)
            state["trailing_stop_triggered"] = False
            await safe_save_json(os.path.join(CURRENT_DIR, f"state_{ticker}.json"), state)

    # 3. Анализ и Рейтинг
    try:
        scanner_results = await run_scanner(quiet=True, min_volume=10_000_000)
        scanner_tickers = [r['symbol'] for r in scanner_results]
    except: scanner_tickers = []
    
    eval_pool = [t for t in list(actual_running_tickers | set(scanner_tickers)) if t not in new_black_list]
    
    perf_dict = {}
    for symbol in eval_pool:
        try:
            res = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=backtest_days, quiet=True)
            if res and res.get("profit_pct", 0) > 0:
                perf_dict[symbol] = res.get("profit_pct", 0)
        except: pass

    all_sorted = sorted(perf_dict.keys(), key=lambda x: perf_dict[x], reverse=True)
    top_10 = all_sorted[:max_bots]
    
    # 4. Выбор Чемпионов для REAL (Поддержка нескольких слотов)
    ready_for_real = []
    for ticker in all_sorted:
        state = await safe_load_json(os.path.join(CURRENT_DIR, f"state_{ticker}.json"), {})
        is_already_real = ticker in running_info and not running_info[ticker]['paper']
        
        started_at = state.get("started_at", 0)
        elapsed = (time.time() - started_at) / 3600 if started_at > 0 else 0
        has_profit = state.get("siphoning_reserve", 0.0) > 0
        
        if is_already_real or (has_profit and elapsed >= probation_hours):
            ready_for_real.append(ticker)

    # Выбираем N лучших чемпионов согласно лимиту слотов
    target_real_bots = ready_for_real[:max_real_slots]
    logger.info(f"🎯 Target REAL bots ({len(target_real_bots)}/{max_real_slots}): {target_real_bots}")
    
    # 5. ИСПОЛНЕНИЕ
    # Шаг А: Останавливаем всех реальных ботов, которые НЕ входят в новый список чемпионов
    for ticker, info in running_info.items():
        if not info['paper'] and ticker not in target_real_bots:
            logger.info(f"🚫 Removing OLD REAL bot: {ticker}")
            await stop_bot(ticker)

    # Шаг Б: Запускаем новых чемпионов в REAL
    for ticker in target_real_bots:
        is_running_real = ticker in running_info and not running_info[ticker]['paper']
        if not is_running_real:
            logger.info(f"🏆 Promoting to REAL: {ticker}")
            await stop_bot(ticker) # Гасим бумагу если была
            await start_bot(ticker, paper=False)

    # Шаг В: Управление Бумажным роем (остальные из ТОП-10)
    current_pm2 = await get_running_bots_info()
    # Останавливаем лишних (кто не в ТОП-10 и не в REAL)
    for ticker in (set(current_pm2.keys()) - set(top_10)):
        if ticker not in target_real_bots:
            await stop_bot(ticker)

    # Запускаем бумажных из ТОП-10
    for ticker in top_10:
        if ticker in target_real_bots: continue
        if ticker not in current_pm2 or not current_pm2[ticker]['paper']:
            await start_bot(ticker, paper=True)

    # 6. Финализация конфига
    config["tickers"] = top_10
    config["live_swarm"] = sorted(target_real_bots)
    config["base_ticker"] = all_sorted[0] if all_sorted else "BTCUSDT"
    config["black_list"] = sorted(list(new_black_list))
    
    await safe_save_json(CONFIG_PATH, config)
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
