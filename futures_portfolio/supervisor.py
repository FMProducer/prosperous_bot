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
from backtest_rebalance import run_backtest
from storage import safe_load_json, safe_save_json

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

def read_shared_config(path: str) -> Dict[str, Any]:
    """Чтение общего конфига без использования блокировок."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Critical: Shared config read failed: {e}")
        return {}

async def get_running_bots_info() -> Dict[str, dict]:
    """Получает детальную информацию о запущенных ботах из PM2, включая аптайм"""
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
                    # Извлекаем pm_uptime (timestamp старта в мс)
                    uptime_ms = app.get('pm2_env', {}).get('pm_uptime', 0)
                    bots[ticker] = {
                        "name": app['name'], 
                        "paper": is_paper, 
                        "status": app['pm2_env']['status'],
                        "uptime": uptime_ms
                    }
        return bots
    except Exception as e:
        logger.error(f"Failed to get PM2 list: {e}")
        return {}

async def stop_bot(ticker: str):
    logger.info(f"🛑 Stopping bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    
    # 1. Сначала принудительно останавливаем процесс, чтобы он не открывал новые позиции
    try:
        proc = await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
        await proc.wait()
        await asyncio.sleep(1) # Даем время на завершение процесса
    except: pass
    
    # 2. Затем запускаем скрипт закрытия позиций (он отработает в отдельном процессе и завершится)
    try:
        cmd = f'"{sys.executable}" main.py --config config.json --ticker {ticker} --stop'
        proc = await asyncio.create_subprocess_shell(cmd)
        await asyncio.wait_for(proc.wait(), timeout=30)
        await asyncio.sleep(1) # Даем время Binance обработать ордера
    except Exception as e:
        logger.warning(f"Stop command failed for {ticker}: {e}")

async def start_bot(ticker: str, paper: bool = False):
    mode_str = "PAPER" if paper else "REAL"
    logger.info(f"🚀 Launching {mode_str} bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    paper_flag = "--paper" if paper else ""
    
    # Удаляем старый если есть
    try:
        proc = await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
        await proc.wait()
    except: pass
    
    # Формируем команду запуска
    cmd = f'pm2 start main.py --name bot-{short_name} --cwd "{CURRENT_DIR}" --update-env --interpreter "{sys.executable}" -- --config config.json --ticker {ticker} {paper_flag}'
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
    
    # 1. Сбор реальности
    running_info = await get_running_bots_info()
    
    # 2. Обработка сигналов от ботов (пассивный мониторинг)
    signal_dir = Path("signals")
    if not signal_dir.exists(): signal_dir.mkdir(parents=True)
    
    new_black_list = set(config.get("black_list", []))
    for sig_file in signal_dir.glob("*.flag"):
        try:
            parts = sig_file.stem.split("_")
            if len(parts) >= 2:
                sig_type, ticker = parts[0], parts[1]
                if sig_type == "stop":
                    logger.warning(f"⚠️ {ticker} sent emergency STOP signal.")
                    await stop_bot(ticker)
                    new_black_list.add(ticker)
                elif sig_type == "exit":
                    logger.info(f"✅ {ticker} sent profitable EXIT signal.")
                    await stop_bot(ticker)
            sig_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error processing signal {sig_file}: {e}")

    # 3. Анализ и Рейтинг
    logger.info("Starting scanner analysis...")
    try:
        # Добавляем таймаут для безопасности
        scanner_results = await asyncio.wait_for(run_scanner(quiet=True, min_volume=10_000_000), timeout=60)
        scanner_tickers = [r['symbol'] for r in scanner_results]
        logger.info(f"Scanner found {len(scanner_tickers)} tickers.")
    except Exception as e:
        logger.error(f"Scanner failed or timed out: {e}")
        scanner_tickers = []
    
    eval_pool = [t for t in list(set(running_info.keys()) | set(scanner_tickers)) if t not in new_black_list]
    
    perf_dict = {}
    logger.info(f"Starting backtest analysis for {len(eval_pool)} tickers...")
    for symbol in eval_pool:
        try:
            # Добавляем таймаут для каждого тикера
            res = await asyncio.wait_for(run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=backtest_days, quiet=True), timeout=30)
            if res and res.get("profit_pct", 0) > 0 and res.get("siphoning_reserve", 0) > 0:
                # Сохраняем и профит, и сейф для анализа
                perf_dict[symbol] = {
                    "profit": res.get("profit_pct", 0),
                    "safe": res.get("siphoning_reserve", 0)
                }
        except Exception as e:
            logger.warning(f"Backtest failed for {symbol}: {e}")
            pass
    logger.info("Backtest analysis complete.")

    all_sorted = sorted(perf_dict.keys(), key=lambda x: perf_dict[x]['profit'], reverse=True)
    top_10 = all_sorted[:max_bots]
    
    # 4. Выбор Чемпионов для REAL (Математика Чемпионов)
    ready_pool = []
    now_ms = time.time() * 1000
    probation_ms = probation_hours * 3600 * 1000

    for ticker in all_sorted:
        bot_info = running_info.get(ticker)
        perf = perf_dict.get(ticker, {})
        backtest_profit = perf.get("profit", 0)
        backtest_safe = perf.get("safe", 0)
        
        # Только если бэктест прибыльный И зафиксирован SAFE
        if backtest_profit <= 0 or backtest_safe <= 0:
            continue

        # Если бот уже в Реале, он остается кандидатом
        is_already_real = bot_info and not bot_info['paper'] and bot_info['status'] == 'online'
        if is_already_real:
            ready_pool.append(ticker)
            continue

        # Если бот в Песочнице, проверяем время
        is_in_paper = bot_info and bot_info['paper'] and bot_info['status'] == 'online'
        if is_in_paper:
            uptime_ms = bot_info.get('uptime', 0)
            elapsed_ms = now_ms - uptime_ms
            
            if elapsed_ms >= probation_ms:
                logger.info(f"✅ {ticker} passed probation ({elapsed_ms/3600000:.2f}h) with profit {backtest_profit:.2f}% and SAFE {backtest_safe:.2f}")
                ready_pool.append(ticker)
            else:
                logger.info(f"⏳ {ticker} is maturing ({elapsed_ms/3600000:.2f}h / {probation_hours:.2f}h)")

    # Сортируем ВСЕХ готовых по текущей прибыли бэктеста
    ready_pool.sort(key=lambda x: perf_dict.get(x, {}).get('profit', 0), reverse=True)
    target_real_bots = ready_pool[:max_real_slots]
    
    logger.info(f"🎯 Target REAL bots (Best of Ready): {target_real_bots}")
    
    # 5. ИСПОЛНЕНИЕ
    for ticker, info in running_info.items():
        if not info['paper'] and ticker not in target_real_bots:
            logger.info(f"🚫 Removing OLD REAL bot: {ticker}")
            await stop_bot(ticker)

    for ticker in target_real_bots:
        if ticker not in running_info or running_info[ticker]['paper']:
            logger.info(f"🏆 Promoting to REAL: {ticker}")
            await stop_bot(ticker)
            await start_bot(ticker, paper=False)

    current_pm2 = await get_running_bots_info()
    for ticker in (set(current_pm2.keys()) - set(top_10)):
        if ticker not in target_real_bots:
            await stop_bot(ticker)

    for ticker in top_10:
        if ticker in target_real_bots: continue
        if ticker not in current_pm2 or not current_pm2[ticker]['paper']:
            await start_bot(ticker, paper=True)

    # 6. Финализация конфига (принудительное обновление)
    config["tickers"] = top_10
    config["live_swarm"] = sorted(target_real_bots)
    config["base_ticker"] = all_sorted[0] if all_sorted else "BTCUSDT"
    config["black_list"] = sorted(list(new_black_list))
    
    logger.info(f"Writing to config: tickers={len(config['tickers'])}, swarm={config['live_swarm']}")
    await safe_save_json(CONFIG_PATH, config)
    
    # Релоад PM2
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
