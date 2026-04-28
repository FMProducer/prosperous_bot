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

async def get_running_bots() -> Set[str]:
    """Получает список тикеров запущенных ботов из PM2"""
    try:
        proc = await asyncio.create_subprocess_shell(
            "pm2 jlist",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        bots = set()
        for app in data:
            if app['name'].startswith('bot-'):
                # Пытаемся найти тикер в аргументах или восстановить из имени
                # В нашем случае имя это bot-shortname
                # Но лучше смотреть аргументы
                args = app.get('pm2_env', {}).get('args', [])
                for i, arg in enumerate(args):
                    if arg == '--ticker' and i + 1 < len(args):
                        bots.add(args[i+1])
        return bots
    except Exception as e:
        logger.error(f"Failed to get PM2 list: {e}")
        return set()

async def stop_bot(ticker: str):
    logger.info(f"🛑 Stopping bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    try:
        # 1. Закрываем позиции
        cmd_stop = f"python main.py --config config.json --ticker {ticker} --stop"
        proc = await asyncio.create_subprocess_shell(cmd_stop)
        await proc.wait()
        
        # 2. Удаляем из PM2
        proc_del = await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
        await proc_del.wait()
    except Exception as e:
        logger.error(f"Failed to stop {ticker}: {e}")

async def start_bot(ticker: str, paper: bool = False):
    mode_str = "PAPER" if paper else "REAL"
    logger.info(f"🚀 Launching {mode_str} bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    paper_flag = "--paper" if paper else ""
    cmd = f"pm2 start main.py --name bot-{short_name} --cwd {CURRENT_DIR} --update-env --interpreter python -- --config config.json --ticker {ticker} {paper_flag}"
    proc = await asyncio.create_subprocess_shell(cmd)
    await proc.wait()

async def manage_swarm():
    logger.info("--- Starting Incubator-Based Supervisor Cycle ---")
    
    config = await safe_load_json(CONFIG_PATH, {})
    if not config:
        logger.error("Config not found!")
        return
    
    max_bots = config.get("max_bots", 10)
    paper_mode_bots = config.get("paper_mode_bots", 2)
    probation_hours = config.get("probation_period_hours", 1)
    live_swarm = set(config.get("live_swarm", []))
    current_tickers_list = config.get("tickers", []) # Из конфига
    
    # 1. Синхронизация с реальностью (PM2)
    running_bots = await get_running_bots()
    logger.info(f"Currently running bots: {running_bots}")
    
    # 2. Проверка инкубатора (Promotion)
    new_live_swarm = live_swarm.copy()
    for ticker in running_bots:
        if ticker not in live_swarm:
            # Бот в инкубаторе (paper mode)
            state_path = os.path.join(CURRENT_DIR, f"state_{ticker}.json")
            state = await safe_load_json(state_path, {})
            
            siphoning_reserve = state.get("siphoning_reserve", 0.0)
            started_at = state.get("started_at", 0)
            elapsed_hours = (time.time() - started_at) / 3600 if started_at > 0 else 0
            
            logger.info(f"   > Incubator {ticker}: SAFE={siphoning_reserve:.2f}, Elapsed={elapsed_hours:.2f}h")
            
            if siphoning_reserve > 0 and elapsed_hours >= probation_hours:
                logger.info(f"🎓 PROMOTING {ticker} to REAL mode!")
                await stop_bot(ticker) # Останавливаем бумажного
                await start_bot(ticker, paper=False) # Запускаем реального
                new_live_swarm.add(ticker)
    
    # 3. Сканирование и поиск кандидатов
    try:
        scanner_results = await run_scanner(quiet=True, min_volume=10_000_000)
        scanner_tickers = [r['symbol'] for r in scanner_results]
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
        scanner_tickers = []
    
    # Пул для оценки (Текущие + Новые из сканера)
    eval_pool = list(set(running_bots) | set(scanner_tickers))
    
    ticker_performance = []
    for symbol in eval_pool:
        try:
            res = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=1, quiet=True)
            if res:
                strat_profit = res.get("profit_pct", 0)
                asset_perf_raw = res.get("asset_chg_pct", 0)
                max_dd = res.get("max_dd_pct", 0)
                alpha = strat_profit - asset_perf_raw
                
                # ФИЛЬТРЫ: Positive Alpha + Max Drawdown < 15%
                if alpha > 0 and max_dd <= 15.0:
                    ticker_performance.append({
                        "symbol": symbol,
                        "strategy_profit": strat_profit
                    })
        except: pass

    ticker_performance.sort(key=lambda x: x['strategy_profit'], reverse=True)
    top_performers_data = ticker_performance[:max_bots]
    top_performers_symbols = [t['symbol'] for t in top_performers_data]
    
    # Создаем словарь для быстрого доступа к профиту
    perf_dict = {t['symbol']: t['strategy_profit'] for t in top_performers_data}
    
    logger.info(f"Top performers by Strategy Profit: {top_performers_symbols}")

    # 4. Ротация (Уставшие боты)
    to_stop = running_bots - set(top_performers_symbols)
    for t in to_stop:
        logger.info(f"🥀 Bot {t} is TIRED (Dropped from Top). Removing from swarm.")
        await stop_bot(t)
        if t in new_live_swarm: new_live_swarm.remove(t)
    
    # 5. Запуск новых и Проверка лимитов реальной торговли
    remaining_running = running_bots - to_stop
    
    # Лимит реальных мест
    max_real_slots = max_bots - paper_mode_bots
    
    # Проверяем кандидатов на повышение из тех, кто уже запущен в бумаге
    for t in list(remaining_running - new_live_swarm):
        state_path = os.path.join(CURRENT_DIR, f"state_{t}.json")
        state = await safe_load_json(state_path, {})
        
        siphoning_reserve = state.get("siphoning_reserve", 0.0)
        started_at = state.get("started_at", 0)
        elapsed_hours = (time.time() - started_at) / 3600 if started_at > 0 else 0
        
        if siphoning_reserve > 0 and elapsed_hours >= probation_hours:
            if len(new_live_swarm) < max_real_slots:
                logger.info(f"🎓 PROMOTING {t} to REAL mode (Slot available)!")
                await stop_bot(t)
                await start_bot(t, paper=False)
                new_live_swarm.add(t)
            else:
                logger.info(f"⏳ {t} is ready for promotion, but REAL slots are full ({max_real_slots}). Waiting.")

    # Запуск абсолютно новых ботов (в инкубатор)
    to_start = [t for t in top_performers_symbols if t not in remaining_running]
    for t in to_start:
        if len(remaining_running) < max_bots:
            await start_bot(t, paper=True)
            remaining_running.add(t)
        else:
            logger.info(f"Swarm full. Skipping {t}.")

    # 6. Обновление конфига (Сортировка по профиту!)
    # Сортируем итоговый список тикеров по их производительности из бэктеста
    final_tickers = sorted(list(remaining_running), key=lambda x: perf_dict.get(x, -999), reverse=True)
    
    config["tickers"] = final_tickers
    config["live_swarm"] = sorted(list(new_live_swarm))
    config["base_ticker"] = final_tickers[0] if final_tickers else "BTCUSDT"
    
    await safe_save_json(CONFIG_PATH, config)
    await asyncio.create_subprocess_shell("pm2 save")
    logger.info("Swarm cycle complete.")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
