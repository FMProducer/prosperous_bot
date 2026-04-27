import asyncio
import json
import logging
import os
import time
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
ECOSYSTEM_PATH = os.path.join(CURRENT_DIR, "ecosystem.config.js")
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

async def close_bot_positions(ticker: str):
    """Остановка бота и закрытие его позиций"""
    logger.info(f"🛑 Terminating and closing positions for {ticker}...")
    try:
        short_name = ticker.replace('USDT', '').lower()
        # 1. Закрываем позиции (main.py --stop)
        cmd = f"python main.py --config config.json --ticker {ticker} --stop"
        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()
        
        # 2. Удаляем из PM2
        proc_del = await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
        await proc_del.wait()
    except Exception as e:
        logger.error(f"Failed to close {ticker}: {e}")

async def manage_swarm():
    logger.info("--- Starting Strategy-Based Supervisor Cycle (1-Day Window) ---")
    
    config = await safe_load_json(CONFIG_PATH, {})
    if not config:
        logger.error("Config not found!")
        return
    
    current_tickers = config.get("tickers", [])
    max_bots = config.get("max_bots", 10)
    
    # 1. Сканирование рынка (Discovery)
    try:
        scanner_results = await run_scanner(quiet=True, min_volume=50_000_000)
        scanner_tickers = [r['symbol'] for r in scanner_results]
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
        scanner_tickers = []
    
    # 2. Формируем пул (Current + New)
    pool = list(set(current_tickers) | set(scanner_tickers))
    logger.info(f"Evaluating {len(pool)} unique tickers.")
    
    # 3. Бэктест за 1 день
    ticker_performance = []
    for symbol in pool:
        try:
            res = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=1, quiet=True)
            if res:
                # РАНЖИРОВАНИЕ ПО STRATEGY (%)
                strat_profit = res.get("profit_pct", 0)
                ticker_performance.append({
                    "symbol": symbol,
                    "strategy_profit": strat_profit
                })
                logger.info(f"   > {symbol}: Strategy Profit={strat_profit:+.2f}%")
        except: pass

    # 4. Отбор Топ-10 по Strategy Profit
    ticker_performance.sort(key=lambda x: x['strategy_profit'], reverse=True)
    top_performers = ticker_performance[:max_bots]
    new_ticker_list = [t['symbol'] for t in top_performers]
    
    logger.info(f"Selected Top 10 by Strategy: {new_ticker_list}")

    # 5. Ротация (только если список изменился)
    if set(new_ticker_list) != set(current_tickers):
        to_stop = set(current_tickers) - set(new_ticker_list)
        to_start = set(new_ticker_list) - set(current_tickers)
        
        config["tickers"] = new_ticker_list
        config["base_ticker"] = new_ticker_list[0]
        if not config.get("paper_mode", False):
            config["live_swarm"] = new_ticker_list[:7]
        
        await safe_save_json(CONFIG_PATH, config)
        
        # Остановка только тех, кто вылетел
        for t in to_stop:
            await close_bot_positions(t)
            
        # Запуск только новичков (surgical start)
        for t in to_start:
            short_name = t.replace('USDT', '').lower()
            logger.info(f"🚀 Launching new bot: {t}")
            cmd = f"pm2 start main.py --name bot-{short_name} --cwd {CURRENT_DIR} --update-env --interpreter python -- --config config.json --ticker {t}"
            proc = await asyncio.create_subprocess_shell(cmd)
            await proc.wait()
        
        await asyncio.create_subprocess_shell("pm2 save")
        logger.info("Swarm rotation complete. Existing bots were NOT restarted.")
    else:
        logger.info("Swarm remains stable. No restarts triggered.")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
