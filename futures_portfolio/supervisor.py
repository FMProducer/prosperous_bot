import asyncio
import json
import logging
import os
import time
import sys
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
# Принудительная загрузка .env из текущей директории для надежности
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")
DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"

async def safe_load_json(path: str, default: dict, retries: int = 5) -> dict:
    for i in range(retries):
        try:
            if not os.path.exists(path): return default
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError):
            if i == retries - 1: return default
            await asyncio.sleep(0.5)
    return default

async def safe_save_json(path: str, data: dict, retries: int = 5):
    for i in range(retries):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if os.path.exists(path): os.remove(path)
            os.rename(tmp, path)
            return
        except PermissionError:
            if i == retries - 1: break
            await asyncio.sleep(0.5)

async def manage_swarm():
    logger.info("--- Starting Supervisor Cycle ---")
    
    # 1. Загрузка текущего конфига
    config = await safe_load_json(CONFIG_PATH, {})
    if not config:
        logger.error(f"Config {CONFIG_PATH} not found or empty!")
        return
    
    current_tickers = config.get("tickers", [])
    max_bots = config.get("max_bots", 10)
    
    # 2. Запуск сканера
    logger.info("Step 1: Running Market Scanner (Min Vol: 100M)...")
    try:
        ranked_list = await run_scanner(quiet=True, min_volume=100_000_000)
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
        return
    
    scores = {r['symbol']: r for r in ranked_list}
    
    # 3. Анализ текущих тикеров
    tickers_to_keep = []
    candidates_to_drop = []
    
    now = time.time()
    for t in current_tickers:
        r = scores.get(t)
        # Гистерезис: порог удержания (80) ниже порога входа (150)
        current_score = r['score'] if r else 0
        if r and current_score >= 80:
            tickers_to_keep.append(t)
        else:
            reason = "low score" if not r or current_score < 80 else "rotation"
            logger.info(f"Ticker {t} candidate for replacement ({reason}).")
            candidates_to_drop.append({"symbol": t, "score": current_score})
            
    # 4. Поиск и валидация новых кандидатов
    available_slots = max_bots - len(tickers_to_keep)
    validated_new_tickers = []
    
    if available_slots > 0:
        potential_new = [r for r in ranked_list if r['symbol'] not in current_tickers and r['score'] >= 150]
        potential_new.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Step 2: Validating candidates for {available_slots} slots...")
        for c in potential_new:
            if len(validated_new_tickers) >= available_slots: break
            symbol = c['symbol']
            try:
                # Быстрый бэктест на 1 день
                results = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=1, quiet=True)
                if results and results.get("profit_pct", -1) > -0.5:
                    validated_new_tickers.append(symbol)
                    logger.info(f"✅ {symbol} PASSED Backtest.")
                else:
                    logger.info(f"❌ {symbol} REJECTED (Bad results).")
            except: pass

    # 5. Гарантия размера роя
    total_slots_filled = len(tickers_to_keep) + len(validated_new_tickers)
    if total_slots_filled < max_bots and candidates_to_drop:
        deficit = max_bots - total_slots_filled
        candidates_to_drop.sort(key=lambda x: x['score'], reverse=True)
        for r in candidates_to_drop[:deficit]:
            tickers_to_keep.append(r['symbol'])
            logger.info(f"🛡️ Rescuing {r['symbol']} to maintain swarm size.")

    # 6. Сборка нового списка
    new_ticker_list = tickers_to_keep + validated_new_tickers
    
    # 7. Swarm Promotion (Respect global paper_mode)
    bot_stats = []
    for t in new_ticker_list:
        state_path = os.path.join(CURRENT_DIR, f"state_{t}.json")
        state = await safe_load_json(state_path, {"current_profit": 0.0})
        bot_stats.append({"ticker": t, "profit": state.get("current_profit", 0)})

    bot_stats.sort(key=lambda x: x['profit'], reverse=True)
    
    global_paper_mode = config.get("paper_mode", False)
    if global_paper_mode:
        live_tickers = []
        logger.info("Global PAPER MODE active. All bots forced to paper.")
    else:
        live_tickers = [b['ticker'] for b in bot_stats[:7]] # Топ-7 в Real

    old_live_tickers = config.get("live_swarm", [])
    config["live_swarm"] = live_tickers
    config["tickers"] = new_ticker_list
    if new_ticker_list:
        config["base_ticker"] = new_ticker_list[0]
    
    await safe_save_json(CONFIG_PATH, config)

    # 8. Хирургический PM2 Sync
    if set(new_ticker_list) != set(current_tickers) or set(live_tickers) != set(old_live_tickers):
        to_stop = set(current_tickers) - set(new_ticker_list)
        to_start = set(new_ticker_list) - set(current_tickers)
        to_restart = (set(new_ticker_list) & set(current_tickers)) & (set(live_tickers) ^ set(old_live_tickers))

        logger.info(f"Syncing PM2: Start={to_start}, Stop={to_stop}, Restart={to_restart}")
        
        try:
            # Останавливаем
            for t in (to_stop | to_restart):
                short_name = t.replace('USDT', '').lower()
                proc = await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
                await proc.wait()
            
            # Запускаем
            for t in (to_start | to_restart):
                short_name = t.replace('USDT', '').lower()
                # Используем абсолютные пути и тот же интерпретатор
                cmd = f"pm2 start main.py --name bot-{short_name} --cwd {CURRENT_DIR} --update-env --interpreter {sys.executable} -- --config {CONFIG_PATH} --ticker {t}"
                proc = await asyncio.create_subprocess_shell(cmd)
                await proc.wait()
            
            await asyncio.create_subprocess_shell("pm2 save")
            
            logger.info(f"Swarm Updated: Total={len(new_ticker_list)}, Started={len(to_start)}, Stopped={len(to_stop)}, Restored={len(to_restart)}")
        except Exception as e:
            logger.error(f"PM2 Sync Error: {e}")
    else:
        logger.info("Swarm remains stable.")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
