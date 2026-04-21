import asyncio
import json
import logging
import os
import subprocess
import time
from rank_tickers import main as run_scanner
from backtest_rebalance import run_backtest

# Настройка логирования
log_dir = os.path.join(os.path.dirname(__file__), "logs")
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

CONFIG_PATH = "config.json"
DATA_DIR = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"

async def manage_swarm():
    logger.info("--- Starting Supervisor Cycle ---")
    
    # 1. Загрузка текущего конфига
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config {CONFIG_PATH} not found!")
        return
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse config.json: {e}")
        return
    
    current_tickers = config.get("tickers", [])
    max_bots = config.get("max_bots", 5)
    
    # 2. Запуск сканера
    logger.info("Step 1: Running Market Scanner (Min Vol: 100M)...")
    try:
        ranked_list = await run_scanner(quiet=True, min_volume=100_000_000)
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
        return
    
    scores = {r['symbol']: r for r in ranked_list}
    
    # 3. Анализ текущих тикеров: кого оставляем?
    tickers_to_keep = []
    
    now = time.time()
    for t in current_tickers:
        r = scores.get(t)
        state_path = f"state_{t}.json"
        
        # Проверка на "Застой" (Stagnation)
        stagnated = False
        if os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                
                last_cycles = state.get("rebalance_cycles", 0)
                last_check_cycles = state.get("supervisor_last_cycles", 0)
                last_check_time = state.get("supervisor_last_check_time", 0)
                
                # Проверяем стагнацию только если с прошлой проверки прошло > 3.5 часов
                # И количество циклов не изменилось
                if last_check_time > 0 and (now - last_check_time) > 12000: # ~3.3 часа
                    if last_cycles <= last_check_cycles and last_cycles > 0:
                        logger.info(f"Ticker {t} is stagnated (No cycles for 3.5h).")
                        stagnated = True
                
                # Обновляем метки для следующей проверки
                state["supervisor_last_cycles"] = last_cycles
                state["supervisor_last_check_time"] = now
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
            except: pass

        # Критерии удержания:
        if r and r['score'] >= 150 and not stagnated:
            tickers_to_keep.append(t)
        else:
            reason = "low score" if not r or r['score'] < 150 else "stagnation"
            logger.info(f"Ticker {t} scheduled for replacement (Reason: {reason}).")
            
    # 4. Поиск и валидация новых кандидатов
    available_slots = max_bots - len(tickers_to_keep)
    validated_new_tickers = []
    
    if available_slots > 0:
        # Берем кандидатов Tier-1 и Tier-2 (Score >= 150), которых нет в портфеле
        potential_new = [r for r in ranked_list if r['symbol'] not in tickers_to_keep and r['score'] >= 150]
        # Сортируем по Score (лучшие сверху)
        potential_new.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Step 2: Validating {len(potential_new)} candidates for {available_slots} slots...")
        
        for c in potential_new:
            if len(validated_new_tickers) >= available_slots:
                break
                
            symbol = c['symbol']
            logger.info(f"Testing {symbol} (Score: {c['score']:.2f})...")
            try:
                # Бэктест за последние 24ч
                results = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=1, quiet=True)
                
                # Мягкий фильтр для новых кандидатов:
                # 1. Профит должен быть не сильно отрицательным (волатильность важнее минутной удачи)
                # 2. Но лучше если профит > 0
                if results and results.get("profit_pct", -1) > -0.5 and results.get("max_dd_pct", 100) < 5.0:
                    validated_new_tickers.append(symbol)
                    logger.info(f"✅ {symbol} PASSED Backtest (Profit: {results['profit_pct']:.2f}%).")
                else:
                    logger.info(f"❌ {symbol} REJECTED (Bad backtest results).")
            except Exception as e:
                logger.warning(f"⚠️ {symbol} backtest error: {e}")

    # 5. Сборка нового списка
    new_ticker_list = tickers_to_keep + validated_new_tickers
    
    if set(new_ticker_list) != set(current_tickers) and new_ticker_list:
        logger.info(f"Step 3: Rotating swarm. New: {new_ticker_list}")
        config["tickers"] = new_ticker_list
        if "AAVEUSDT" in new_ticker_list: config["base_ticker"] = "AAVEUSDT"
        else: config["base_ticker"] = new_ticker_list[0]
        
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # 6. Перезапуск
        logger.info("Step 4: PM2 Sync (Targeted delete)...")
        try:
            # Удаляем только торговых ботов, не трогаем supervisor-service
            subprocess.run(["pm2", "delete", "/bot-.*/"], check=True, shell=True)
            subprocess.run(["pm2", "start", "ecosystem.config.js"], check=True, shell=True)
            subprocess.run(["pm2", "save"], check=True, shell=True)
            logger.info("🚀 Swarm updated and saved.")
        except Exception as e:
            logger.error(f"PM2 Error: {e}")
    else:
        logger.info("Swarm remains stable (No better candidates or no churn needed).")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
