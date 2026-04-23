import asyncio
import json
import logging
import os
import time
import sys
from rank_tickers import main as run_scanner
from backtest_rebalance import run_backtest
from notifier import TelegramNotifier

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
    notifier = TelegramNotifier()
    
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
        msg = f"❌ Supervisor: Scanner failed: {e}"
        logger.error(msg)
        await notifier.send_message(msg)
        return
    
    scores = {r['symbol']: r for r in ranked_list}
    
    # 3. Анализ текущих тикеров: кого оставляем?
    tickers_to_keep = []
    candidates_to_drop = []
    
    now = time.time()
    for t in current_tickers:
        r = scores.get(t)
        state_path = f"state_{t}.json"
        
        # Проверка на "Застой" (Stagnation)
        stagnated = False
        if os.path.exists(state_path):
            try:
                state = await safe_load_json(state_path, {})
                last_cycles = state.get("rebalance_cycles", 0)
                last_check_cycles = state.get("supervisor_last_cycles", 0)
                last_check_time = state.get("supervisor_last_check_time", 0)
                
                if last_check_time > 0 and (now - last_check_time) > 12000: # ~3.3 часа
                    if last_cycles <= last_check_cycles and last_cycles > 0:
                        logger.info(f"Ticker {t} is stagnated (No cycles for 3.5h).")
                        stagnated = True
                
                state["supervisor_last_cycles"] = last_cycles
                state["supervisor_last_check_time"] = now
                await safe_save_json(state_path, state)
            except: pass

        # Гистерезис: порог удержания (80) ниже порога входа (150)
        current_score = r['score'] if r else 0
        if r and current_score >= 80 and not stagnated:
            tickers_to_keep.append(t)
        else:
            reason = "low score" if not r or current_score < 80 else "stagnation"
            logger.info(f"Ticker {t} candidate for replacement (Reason: {reason}, Score: {current_score:.1f}).")
            candidates_to_drop.append({"symbol": t, "score": current_score, "reason": reason})
            
    # 4. Поиск и валидация новых кандидатов
    available_slots = max_bots - len(tickers_to_keep)
    validated_new_tickers = []
    
    if available_slots > 0:
        potential_new = [r for r in ranked_list if r['symbol'] not in current_tickers and r['score'] >= 150]
        potential_new.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Step 2: Validating {len(potential_new)} candidates for {available_slots} slots...")
        
        for c in potential_new:
            if len(validated_new_tickers) >= available_slots:
                break
            
            symbol = c['symbol']
            try:
                results = await run_backtest(CONFIG_PATH, DATA_DIR, live_mode=True, ticker_override=symbol, days=1, quiet=True)
                if results and results.get("profit_pct", -1) > -0.5 and results.get("max_dd_pct", 100) < 5.0:
                    validated_new_tickers.append(symbol)
                    logger.info(f"✅ {symbol} PASSED Backtest (Profit: {results['profit_pct']:.2f}%).")
                else:
                    logger.info(f"❌ {symbol} REJECTED (Bad backtest results).")
            except Exception as e:
                logger.warning(f"⚠️ {symbol} backtest error: {e}")

    # 5. ГАРАНТИЯ РАЗМЕРА РОЯ: если новых не хватило, оставляем лучших из "отсеянных"
    total_slots_filled = len(tickers_to_keep) + len(validated_new_tickers)
    if total_slots_filled < max_bots and candidates_to_drop:
        deficit = max_bots - total_slots_filled
        candidates_to_drop.sort(key=lambda x: x['score'], reverse=True)
        to_rescue = candidates_to_drop[:deficit]
        for r in to_rescue:
            tickers_to_keep.append(r['symbol'])
            logger.info(f"🛡️ Rescuing {r['symbol']} (Score: {r['score']:.1f}) to maintain swarm size.")
        # Обновляем список реально выбывших
        candidates_to_drop = [c for c in candidates_to_drop if c['symbol'] not in [r['symbol'] for r in to_rescue]]

    # 6. Сборка нового списка и Хирургический PM2 Sync
    new_ticker_list = tickers_to_keep + validated_new_tickers
    
    if set(new_ticker_list) != set(current_tickers):
        to_stop = set(current_tickers) - set(new_ticker_list)
        to_start = set(new_ticker_list) - set(current_tickers)
        
        msg = (
            f"🔄 <b>Swarm Rotation</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"✅ Keep: {len(tickers_to_keep)} bots\n"
            f"➕ Add: {', '.join(to_start) if to_start else 'None'}\n"
            f"❌ Drop: {', '.join([c['symbol'] for c in candidates_to_drop]) if candidates_to_drop else 'None'}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )
        logger.info(f"Rotating swarm. New: {new_ticker_list}")
        await notifier.send_message(msg)
        
        config["tickers"] = new_ticker_list
        config["base_ticker"] = new_ticker_list[0]
        await safe_save_json(CONFIG_PATH, config)
            
        try:
            # 1. Останавливаем только тех, кто выбыл
            for t in to_stop:
                short_name = t.replace('USDT', '').toLowerCase()
                await asyncio.create_subprocess_shell(f"pm2 delete bot-{short_name}")
                logger.info(f"Stopping bot-{short_name}")
            
            # 2. Запускаем только новых
            for t in to_start:
                short_name = t.replace('USDT', '').toLowerCase()
                cmd = f"pm2 start main.py --name bot-{short_name} --interpreter python -- --config {CONFIG_PATH} --ticker {t}"
                await asyncio.create_subprocess_shell(cmd)
                logger.info(f"Starting bot-{short_name}")
            
            await asyncio.create_subprocess_shell("pm2 save")
            logger.info("🚀 Swarm surgical update complete.")
            if to_start or to_stop:
                await notifier.send_message(f"🚀 <b>Swarm Surgically Updated</b>\nStarted: {len(to_start)} | Stopped: {len(to_stop)}")
        except Exception as e:
            err_msg = f"❌ PM2 Surgical Error: {e}"
            logger.error(err_msg)
            await notifier.send_message(err_msg)
    else:
        logger.info("Swarm remains stable (No churn needed).")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
