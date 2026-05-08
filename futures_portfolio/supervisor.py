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
# run_backtest удален
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
        if not stdout:
            return {}
        data = json.loads(stdout.decode('utf-8', errors='replace'))
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

async def get_real_bot_stats(ticker: str, initial_capital: float) -> dict:
    """Получает реальную статистику бота из его файлов состояния."""
    state_path = f"state_{ticker}.json"
    paper_state_path = f"paper_state_{ticker}.json"
    
    state = await safe_load_json(state_path, {})
    paper_state = await safe_load_json(paper_state_path, {})
    
    if not state or not paper_state:
        return {}
        
    paper_balance = paper_state.get("balance", initial_capital)
    siphoned = state.get("siphoning_reserve", 0.0)
    cycles = state.get("rebalance_cycles", 0)
    last_update = state.get("last_update", 0)
    
    # Расчет профита: (Текущий баланс - Начальный) + SAFE
    profit_usdt = (paper_balance - initial_capital) + siphoned
    profit_pct = (profit_usdt / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Используем profit_probation если он есть (рассчитывается в main.py на базе probation_period_days)
    profit_prob_usdt = state.get("profit_probation", 0.0)
    profit_prob_pct = (profit_prob_usdt / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Проверка активности (если бот "завис", его стата может быть неактуальной)
    is_active = (time.time() - last_update) < 600 if last_update > 0 else False
    
    # Прунинг: проверка дельты за 3 часа (profit_prob_usdt)
    if profit_prob_usdt < -0.1:
        logger.warning(f"🔥 Pruning {ticker}: Delta PnL {profit_prob_usdt:.4f} < -0.1. Firing bot.")
        await stop_bot(ticker)
        # Rename files to .fired to prevent re-scan
        for ext in ["state_", "paper_state_"]:
            old_name = f"{ext}{ticker}.json"
            new_name = f"{ext}{ticker}.json.fired"
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    try: os.remove(new_name)
                    except: pass
                try: os.rename(old_name, new_name)
                except Exception as e: logger.error(f"Failed to rename {old_name}: {e}")
        return {}

    return {
        "profit": profit_pct,
        "profit_delta": profit_prob_usdt, # 3h delta
        "profit_probation": profit_prob_pct,
        "safe": siphoned,
        "cycles": cycles,
        "is_active": is_active,
        "is_real_data": True,
        "trailing_stop_paper_timeout_end": state.get("trailing_stop_paper_timeout_end", 0.0)
    }

async def manage_swarm():
    logger.info("--- Starting Pure Live Multi-Slot Supervisor Cycle ---")
    
    config = await safe_load_json(CONFIG_PATH, {})
    if not config: return
    
    max_bots = config.get("max_bots", 10)
    paper_mode_bots = config.get("paper_mode_bots", 8)
    max_real_slots = max(0, max_bots - paper_mode_bots)
    
    probation_hours = config.get("probation_period_days", 0.041) * 24
    
    # 1. Сбор реальности
    running_info = await get_running_bots_info()
    initial_capital = config['portfolios'][0].get('initial_capital', 65.0)
    
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
                    
                    # Проверяем: если бот прибыльный в целом, не кидаем в Blacklist, а даем шанс на пробацию
                    stats = await get_real_bot_stats(ticker, initial_capital)
                    if stats and stats.get('profit', 0) > 0:
                        logger.info(f"🛡️ {ticker} is profitable ({stats['profit']:.2f}%). Moving to probation instead of blacklist.")
                    else:
                        new_black_list.add(ticker)
                elif sig_type == "exit":
                    logger.info(f"✅ {ticker} sent profitable EXIT signal.")
                    await stop_bot(ticker)
            sig_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error processing signal {sig_file}: {e}")

    # 3. Анализ и Рейтинг
    logger.info("Starting scanner analysis (Volatility Factor)...")
    try:
        scanner_results = await asyncio.wait_for(run_scanner(quiet=True, min_volume=10_000_000), timeout=60)
        scanner_tickers = [r['symbol'] for r in scanner_results]
        logger.info(f"Scanner found {len(scanner_tickers)} volatile tickers.")
    except Exception as e:
        logger.error(f"Scanner failed or timed out: {e}")
        scanner_tickers = []
    
    perf_dict = {}
    
    # 3.1. Оцениваем ВСЕХ ботов с реальной историей (The Fact Protocol)
    logger.info("Scanning for real bot states...")
    for state_file in Path(".").glob("state_*.json"):
        ticker = state_file.stem.replace("state_", "")
        if ticker in new_black_list: continue
        
        real_stats = await get_real_bot_stats(ticker, initial_capital)
        if real_stats:
            perf_dict[ticker] = real_stats
            status = "RUNNING" if ticker in running_info else "STOPPED"
            logger.info(f"📈 {ticker} ({status}): Real Profit {real_stats['profit']:.2f}% (Cycles: {real_stats['cycles']})")

    # 3.2. Добавляем КАНДИДАТОВ из сканера (БЕЗ БЭКТЕСТА)
    candidates = [t for t in scanner_tickers if t not in perf_dict and t not in new_black_list]
    logger.info(f"New candidates from scanner: {len(candidates)}")
    
    for symbol in candidates[:max_bots]: 
        if symbol not in perf_dict:
            perf_dict[symbol] = {
                "profit": 0.0,
                "profit_delta": 0.0,
                "is_real_data": False,
                "trailing_stop_paper_timeout_end": 0.0
            }

    # Сортировка: Приоритет запущенным прибыльным -> прибыльным -> топу сканера
    def ranking_key(t):
        p = perf_dict[t]
        if t in running_info and p.get('profit_delta', 0) > 0:
            return 1000 + p['profit_delta']
        if p.get('profit', 0) > 0:
            return 500 + p['profit']
        if t in scanner_tickers:
            return 100 - scanner_tickers.index(t)
        return 0

    all_sorted = sorted(perf_dict.keys(), key=ranking_key, reverse=True)
    
    # 4. Выбор Чемпионов для REAL
    ready_pool = []
    now = time.time()
    now_ms = now * 1000
    probation_ms = probation_hours * 3600 * 1000

    logger.info(f"Selecting champions (Max REAL slots: {max_real_slots})...")
    for ticker in all_sorted:
        perf = perf_dict.get(ticker, {})
        profit = perf.get("profit", 0)
        bot_info = running_info.get(ticker)
        ts_timeout_end = perf.get("trailing_stop_paper_timeout_end", 0.0)
        
        if now < ts_timeout_end:
            continue

        if profit <= 0:
            continue

        is_in_paper = bot_info and bot_info['paper'] and bot_info['status'] == 'online'
        if is_in_paper:
            uptime_ms = bot_info.get('uptime', 0)
            elapsed_ms = now_ms - uptime_ms
            if elapsed_ms >= probation_ms:
                ready_pool.append(ticker)
            continue
            
        if bot_info and not bot_info['paper']:
            ready_pool.append(ticker)

    # Сортировка REAL пула (лучшие из ПРИБЫЛЬНЫХ и ГОТОВЫХ)
    ready_pool.sort(key=lambda x: perf_dict[x]['profit'], reverse=True)
    target_real_bots = ready_pool[:max_real_slots]
    
    # ИТОГОВЫЙ СПИСОК (всего max_bots слотов)
    final_swarm = []
    final_swarm.extend(target_real_bots)
    
    # Дозабиваем остаток PAPER слотов из топа всех прибыльных (включая новых кандидатов)
    for ticker in all_sorted:
        if len(final_swarm) >= max_bots: break
        if ticker not in final_swarm:
            final_swarm.append(ticker)
    
    logger.info(f"🎯 Target REAL (PnL > 0 only): {target_real_bots}")
    logger.info(f"📦 Total Swarm ({len(final_swarm)}): {final_swarm}")

    # 5. ИСПОЛНЕНИЕ
    current_pm2 = await get_running_bots_info()
    
    # Сначала останавливаем тех, кто не в финальном списке ИЛИ должен сменить режим (Paper -> Real)
    for ticker, info in current_pm2.items():
        # Если бота вообще нет в новом списке
        if ticker not in final_swarm:
            logger.info(f"🚫 Stopping bot (not in top): {ticker}")
            await stop_bot(ticker)
            continue
            
        # Если бот должен быть REAL, а он PAPER
        should_be_real = ticker in target_real_bots
        if should_be_real and info['paper']:
            logger.info(f"🔄 Switching {ticker} from PAPER to REAL")
            await stop_bot(ticker)
            
        # Если бот должен быть PAPER, а он REAL
        if not should_be_real and not info['paper']:
            logger.info(f"📉 Demoting {ticker} from REAL to PAPER")
            await stop_bot(ticker)

    # Теперь запускаем тех, кто не запущен в нужном режиме
    updated_pm2 = await get_running_bots_info()
    for ticker in final_swarm:
        should_be_real = ticker in target_real_bots
        
        if ticker not in updated_pm2:
            await start_bot(ticker, paper=(not should_be_real))
        else:
            # На всякий случай проверяем режим еще раз
            if updated_pm2[ticker]['paper'] != (not should_be_real):
                await stop_bot(ticker)
                await start_bot(ticker, paper=(not should_be_real))

    # 6. Финализация конфига (принудительное обновление)
    config["tickers"] = final_swarm
    config["live_swarm"] = sorted(target_real_bots)
    config["base_ticker"] = final_swarm[0] if final_swarm else "BTCUSDT"
    config["black_list"] = sorted(list(new_black_list))
    
    logger.info(f"Writing to config: tickers={len(config['tickers'])}, swarm={config['live_swarm']}")
    await safe_save_json(CONFIG_PATH, config)
    
    # Релоад PM2
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
