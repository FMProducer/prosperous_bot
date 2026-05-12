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
# Динамический путь к данным относительно корня проекта
DATA_DIR = str(BASE_PATH.parent / "third_party" / "rl-trading-binance" / "user_data" / "data" / "binance" / "futures")

def read_shared_config(path: str) -> Dict[str, Any]:
    """Чтение общего конфига без использования блокировок."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Critical: Shared config read failed: {e}")
        return {}

async def get_ticker_pnl(ticker: str) -> float:
    """Чтение прибыли из мастер-бумажного стейта для ранжирования."""
    state_path = BASE_PATH / f"paper_state_{ticker}.json"
    state = await safe_load_json(str(state_path), {})
    return state.get("last_profit", 0.0)

async def get_running_bots_info() -> Dict[str, dict]:
    """Получает информацию о ботах с учетом префиксов paper- и real-"""
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
            name = app['name']
            if name.startswith(('paper-', 'real-')):
                args = app.get('pm2_env', {}).get('args', [])
                ticker = next((args[i+1] for i, a in enumerate(args) if a == '--ticker'), None)
                is_paper = name.startswith('paper-')
                
                if ticker:
                    # Извлекаем pm_uptime (timestamp старта в мс)
                    uptime_ms = app.get('pm2_env', {}).get('pm_uptime', 0)
                    # Ключ теперь включает режим, так как один тикер может быть в обоих режимах
                    bots[f"{'p' if is_paper else 'r'}_{ticker}"] = {
                        "name": name, 
                        "paper": is_paper, 
                        "status": app['pm2_env']['status'],
                        "uptime": uptime_ms
                    }
        return bots
    except Exception as e:
        logger.error(f"Failed to get PM2 list: {e}")
        return {}

async def stop_bot(ticker: str, is_paper: bool = False):
    prefix = "paper" if is_paper else "real"
    short_name = ticker.replace('USDT', '').lower()
    proc_name = f"{prefix}-{short_name}"
    logger.info(f"🛑 Stopping process {proc_name}...")
    
    # 1. Всегда останавливаем процесс в PM2
    try:
        proc = await asyncio.create_subprocess_shell(f"pm2 delete {proc_name}")
        await proc.wait()
        await asyncio.sleep(0.5)
    except: pass

async def start_bot(ticker: str, is_paper: bool = True):
    prefix = "paper" if is_paper else "real"
    mode_str = prefix.upper()
    logger.info(f"🚀 Launching {mode_str} bot for {ticker}...")
    short_name = ticker.replace('USDT', '').lower()
    proc_name = f"{prefix}-{short_name}"
    paper_flag = "--paper" if is_paper else ""
    
    # Удаляем старый если есть
    try:
        proc = await asyncio.create_subprocess_shell(f"pm2 delete {proc_name}")
        await proc.wait()
    except: pass
    
    # Формируем команду запуска
    cmd = f'pm2 start main.py --name {proc_name} --cwd "{BASE_PATH}" --update-env --interpreter "{sys.executable}" -- --config config.json --ticker {ticker} {paper_flag}'
    proc = await asyncio.create_subprocess_shell(cmd)
    await proc.wait()

async def get_real_bot_stats(ticker: str, initial_capital: float, config: dict, running_info: dict) -> dict:
    """Получает реальную статистику бота из его файлов состояния."""
    state_path = BASE_PATH / f"paper_state_{ticker}.json"
    paper_state_path = state_path # В новой архитектуре бумажный стейт един
    
    state = await safe_load_json(str(state_path), {})
    
    if not state:
        return {}
        
    siphoned = state.get("siphoning_reserve", 0.0)
    cycles = state.get("rebalance_cycles", 0)
    last_update = state.get("last_update", 0)
    
    # Архитектурное исправление: Строгий SSOT.
    profit_usdt = state.get("last_profit", 0.0)
    
    # Если бот уже уволен, читаем его зафиксированный финальный PnL
    # (Хотя в новой архитектуре мы их реже увольняем из инкубатора)
    if "final_profit" in state and f"p_{ticker}" not in running_info:
        profit_usdt = state.get("final_profit", profit_usdt)

    profit_pct = (profit_usdt / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Проверка активности
    is_active = (time.time() - last_update) < 600 if last_update > 0 else False
    
    # Efficiency calculation: Average profit per cycle, protected from division by zero
    min_cycles = config.get("min_cycles_for_rank", 20)
    efficiency = profit_usdt / max(cycles, min_cycles)

    # Прунинг: Только по абсолютному убытку (Overall PnL < 0)
    if profit_usdt < 0:
        # ЗАЩИТА НОВИЧКА: Не убиваем сразу после старта (даем время отбить комиссию)
        probation_days = config.get("probation_period_days", 0.041)
        probation_sec = probation_days * 86400
        uptime_sec = (time.time() - last_update) if last_update > 0 else 0

        if last_update > 0 and uptime_sec < probation_sec:
            logger.info(f"🛡️ {ticker} has negative PnL ({profit_usdt:.4f}), but is still in startup grace period. Skipping.")
        else:
            logger.warning(f"🔥 Pruning {ticker}: Overall PnL {profit_usdt:.4f} < 0. Firing bot.")

            # Сохраняем финальный профит ПЕРЕД сбросом стейта
            try:
                state_data = await safe_load_json(str(state_path), {})
                state_data["final_profit"] = profit_usdt
                await safe_save_json(str(state_path), state_data)
                logger.info(f"✅ Final profit {profit_usdt:+.4f} USDT saved for fired bot {ticker}")
            except Exception as e:
                logger.error(f"Failed to save final profit for {ticker}: {e}")

            await stop_bot(ticker, is_paper=True)
            return {}

    return {
        "profit": profit_pct,
        "profit_usdt": profit_usdt,
        "efficiency": efficiency,
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
    
    # 1. Сбор реальности
    running_info = await get_running_bots_info()
    initial_capital = config['portfolios'][0].get('initial_capital', 60.0)
    
    # 0. Загрузка текущего состояния
    current_tickers = set(config.get("tickers", []))

    # 2. Обработка сигналов от ботов (пассивный мониторинг)
    signal_dir = BASE_PATH / "signals"
    if not signal_dir.exists(): signal_dir.mkdir(parents=True)
    
    # Очищаем черный список каждый цикл, давая тикерам шанс на рециркуляцию
    new_black_list = set()
    for sig_file in signal_dir.glob("*.flag"):
        try:
            parts = sig_file.stem.split("_")
            if len(parts) >= 2:
                sig_type, ticker = parts[0], parts[1]
                if sig_type == "stop":
                    logger.warning(f"⚠️ {ticker} sent emergency STOP signal.")
                    await stop_bot(ticker)
                    
                    # Проверяем: если бот прибыльный в целом, не кидаем в Blacklist, а даем шанс на пробацию
                    stats = await get_real_bot_stats(ticker, initial_capital, config, running_info)
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
    logger.info("🔍 Running ticker scanner (Volatility Factor)...")
    try:
        scanner_results: list[dict[str, Any]] = await asyncio.wait_for(run_scanner(quiet=True, min_volume=10_000_000), timeout=60)

        if not scanner_results:
            logger.warning(f"⚠️ Scanner found 0 tickers in {DATA_DIR}. Check data availability.")

        scanner_tickers = [r['symbol'] for r in scanner_results]
        logger.info(f"📊 Scanner found {len(scanner_tickers)} potential tickers.")
    except Exception as e:
        logger.error(f"❌ Scanner failed or timed out: {e}")
        scanner_tickers = []
    
    perf_dict = {}
    
    # 3.1. Оцениваем ВСЕХ ботов с реальной историей (The Fact Protocol)
    logger.info("Scanning for real bot states...")
    # В новой архитектуре мы ищем paper_state_*.json как основной источник истины для рейтинга
    for state_file in BASE_PATH.glob("paper_state_*.json"):
        ticker = state_file.stem.replace("paper_state_", "")
        if ticker in new_black_list: continue
        
        real_stats = await get_real_bot_stats(ticker, initial_capital, config, running_info)
        if real_stats:
            perf_dict[ticker] = real_stats
            status = "RUNNING" if f"p_{ticker}" in running_info else "STOPPED"
            logger.info(f"📈 {ticker} ({status}): Real Profit {real_stats['profit']:.2f}% (Cycles: {real_stats['cycles']})")

    # 3.2. Добавляем КАНДИДАТОВ из сканера (БЕЗ БЭКТЕСТА)
    candidates = [t for t in scanner_tickers if t not in perf_dict and t not in new_black_list]
    logger.info(f"New candidates from scanner: {len(candidates)}")
    
    for symbol in candidates:
        if symbol not in perf_dict:
            perf_dict[symbol] = {
                "profit": 0.0,
                "efficiency": 0.0,
                "is_real_data": False,
                "trailing_stop_paper_timeout_end": 0.0
            }

    # Сортировка: По эффективности (средний профит на цикл)
    def ranking_key(t):
        p = perf_dict[t]
        # Rank by efficiency (USDT/Cycle)
        return p.get('efficiency', 0.0)

    all_sorted = sorted(perf_dict.keys(), key=ranking_key, reverse=True)

    # Логика ротации (Dynamic Ticker Recirculation):
    # 1. Находим кандидатов на вылет (PnL < 0) среди ТЕКУЩИХ активных ботов ИНКУБАТОРА
    underperformers = []
    for ticker in list(current_tickers):
        # Проверяем прибыль в инкубаторе
        pnl = await get_ticker_pnl(ticker)
        if pnl < 0:
            underperformers.append((ticker, pnl))

    # 2. Находим новых лидеров из сканера, которых нет в текущем рое
    new_candidates = [t for t in scanner_tickers if t not in current_tickers and t not in new_black_list]

    # 3. Производим замену (Swap) в ИНКУБАТОРЕ
    swapped_count = 0
    max_swaps = max(1, len(current_tickers) // 5) if current_tickers else 1 # Не более 20%

    underperformers.sort(key=lambda x: x[1]) # От худшего к лучшему

    for loser_ticker, pnl in underperformers:
        if new_candidates and swapped_count < max_swaps:
            winner_ticker = new_candidates.pop(0)
            logger.info(f"🔄 Swapping Incubator: {loser_ticker} (PnL: {pnl:.2f}) -> {winner_ticker} (Rank Leader)")
            if loser_ticker in current_tickers: current_tickers.remove(loser_ticker)
            current_tickers.add(winner_ticker)
            swapped_count += 1
    
    # 4. Выбор Чемпионов для REAL (Меритократия + Белый список)
    ready_pool = []
    now = time.time()
    min_cycles = config.get("min_cycles_for_rank", 20)
    real_whitelist = set(config.get("real_whitelist", []))

    logger.info(f"Selecting champions (Max REAL slots: {max_real_slots}, Min Cycles: {min_cycles}, Whitelist: {len(real_whitelist)})...")
    for ticker in all_sorted:
        perf = perf_dict.get(ticker, {})
        profit = perf.get("profit", 0)
        bot_cycles = perf.get("cycles", 0)
        ts_timeout_end = perf.get("trailing_stop_paper_timeout_end", 0.0)
        
        if now < ts_timeout_end:
            continue

        # ПРОВЕРКА БЕЛОГО СПИСКА
        is_whitelisted = ticker in real_whitelist
        if not is_whitelisted:
            # logger.debug(f"ℹ️ {ticker} not in real_whitelist. Paper-only.")
            continue

        # СТРОГИЙ ОТБОР: Только прибыльные И накопившие достаточно циклов (опыта)
        if profit > 0 and bot_cycles >= min_cycles:
            logger.info(f"✅ {ticker} qualified for REAL: Profit {profit:.2f}%, Cycles {bot_cycles}/{min_cycles}")
            ready_pool.append(ticker)
        else:
            reason = ""
            if profit <= 0: reason += f"Profit {profit:.2f} <= 0 "
            if bot_cycles < min_cycles: reason += f"Cycles {bot_cycles} < {min_cycles}"
            logger.info(f"❌ {ticker} NOT qualified: {reason}")

    # Сортировка REAL пула (лучшие из ПРИБЫЛЬНЫХ и ГОТОВЫХ)
    ready_pool.sort(key=lambda x: perf_dict[x]['efficiency'], reverse=True)
    target_real_bots = ready_pool[:max_real_slots]
    
    # ИТОГОВЫЙ СПИСОК ИНКУБАТОРА (всего paper_mode_bots слотов)
    final_incubator = list(current_tickers)
    
    # Приоритет 3: Дозабиваем остаток из топа прибыльных / кандидатов
    for ticker in all_sorted:
        if len(final_incubator) >= paper_mode_bots: break
        if ticker not in final_incubator:
            final_incubator.append(ticker)
    
    logger.info(f"🎯 Combat Swarm (REAL): {target_real_bots}")
    logger.info(f"🧪 Incubator Swarm (PAPER): {final_incubator}")

    # 5. ИСПОЛНЕНИЕ (Strict Isolation Protocol)
    updated_pm2 = await get_running_bots_info()
    
    # 5.1. Управление ИНКУБАТОРОМ (Paper)
    for ticker in final_incubator:
        if f"p_{ticker}" not in updated_pm2:
            await start_bot(ticker, is_paper=True)
            
    # Останавливаем лишних в инкубаторе
    for key, info in updated_pm2.items():
        if key.startswith("p_"):
            ticker = key.replace("p_", "")
            if ticker not in final_incubator:
                logger.info(f"🚫 Removing from Incubator: {ticker}")
                await stop_bot(ticker, is_paper=True)

    # 5.2. Управление БОЕВЫМ РОЕМ (Real)
    for ticker in target_real_bots:
        if f"r_{ticker}" not in updated_pm2:
            await start_bot(ticker, is_paper=False)
            
    # Останавливаем лишних в боевом рою
    for key, info in updated_pm2.items():
        if key.startswith("r_"):
            ticker = key.replace("r_", "")
            if ticker not in target_real_bots:
                logger.info(f"🚫 Removing from Combat Swarm: {ticker}")
                # 1. Удаляем процесс из PM2
                await stop_bot(ticker, is_paper=False)
                
                # 2. Закрываем позиции на бирже, но НЕ стираем файлы стейта!
                # Флаг --stop в паре с исправленным main.py теперь безопасен.
                cmd = f'"{sys.executable}" main.py --ticker {ticker} --stop'
                await (await asyncio.create_subprocess_shell(cmd)).wait()
                logger.info(f"✅ Combat positions for {ticker} closed. State preserved.")

    # 6. Финализация конфига (принудительное обновление)
    # Обновляем только если есть валидные данные
    try:
        # Сохраняем обновленный состав после замен
        config["tickers"] = sorted(final_incubator)
        config["live_swarm"] = sorted(target_real_bots)
        config["base_ticker"] = config["tickers"][0] if config["tickers"] else "BTCUSDT"
        config["black_list"] = sorted(list(new_black_list))

        logger.info(f"💾 Config Integrity Verified. Writing: {len(config['tickers'])} incubator, {len(target_real_bots)} in combat swarm.")
        await safe_save_json(CONFIG_PATH, config)
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}", exc_info=True)
    
    # Релоад PM2
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())
