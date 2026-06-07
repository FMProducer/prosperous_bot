import asyncio
import json
import logging
import os
import time
import math
import random
import sys
import shutil
from typing import Dict, List, Set, Any
from decimal import Decimal, InvalidOperation
from dotenv import load_dotenv
from pathlib import Path

# Загрузка окружения
load_dotenv()

from rank_tickers import main as run_scanner
from storage import safe_load_json, safe_save_json
from connector import BinanceConnector

# Настройка логирования
BASE_PATH = Path(__file__).resolve().parent
log_dir = BASE_PATH / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

def setup_logger():
    l = logging.getLogger("Supervisor")
    l.setLevel(logging.INFO)
    # Clear existing handlers if any
    if l.handlers:
        l.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    
    # File Handler
    fh = logging.FileHandler(log_dir / "supervisor.log", encoding="utf-8")
    fh.setFormatter(formatter)
    l.addHandler(fh)
    
    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    l.addHandler(sh)
    
    # Prevent propagation to root logger
    l.propagate = False
    return l

logger = setup_logger()

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

async def enforce_swarm_consistency(connector: BinanceConnector, config: dict) -> Set[str]:
    """
    Checks for unauthorized positions on the exchange.
    If a position exists but the PM2 process is missing, attempts to HEAL (relaunch)
    the bot instead of closing its positions, provided a valid state file exists.
    """
    logger.info("🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.")
    
    live_swarm_tickers = set(config.get("live_swarm", []))
    real_whitelist = set(config.get("real_whitelist", []))
    allowed_tickers = live_swarm_tickers.union(real_whitelist)

    active_positions = await connector.get_positions() # Get all open positions from exchange
    if not active_positions:
        return set()

    # Федчим состояние процессов прямо сейчас для выявления упавших юнитов
    running_bots = await get_running_bots_info()
    running_real = {k.replace("r_", "") for k in running_bots.keys() if k.startswith("r_")}

    to_close_tickers = set()
    to_heal_tickers = set()

    for pos_key in active_positions.keys():
        ticker = pos_key.split('_')[0]

        # Ситуация нормальная: позиция разрешена и процесс работает
        if ticker in allowed_tickers and ticker in running_real:
            continue

        # Аномалия: Позиция активна, но процесс в PM2 отсутствует
        state_path = BASE_PATH / f"real_state_{ticker}.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)

                # Инвариант легитимности: у бота есть история или открытый виртуальный объем
                if state_data.get("rebalance_cycles", 0) > 0 or abs(state_data.get("virt_qty", 0.0)) > 0:
                    # Защита от воскрешения ботов, остановленных по Trailing Stop
                    if state_data.get("trailing_stop_triggered", False):
                        logger.warning(f"⚠️ HEAL REJECTED: {ticker} was stopped by Trailing Stop. Scheduling position liquidation.")
                        to_close_tickers.add(ticker)
                        continue

                    logger.warning(f"🚨 HEAL TRIGGERED: Active position for {ticker} detected on exchange, but PM2 process is dead. Real state file is valid. Initiating recovery...")
                    to_heal_tickers.add(ticker)
                    continue
            except Exception as e:
                logger.error(f"Failed to verify state data integrity for {ticker}: {e}")

        # Если вайтлисты пусты или это действительно чужая позиция (нет стейт файла)
        if not allowed_tickers:
            logger.warning(f"⚠️ Unauthorized stray position for {ticker} (allowed pools are empty). Scheduled for liquidation.")
            to_close_tickers.add(ticker)
        elif ticker not in allowed_tickers:
            logger.warning(f"⚠️ Unauthorized position found for {ticker} (not in live_swarm or real_whitelist). Scheduled for liquidation.")
            to_close_tickers.add(ticker)

    # Выполняем регенерацию процессов
    for ticker in to_heal_tickers:
        logger.info(f"⚡ Relaunching crashed REAL process for {ticker} from saved state.")
        if "live_swarm" not in config:
            config["live_swarm"] = []
        if ticker not in config["live_swarm"]:
            config["live_swarm"].append(ticker)

        # Перезапускаем. Так как real_state_{ticker}.json существует, start_bot НЕ сотрет его данные
        await start_bot(ticker, is_paper=False, config=config)
        logger.info(f"✅ Relaunch command sent for {ticker}.")

    for ticker in to_close_tickers:
        logger.info(f"🧹 Closing all positions for unauthorized ticker: {ticker}")
        cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
        await (await asyncio.create_subprocess_shell(cmd_stop)).wait()
        logger.info(f"✅ Positions for {ticker} closed successfully.")

    return to_heal_tickers

async def enforce_invariant_gate(connector: BinanceConnector, config: dict):
    """
    Гарантия непрерывности данных: никогда не удаляем тикер из live_swarm, если по нему открыта позиция на бирже.
    Использует Decimal для строгих финансовых расчетов и Dust Guard фильтрацию.
    """
    try:
        raw_positions = await connector.get_positions()
        if not raw_positions:
            return

        dust_threshold_usdt = config.get("min_notional_usdt", 5.5)
        live_swarm = set(config.get("live_swarm", []))

        # Структурируем позиции (учитывая Hedge Mode: LONG и SHORT)
        for pos_key, pos_data in raw_positions.items():
            qty = Decimal(str(pos_data.get("qty", "0.0"))).copy_abs()
            if qty == Decimal("0"):
                continue

            price = Decimal(str(pos_data.get("mark_price", pos_data.get("entry_price", "0.0"))))
            notional_value = qty * price

            # Фильтр Dust Guard по USDT номиналу
            if notional_value >= Decimal(str(dust_threshold_usdt)):
                ticker = pos_key.split("_")[0]
                if ticker not in live_swarm:
                    live_swarm.add(ticker)
                    logger.warning(f"🛡️ Invariant Protection Gate: Forced retention of {ticker} in live_swarm due to active exposure (${notional_value:.2f} USDT).")

        config["live_swarm"] = sorted(list(live_swarm))
    except Exception as e:
        logger.error(f"Failed to verify exchange exposure tracking in Invariant Gate: {e}")

async def reset_bot_state_files(ticker: str, is_paper: bool, config: dict) -> None:
    """
    Сбрасывает файлы состояния к чистым начальным значениям, используя
    динамическое ребазирование капитала для предотвращения инфляции баланса.
    """
    portfolio_cfg = config.get("portfolios", [{}])[0]
    
    if is_paper:
        config_capital = portfolio_cfg.get("paper_initial_capital", portfolio_cfg.get("initial_capital", 115.0))
        base_state_file = f"paper_state_{ticker}.json"
        shadow_state_file = f"paper_shadow_{ticker}.json"
    else:
        config_capital = portfolio_cfg.get("initial_capital", 80.0)
        base_state_file = f"real_state_{ticker}.json"
        shadow_state_file = f"shadow_state_{ticker}.json"

    base_state_path = BASE_PATH / base_state_file
    shadow_state_path = BASE_PATH / shadow_state_file
    history_dir = BASE_PATH / "history"
    history_dir.mkdir(exist_ok=True)

    old_state = await safe_load_json(str(base_state_path), {})
    
    # Динамическое ребазирование: сохраняем накопленный или урезанный стопом капитал
    final_capital = old_state.get("last_tpv", config_capital) if old_state else config_capital

    if old_state and (old_state.get("trailing_stop_triggered") or old_state.get("trailing_stop_violation_start", 0.0) > 0.0):
        logger.warning(f"📉 Обнаружен Trailing Stop для {ticker}. Капитал ребазирован: {config_capital} -> {final_capital}")

    # Архивируем старые файлы
    for path in [base_state_path, shadow_state_path]:
        if path.exists():
            archive_path = history_dir / f"archive_{ticker}_{int(time.time())}_{path.name}"
            try:
                shutil.copy(path, archive_path)
                logger.info(f"💾 Archived {'paper ' if is_paper else ''}state for {ticker}: {path.name} -> {archive_path.name}")
                os.remove(path)
                logger.info(f"🗑️ Deleted stale state file: {path.name}")
            except Exception as e:
                logger.error(f"Failed to archive {path}: {e}")

    new_state = {
        "virt_qty": 0.0,
        "base_ticker": ticker,
        "siphoning_reserve": 0.0,
        "balance": final_capital,
        "initial_tpv": final_capital,
        "reference_tpv": final_capital,
        "tpv_ath": final_capital,
        "trailing_stop_violation_start": 0.0,
        "trailing_stop_paper_timeout_end": 0.0,
        "trailing_stop_triggered": False,
        "rebalance_cycles": 0,
        "last_rebalance_price": 0.0,
        "started_at": time.time(),
        "virt_debt": 0.0,
        "last_tpv": final_capital,
        "last_profit": 0.0,
        "total_pnl_pct": 0.0,
        "last_update": time.time()
    }
    
    new_shadow_state = {
        "balance": final_capital,
        "positions": {f"{ticker}_LONG": 0.0, f"{ticker}_SHORT": 0.0},
        "last_price": 0.0,
        "base_ticker": ticker,
        "long_entry_price": 0.0,
        "short_entry_price": 0.0,
        "long_liquidation_price": 0.0,
        "short_liquidation_price": 0.0
    }

    await safe_save_json(str(base_state_path), new_state)
    await safe_save_json(str(shadow_state_path), new_shadow_state)
    logger.info(f"✨ Reset {'paper' if is_paper else 'real'} state files for {ticker} to clean initial values (capital: {final_capital}).")

async def start_bot(ticker: str, is_paper: bool = True, config: dict = None):
    prefix = "paper" if is_paper else "real"
    proc_name = f"{prefix}-{ticker.replace('USDT', '').lower()}"
    mode_flag = "--paper" if is_paper else "--real"
    
    # Перед запуском: сбрасываем state для paper (чистый старт), для real — только если файлов нет
    if config:
        if is_paper:
            await reset_bot_state_files(ticker, is_paper=True, config=config)  # Paper always starts fresh
        else:
            state_path = BASE_PATH / f"real_state_{ticker}.json"
            if not state_path.exists():
                await reset_bot_state_files(ticker, is_paper=False, config=config)

    cmd = f'pm2 start main.py --name "{proc_name}" --cwd "{BASE_PATH}" --update-env --interpreter "{sys.executable}" --instances 1 -- --config config.json --ticker {ticker} {mode_flag}'
    await (await asyncio.create_subprocess_shell(cmd)).wait()

def calculate_bot_score(ticker: str, p: dict, is_running_real: bool, is_in_drawdown: bool, min_cycles_required: int) -> float:
    """
    Рассчитывает скоринг для бота с учетом минимальных циклов и защиты от просадки.
    """
    if is_in_drawdown:
        logger.info(f"🛡️ Scored {ticker}: REAL bot in drawdown. LOCKED IN COMBAT (Score: INF).")
        return float('inf')

    cycles = p.get('cycles', 0)
    net_pnl = p.get('profit', 0.0)

    # Фильтр по минимальным циклам для бумажных кандидатов
    if not is_running_real and cycles < min_cycles_required:
        logger.info(f"❌ Scored {ticker}: Rejected (Cycles {cycles} < {min_cycles_required}). (Score: -INF)")
        return -float('inf')

    if net_pnl > 0:
        eff_cycles = max(cycles, min_cycles_required)
        base_score = (float(net_pnl) / eff_cycles) * math.log1p(cycles)
        # Hysteresis: +20% bonus for existing profitable real bots
        sort_eff = base_score * 1.2 if is_running_real else base_score
        logger.info(f"⚖️ Scored {ticker}: Net:{net_pnl:.2f}, Cyc:{cycles}, Score:{sort_eff:.4f}")
        return sort_eff
    else:
        logger.info(f"❌ Scored {ticker}: Unprofitable. (Score: -INF)")
        return -float('inf')


async def selective_merge_incubator(
    old_incubator: list,
    scanner_results: list,
    config: dict,
    perf_map: dict,
) -> list:
    """
    Ротация инкубатора (paper):
    1. Из текущих 20 ботов Супервайзер ранжирует по calculate_bot_score
       (перспективность для реальной торговли: profit/cycles * log(cycles))
    2. Лучшие 10 остаются ВСЕГДА
    3. Худшие 10 могут быть заменены новыми из сканера,
       но не более max_replace_per_cycle (10) за цикл
    4. Сканер уже ранжирован по волатильности — берём первых из топа
    5. При первом старте (old=[]) — запускаем всех max_bots из сканера
    """
    max_bots = config.get("max_bots", 20)
    max_replace = config.get("max_replace_per_cycle", max_bots // 2)
    min_cycles = config.get("min_cycles_for_rank", 10)

    # Топ-max_bots из сканера (ранжированы по волатильности)
    scanner_top = [r['symbol'] for r in scanner_results[:max_bots]]
    scanner_set = set(scanner_top)

    # Первый старт: берём max_bots из сканера
    if not old_incubator:
        final = scanner_top[:max_bots]
        logger.info(f"🚀 First-start: launching {len(final)} bots from scanner: {final}")
        return final

    # 1. Ранжируем ТЕКУЩИХ ботов по calculate_bot_score (перспективность для REAL)
    scored_old = {}
    for t in old_incubator:
        p = perf_map.get(t, {"profit": 0.0, "cycles": 0})
        score = _calc_rotation_score(t, p, min_cycles)
        scored_old[t] = score

    # 2. Боты с положительным PnL — ВСЕГДА остаются в рое (не подлежат замене)
    old_set = set(old_incubator)
    profitable = {t for t in old_incubator if perf_map.get(t, {}).get("profit", 0) > 0}
    # Боты с отрицательным PnL — кандидаты на замену (конкурируют с новыми)
    unprofitable = old_set - profitable

    # 3. Новые тикеры из сканера получают score 0 (нейтральный)
    scanner_new = [t for t in scanner_top if t not in old_set]

    # 4. Формируем финал: сначала прибыльные (всегда остаются), потом лучшие из остальных
    final = list(profitable)  # Прибыльные боты — бессрочно в рое

    # 5. Слоты для остальных: конкуренция между убыточными старыми и новыми
    # Ранжируем убыточных по score
    unprofiled_scored = [(t, scored_old[t]) for t in unprofitable]
    unprofiled_scored.sort(key=lambda x: x[1], reverse=True)

    # Новые с score 0
    new_scored = [(t, 0.0) for t in scanner_new]

    # Объединяем и сортируем
    competitors = unprofiled_scored + new_scored
    competitors.sort(key=lambda x: x[1], reverse=True)

    # Заполняем оставшиеся слоты
    for t, score in competitors:
        if len(final) >= max_bots:
            break
        final.append(t)

    final_set = set(final)
    to_remove = [t for t in old_incubator if t not in final_set]
    to_add = [t for t in final if t not in old_set]

    logger.info(f"🔄 Rotation: keep={len(final) - len(to_add)}, add={len(to_add)}, remove={len(to_remove)}")
    if to_remove:
        logger.info(f"🔄   removing (worst by score): {to_remove}")
    if to_add:
        logger.info(f"🔄   adding (best from scanner): {to_add}")
    logger.info(f"📋 Final incubator ({len(final)} bots): {final}")
    return final


def _calc_rotation_score(ticker: str, p: dict, min_cycles: int) -> float:
    """
    Рассчитывает score для ротации инкубатора (упрощённый calculate_bot_score).
    Для ботов без paper-state (новых) используется proxy данных от сканера.
    """
    cycles = p.get("cycles", 0)
    net_pnl = p.get("profit", 0.0)

    # Фильтр: если cycles < min_cycles и net_pnl <= 0 — скор = -inf
    if cycles < min_cycles and net_pnl <= 0:
        return -float('inf')

    if net_pnl > 0:
        eff_cycles = max(cycles, min_cycles)
        base_score = (float(net_pnl) / eff_cycles) * math.log1p(cycles)
        return base_score
    else:
        # Бот без прибыли — получает низкий, но не бесконечно низкий скор
        # Чтобы новые тикеры с proxy > 0 могли вытеснить убыточных
        return net_pnl * 0.01  # маленький отрицательный скор

async def get_bot_efficiency(ticker: str, config: dict) -> dict:
    """
    Строгий расчет эффективности на основе непрерывного трека инкубатора.
    Обеспечивает доктрину параллельного слежения без рассинхронизации.
    """
    state = await safe_load_json(str(BASE_PATH / f"paper_state_{ticker}.json"), {})
    min_cycles = config.get("min_cycles_for_rank", 10)

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

    # Инициализация коннектора для операций на бирже
    api_key = os.environ.get("BINANCE_API_KEY", config.get("api_key", ""))
    secret_key = os.environ.get("BINANCE_SECRET_KEY", config.get("secret_key", ""))
    connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=config.get("testnet", True))
    await connector.verify_connection()

    # !!! [SAFETY GATE] !!!
    # 0. Принудительная проверка согласованности роя с биржевыми позициями
    healed_tickers = await enforce_swarm_consistency(connector, config)

    # 0.5. Чтение сигналов от ботов (stop/exit flags)
    signals_dir = BASE_PATH / "signals"
    now_ts = time.time()
    toxic_blacklist = config.get("toxic_blacklist", {})
    cooldown_days = config.get("toxic_cooldown_days", config.get("scanner_period_days", 1.0))
    cooldown_sec = cooldown_days * 86400

    if signals_dir.exists():
        # Собираем тикеры со stop-сигналом (убыток) и exit-сигналом (прибыль)
        for flag_prefix in ["stop_", "exit_"]:
            for flag_file in signals_dir.glob(f"{flag_prefix}*.flag"):
                ticker = flag_file.stem[len(flag_prefix):]
                expiry = now_ts + cooldown_sec
                toxic_blacklist[ticker] = expiry
                signal_type = "STOP" if flag_prefix == "stop_" else "EXIT"
                logger.info(f"🚫 {signal_type} signal received for {ticker}. Blacklisted until {time.ctime(expiry)}")

                try:
                    flag_file.unlink()
                except Exception:
                    pass

        # Prune expired entries from toxic_blacklist
        toxic_blacklist = {s: exp for s, exp in toxic_blacklist.items() if exp > now_ts}
        config["toxic_blacklist"] = toxic_blacklist

    # [CACHE] Один вызов pm2 jlist на цикл — результат переиспользуется везде
    running_bots = await get_running_bots_info()

    # Дополняем кэш процессов "оздоровленными" тикерами, чтобы защитить их от PM2 latency
    for t in healed_tickers:
        r_key = f"r_{t}"
        if r_key not in running_bots:
            running_bots[r_key] = {"name": f"real-{t.replace('USDT', '').lower()}", "paper": False}

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

    # 2. Формирование списка Инкубатора (Selective Rotation)
    # Сначала фильтруем сканер от toxic, потом мержим с текущим инкубатором
    toxic_blacklist = config.get("toxic_blacklist", {})
    now = time.time()
    cooldown_sec = config.get("toxic_cooldown_days", config.get("scanner_period_days", 1.0)) * 86400

    # Обновляем toxic_blacklist от сканера
    for r in scanner_results:
        symbol = r['symbol']
        if r.get('is_toxic') and symbol not in toxic_blacklist:
            expiry = now + cooldown_sec
            toxic_blacklist[symbol] = expiry
            logger.info(f"🚫 {symbol} marked toxic by scanner. Blacklisted until {time.ctime(expiry)}")

    config["toxic_blacklist"] = toxic_blacklist

    # "Чистый" результат сканера (без toxic) — для selective_merge
    clean_scanner = []
    for r in scanner_results:
        symbol = r['symbol']
        if symbol in toxic_blacklist:
            logger.info(f"⏳ {symbol} is in Toxic Quarantine. Skipping.")
            continue
        clean_scanner.append(r)

    # [FIX-1] Старый инкубатор: статический список из конфига + реально запущенные paper-боты из PM2
    # Раньше брался ТОЛЬКО из config.json — если тикер был добавлен вручную через PM2 (как DEXEUSDT),
    # он терялся при ротации и мог быть убит супервайзером
    running_paper_from_pm2 = [k.replace("p_", "") for k in running_bots.keys() if k.startswith("p_")]
    old_incubator = list(set(config.get("tickers", [])).union(running_paper_from_pm2))

    # Собираем метрики для старых тикеров (для оценки PnL в ротации)
    old_perf_tasks = [get_bot_efficiency(t, config) for t in old_incubator]
    old_perf_results = await asyncio.gather(*old_perf_tasks)
    global_perf_map = dict(zip(old_incubator, old_perf_results))

    # Selective Rotation: мержим старых + новых
    limit_bots = config.get("max_bots", 20)
    final_incubator = await selective_merge_incubator(
        old_incubator=old_incubator,
        scanner_results=clean_scanner,
        config=config,
        perf_map=global_perf_map,
    )

    logger.info(f"📋 Incubator ready: {final_incubator}")

    # 3. Выбор Чемпионов для REAL
    # [FIX-2] Собираем метрики для ВСЕХ кандидатов (final_incubator + текущие REAL боты),
    # а не только для final_incubator. Иначе REAL бот, выпавший из инкубатора,
    # получает fallback profit=0.0 → -INF → автоматический стоп.
    # [OPTIMIZATION] Повторно используем running_bots из начала цикла (строка ~424) —
    # читаем с диска только тикеров, которых ещё нет в global_perf_map.
    current_real_tickers = [k.replace("r_", "") for k in running_bots.keys() if k.startswith("r_")]
    # Объединяем с только что восстановленными ботами для компенсации задержки старта PM2
    current_real_tickers = list(set(current_real_tickers).union(healed_tickers))
    all_evaluated_tickers = set(final_incubator).union(current_real_tickers)

    missing_tickers = [t for t in all_evaluated_tickers if t not in global_perf_map]
    if missing_tickers:
        logger.info(f"📊 Fetching missing performance data for {len(missing_tickers)} candidates...")
        missing_tasks = [get_bot_efficiency(t, config) for t in missing_tickers]
        missing_results = await asyncio.gather(*missing_tasks)
        global_perf_map.update(dict(zip(missing_tickers, missing_results)))

    perf_map = global_perf_map

    # Сортировка по эффективности
    replacement_threshold = config.get("replacement_efficiency_threshold_pct", 20.0)
    min_cycles_required = config.get("min_cycles_for_rank", 10)
    real_whitelist = set(config.get("real_whitelist", []))

    ready_pool = []
    use_whitelist = config.get("use_real_whitelist", True)

    for ticker in all_evaluated_tickers:
        # Whitelist check
        if use_whitelist and ticker not in real_whitelist and ticker not in current_real_tickers:
            logger.info(f"🚫 Skipping {ticker}: Not in real_whitelist and not a currently running real bot.")
            continue

        # 1. Fetch data
        is_running_real = ticker in current_real_tickers
        p = perf_map.get(ticker)

        # [FIX-3] Fallback Guard: если REAL бот не попал в perf_map,
        # читаем real_state вместо fallback на нули.
        if p is None and is_running_real:
            real_state = await safe_load_json(str(BASE_PATH / f"real_state_{ticker}.json"), {})
            if real_state:
                p = {
                    "profit": real_state.get("last_profit", 0.0),
                    "cycles": real_state.get("rebalance_cycles", 0),
                    "eff": 0.0,
                    "trailing_stop_paper_timeout_end": 0.0,
                }
                logger.info(f"🛡️ Fallback to real_state for {ticker}: profit={p['profit']:.2f}")
            else:
                p = {"profit": 0.0, "cycles": 0, "eff": 0.0, "trailing_stop_paper_timeout_end": 0.0}
        elif p is None:
            p = {"profit": 0.0, "cycles": 0, "eff": 0.0, "trailing_stop_paper_timeout_end": 0.0}

        # 2. Strict Drawdown Protection (Highest Priority)
        is_in_drawdown = False
        if is_running_real:
            real_state = await safe_load_json(str(BASE_PATH / f"real_state_{ticker}.json"), {})
            if real_state.get("last_profit", 0.0) < 0:
                is_in_drawdown = True

        # 3. Trailing Stop Check (Only for paper candidates)
        if not is_running_real and time.time() < p.get('trailing_stop_paper_timeout_end', 0.0):
            logger.info(f"⏳ Skipping {ticker}: Still in trailing stop paper timeout.")
            continue

        # 4. Scoring Logic
        sort_eff = calculate_bot_score(
            ticker=ticker,
            p=p,
            is_running_real=is_running_real,
            is_in_drawdown=is_in_drawdown,
            min_cycles_required=min_cycles_required
        )

        p['sort_eff'] = sort_eff
        perf_map[ticker] = p
        if sort_eff > -float('inf'):
            ready_pool.append(ticker)

    ready_pool.sort(key=lambda x: perf_map.get(x, {}).get('sort_eff', -float('inf')), reverse=True)
    max_real_slots = max(0, config.get("max_bots", 10) - config.get("paper_mode_bots", 9))

    # Sophisticated substitution logic: hysteresis, profit protection, and score cushion
    current_real_tickers = [k.replace("r_", "") for k in running_bots.keys() if k.startswith("r_")]

    # 1. Start with currently running bots
    target_real_bots = list(current_real_tickers)

    # 2. Identify candidates from incubator (ready_pool)
    # Candidates are those in ready_pool NOT currently running as REAL
    incubator_candidates = [t for t in ready_pool if t not in current_real_tickers]

    # 3. Filling empty slots
    while len(target_real_bots) < max_real_slots and incubator_candidates:
        best_cand = incubator_candidates.pop(0)
        if best_cand not in target_real_bots:
            target_real_bots.append(best_cand)
            logger.info(f"✅ Filling empty REAL slot with {best_cand}")

    # 2. Evaluate replacing existing REAL bots with better candidates
    if incubator_candidates:
        real_bots_scoring = []
        for t in current_real_tickers:
            p = perf_map.get(t, {})
            score = p.get('sort_eff', -float('inf'))
            profit = p.get('profit', 0.0)
            cycles = p.get('cycles', 0)
            real_bots_scoring.append((t, score, profit, cycles))

        # Sort: worst score first for potential replacement
        real_bots_sorted = sorted(real_bots_scoring, key=lambda x: (x[1], x[2]))

        max_replace = config.get("max_replace_per_cycle", 1)
        replacements_count = 0

        for ticker, score, profit, cycles in real_bots_sorted:
            if not incubator_candidates:
                break

            # Skip if score is INF (drawdown protection - locked in combat)
            if score == float('inf'):
                continue

            # Сбор метрик времени жизни бота для гистерезиса
            state_path = BASE_PATH / f"real_state_{ticker}.json"
            started_at = time.time()
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                        started_at = sdata.get("started_at", time.time())
                except Exception:
                    pass

            age_hours = (time.time() - started_at) / 3600.0
            probation_days = config.get("probation_period_days", 0.01)
            probation_hours = probation_days * 24.0

            # Если у бота много циклов, он не на испытательном сроке, даже если стейт файл новый
            is_probation = age_hours < probation_hours and cycles < config.get("min_cycles_for_rank", 20)

            best_cand = incubator_candidates[0]
            cand_score = perf_map.get(best_cand, {}).get('sort_eff', -float('inf'))

            logger.info(
                f"🧐 Evaluating REAL {ticker} [Score: {score:.4f}, Profit: ${profit:.2f}, Age: {age_hours:.2f}h] "
                f"vs Candidate {best_cand} [Score: {cand_score:.4f}]"
            )

            # Защита 1: Временной гистерезис (Испытательный срок)
            if is_probation:
                logger.info(f"🛡️ Hysteresis Guard: Protecting {ticker}. Age: {age_hours:.2f}h, Cycles: {cycles} < Min Cycles.")
                continue

            # Защита 2: Защита прибыльных позиций
            if profit > 0.0:
                logger.info(f"🛡️ Profit Guard: Protecting {ticker} from replacement because it has positive profit (${profit:.2f})")
                continue

            # Защита 3: Порог изменения скоринга (Score Cushion)
            SCORE_CUSHION = 0.25
            if cand_score > (score + SCORE_CUSHION):
                logger.info(f"♻️ Substitution Triggered: Replacing {ticker} with {best_cand} (Score delta {cand_score - score:.4f} > Cushion {SCORE_CUSHION})")
                if ticker in target_real_bots:
                    target_real_bots.remove(ticker)
                target_real_bots.append(best_cand)
                incubator_candidates.pop(0)
                replacements_count += 1
                if replacements_count >= max_replace:
                    break
            else:
                logger.info(f"⏭️ Skipping Replacement: Candidate {best_cand} score is not high enough to warrant rotation (Required cushion: +{SCORE_CUSHION})")

    # 3. Final safety check: if we somehow have more bots than slots (e.g. config change), trim the worst ones
    if len(target_real_bots) > max_real_slots:
        target_real_bots_scored = []
        for t in target_real_bots:
            target_real_bots_scored.append((t, perf_map.get(t, {}).get('sort_eff', -float('inf'))))
        # Keep INF scores (drawdown protection) and then best scores
        target_real_bots_scored.sort(key=lambda x: x[1], reverse=True)
        target_real_bots = [t for t, s in target_real_bots_scored[:max_real_slots]]
        logger.info(f"✂️ Reduced REAL swarm to {max_real_slots} slots by removing least efficient bots.")
    # -------------------------------------------------------------------------
    # 3.5. [Authoritative Cleanup] Close exchange positions for those not in live_swarm/whitelist,
    # even if PM2 process is missing (stray positions).
    real_whitelist = config.get("real_whitelist", [])
    active_positions = await connector.get_positions()
    allowed_cleanup_tickers = set(target_real_bots).union(set(real_whitelist))
    for pos_key in active_positions.keys():
        # pos_key is like "SYMBOL_SIDE"
        ticker = pos_key.split('_')[0]
        if ticker not in allowed_cleanup_tickers:
            qty = active_positions[pos_key].get("qty", 0.0)
            if float(qty) != 0:
                logger.warning(f"🧹 Authoritative Cleanup: Found stray position for {ticker} (not in live_swarm or whitelist). Closing it!")
                cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
                await (await asyncio.create_subprocess_shell(cmd_stop)).wait()

    # 4. Исполнение в PM2: Доктрина Параллельного Слежения (Защищенная версия)
    # -------------------------------------------------------------------------

    # Создаем мутабельное множество текущих запущенных ключей для исключения рассинхрона
    active_running_keys = set(running_bots.keys())

    # Сначала гасим тех, кто вылетел
    real_whitelist = config.get("real_whitelist", [])
    for key, info in list(running_bots.items()):
        ticker = key.split("_")[1]

        if key.startswith("p_") and ticker not in final_incubator:
            logger.info(f"🛑 Stopping Incubator (Out of Scanner): {ticker}")
            await stop_bot(ticker, is_paper=True)
            active_running_keys.discard(key)

        if key.startswith("r_") and ticker not in target_real_bots:
            if ticker in real_whitelist:
                logger.info(f"🛡️ Whitelist Guard: {ticker} is not in target_real_bots but is explicitly whitelisted. Skipping termination.")
                continue

            logger.info(f"🛑 Stopping Combat REAL process: {ticker} (Rolling back to pure paper tracking)")
            await stop_bot(ticker, is_paper=False)
            active_running_keys.discard(key)

            # Экстренно закрываем позиции на бирже для этого тикера
            cmd_stop = f'"{sys.executable}" "{BASE_PATH / "main.py"}" --config config.json --ticker {ticker} --stop --real'
            await (await asyncio.create_subprocess_shell(cmd_stop)).wait()

    # Пауза для стабильности дескрипторов PM2
    await asyncio.sleep(1.5)

    # ЗАПУСК: Инкубатор (PAPER) работает ВСЕГДА для всех тикеров из сканера
    for ticker in final_incubator:
        p_key = f"p_{ticker}"
        if p_key not in active_running_keys:
            logger.info(f"🚀 [A] Launching Continuous Incubator (PAPER): {ticker}")
            await start_bot(ticker, is_paper=True, config=config)
            active_running_keys.add(p_key) # Жестко фиксируем запуск локально

    # ЗАПУСК: Боевые боты (REAL) включаются ПАРАЛЛЕЛЬНО к бумажным
    for ticker in target_real_bots:
        r_key = f"r_{ticker}"
        if r_key not in active_running_keys:
            logger.info(f"🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): {ticker}")
            await start_bot(ticker, is_paper=False, config=config)
            active_running_keys.add(r_key) # Жестко фиксируем запуск локально

    # 5. Сохранение конфига: СНАЧАЛА пишем целевой пул, ЗАТЕМ прогоняем через Invariant Gate
    config["live_swarm"] = sorted(target_real_bots)

    # --- [INVARIANT GATE] ---
    await enforce_invariant_gate(connector, config)
    # Safety: if tickers is empty, restore from scanner to avoid losing all tickers
    if not config.get("tickers"):
        fallback = [r['symbol'] for r in scanner_results[:config.get("max_bots", 20)]]
        config["tickers"] = sorted(fallback)
        config["base_ticker"] = config["tickers"][0] if config["tickers"] else "ALGOUSDT"
        logger.warning(f"⚠️ tickers was empty — restored {len(config['tickers'])} tickers from scanner")
    await safe_save_json(CONFIG_PATH, config)
    await (await asyncio.create_subprocess_shell("pm2 save")).wait()
    logger.info(f"Cycle Complete. REAL Swarm: {config['live_swarm']}")

    # -------------------------------------------------------------------------
    # 5.5. [GUARD] Ensure real bots from live_swarm are actually running in PM2.
    # After supervisor restart, PM2 may have lost processes but config still has
    # them in live_swarm. This detects missing processes and relaunches them.
    # -------------------------------------------------------------------------
    await _ensure_real_bots_alive(config)


async def _ensure_real_bots_alive(config: dict):
    """
    Verifies that all tickers in live_swarm have running PM2 processes.
    If a real bot is missing from PM2, relaunches it.
    This handles the case where supervisor restart kills all PM2 processes
    but live_swarm config still lists them.
    """
    live_swarm = config.get("live_swarm", [])
    if not live_swarm:
        return

    running_bots = await get_running_bots_info()
    running_real = {k.replace("r_", "") for k in running_bots if k.startswith("r_")}

    for ticker in live_swarm:
        if ticker not in running_real:
            logger.warning(
                f"🔄 Real bot {ticker} is in live_swarm but not running in PM2. Restarting..."
            )
            try:
                await start_bot(ticker, is_paper=False, config=config)
                logger.info(f"✅ Restarted real bot {ticker}")
            except Exception as e:
                logger.error(f"❌ Failed to restart real bot {ticker}: {e}")

if __name__ == "__main__":
    asyncio.run(manage_swarm())