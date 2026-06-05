#!/usr/bin/env python3
"""
Data Collector — сбор метрик из файлов состояния и логов (READ-ONLY).

НЕ изменяет файлы бота. Только читает:
  - state-файлы (paper_state_*.json, real_state_*.json)
  - shadow-файлы (paper_shadow_*.json, shadow_state_*.json)
  - логи (logs/*.log, PM2 logs)
  - PM2 процессы (pm2 jlist)
  - config.json (только чтение)
"""

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

# ─── Config ───────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).parent
CONFIG_PATH = DASHBOARD_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PROJECT_PATH = Path(config["paths"]["project"])
LOGS_PATH = Path(config["paths"]["logs"])
DATA_PATH = Path(config["paths"]["data"])
PM2_LOGS_PATH = Path(config["paths"]["pm2_logs"])

HEARTBEAT_MAX_AGE_SEC = 120


# ─── Helpers ──────────────────────────────────────────────────


def _read_json(path: Path) -> Optional[dict]:
    """Безопасное чтение JSON."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _find_state_files(mode: str) -> list[str]:
    """Найти все тикеры по state-файлам."""
    pattern = f"{mode}_state_*.json"
    files = list(PROJECT_PATH.glob(pattern))
    return [f.stem.replace(f"{mode}_state_", "") for f in files]


def _parse_log_timestamp(line: str) -> Optional[datetime]:
    """Парсинг таймстампа из строки лога."""
    try:
        ts_str = line[:19]
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return None


# ─── System Overview ──────────────────────────────────────────


def get_system_overview() -> dict:
    """Обзор всей системы — для главной страницы."""
    now = datetime.now()

    # PM2 процессы
    pm2_procs = _get_pm2_list()

    # Тикеры
    paper_tickers = _find_state_files("paper")
    real_tickers = _find_state_files("real")

    # Собираем данные по каждому тикеру
    paper_bots = []
    for t in paper_tickers:
        paper_bots.append(_get_ticker_summary(t, "paper"))

    real_bots = []
    for t in real_tickers:
        real_bots.append(_get_ticker_summary(t, "real"))

    # Агрегированные метрики
    total_tpv = sum(b.get("tpv", 0) for b in real_bots)
    total_pnl = sum(b.get("pnl", 0) for b in real_bots)
    total_initial = sum(b.get("initial_capital", 0) for b in real_bots)

    # Supervisor статус
    supervisor_status = "offline"
    for p in pm2_procs:
        if p.get("name") == "supervisor-service":
            supervisor_status = p.get("status", "unknown")
            break

    return {
        "timestamp": now.isoformat(),
        "supervisor": supervisor_status,
        "summary": {
            "paper_count": len(paper_bots),
            "real_count": len(real_bots),
            "total_tpv": round(total_tpv, 2),
            "total_pnl": round(total_pnl, 2),
            "total_initial": round(total_initial, 2),
            "total_pnl_pct": round(
                (total_pnl / total_initial * 100) if total_initial > 0 else 0, 2
            ),
        },
        "paper_bots": paper_bots,
        "real_bots": real_bots,
        "pm2_processes": pm2_procs,
        "alerts": _get_alerts(paper_bots, real_bots, pm2_procs),
    }


def _get_ticker_summary(ticker: str, mode: str) -> dict:
    """Краткая сводка по тикеру."""
    state = _read_json(PROJECT_PATH / f"{mode}_state_{ticker}.json")
    shadow = _read_json(PROJECT_PATH / f"{mode}_shadow_{ticker}.json")

    if not state:
        return {"ticker": ticker, "mode": mode, "status": "no_state"}

    tpv = state.get("tpv", 0)
    initial = state.get("initial_capital", 0)
    pnl = state.get("pnl", tpv - initial if initial else 0)
    pnl_pct = (pnl / initial * 100) if initial > 0 else 0
    cycles = state.get("rebalance_cycles", state.get("cycles", 0))

    # Heartbeat
    heartbeat_ok, heartbeat_age = _check_heartbeat(ticker, mode)

    # Статус
    status = "ok"
    if not heartbeat_ok:
        status = "stale"
    if state.get("pnl_guard_active"):
        status = "pnl_guard"

    # Позиции из shadow
    positions = {}
    if shadow:
        positions = shadow.get("positions", shadow.get("pos", {}))

    return {
        "ticker": ticker,
        "mode": mode,
        "status": status,
        "tpv": round(tpv, 2),
        "initial_capital": round(initial, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "cycles": cycles,
        "heartbeat_ok": heartbeat_ok,
        "heartbeat_age_sec": heartbeat_age,
        "positions": positions,
        "last_update": state.get("last_update", state.get("timestamp", "")),
    }


def _check_heartbeat(ticker: str, mode: str) -> tuple:
    """Проверка свежести heartbeat."""
    log_file = LOGS_PATH / f"{mode}_{ticker}.log"
    if not log_file.exists():
        return False, None

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-100:]

        for line in reversed(lines):
            if "Heartbeat" in line or "heartbeat" in line:
                ts = _parse_log_timestamp(line)
                if ts:
                    age = (datetime.now() - ts).total_seconds()
                    return age < HEARTBEAT_MAX_AGE_SEC, int(age)
    except OSError:
        pass

    return False, None


def _get_alerts(paper_bots: list, real_bots: list, pm2_procs: list) -> list:
    """Сбор алертов."""
    alerts = []

    # Supervisor
    sup = next(
        (p for p in pm2_procs if p.get("name") == "supervisor-service"), None
    )
    if not sup:
        alerts.append({
            "level": "critical",
            "message": "Supervisor не найден в PM2",
        })
    elif sup.get("status") != "online":
        alerts.append({
            "level": "critical",
            "message": f"Supervisor статус: {sup.get('status')}",
        })

    # Real bots
    for bot in real_bots:
        if bot.get("status") == "stale":
            alerts.append({
                "level": "warning",
                "message": (
                    f"REAL {bot['ticker']}: heartbeat stale "
                    f"({bot.get('heartbeat_age_sec', '?')}s)"
                ),
            })
        if bot.get("pnl_pct", 0) < -10:
            alerts.append({
                "level": "warning",
                "message": (
                    f"REAL {bot['ticker']}: PnL {bot['pnl_pct']:.1f}%"
                ),
            })

    return alerts


# ─── Ticker Details ───────────────────────────────────────────


def get_ticker_details(mode: str, ticker: str) -> dict:
    """Детальная информация по тикеру."""
    state = _read_json(PROJECT_PATH / f"{mode}_state_{ticker}.json")
    shadow = _read_json(PROJECT_PATH / f"{mode}_shadow_{ticker}.json")

    if not state:
        return {"error": f"State file not found for {mode}/{ticker}"}

    # PM2 процесс
    pm2_proc = None
    pm2_name = f"{mode}-{ticker.lower()}"
    for p in _get_pm2_list():
        if p.get("name") == pm2_name:
            pm2_proc = p
            break

    # Heartbeat
    hb_ok, hb_age = _check_heartbeat(ticker, mode)

    # Последние записи лога
    log_tail = _read_log_lines(ticker, mode, 50)

    return {
        "ticker": ticker,
        "mode": mode,
        "state": state,
        "shadow": shadow,
        "pm2": pm2_proc,
        "heartbeat": {"ok": hb_ok, "age_sec": hb_age},
        "log_tail": log_tail,
    }


# ─── History ──────────────────────────────────────────────────


def get_history_data() -> dict:
    """Исторические данные — PnL по дням из state-файлов."""
    # Собираем текущие значения всех real ботов
    real_tickers = _find_state_files("real")
    history = []

    for t in real_tickers:
        state = _read_json(PROJECT_PATH / f"real_state_{t}.json")
        if state:
            tpv = state.get("tpv", 0)
            initial = state.get("initial_capital", 0)
            pnl = tpv - initial if initial else 0
            history.append({
                "ticker": t,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "tpv": round(tpv, 2),
                "initial": round(initial, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(
                    (pnl / initial * 100) if initial > 0 else 0, 2
                ),
            })

    return {"history": history, "timestamp": datetime.now().isoformat()}


# ─── Logs ─────────────────────────────────────────────────────


def get_log_tail(
    mode: str, ticker: str, lines: int = 100, offset: int = 0
) -> dict:
    """Последние строки лога."""
    log_file = LOGS_PATH / f"{mode}_{ticker}.log"

    if not log_file.exists():
        # Проверяем PM2 логи
        pm2_logs = list(PM2_LOGS_PATH.glob(f"{mode}-{ticker.lower()}*.log"))
        if pm2_logs:
            log_file = pm2_logs[0]
        else:
            return {"lines": [], "error": "Log file not found"}

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        start = max(0, len(all_lines) - lines - offset)
        end = len(all_lines) - offset if offset > 0 else len(all_lines)
        selected = all_lines[start:end]

        return {
            "lines": [line.rstrip("\n") for line in selected],
            "total": len(all_lines),
            "file": str(log_file),
        }
    except OSError as e:
        return {"lines": [], "error": str(e)}


# ─── PM2 ──────────────────────────────────────────────────────


def get_pm2_processes() -> list:
    """Список PM2 процессов."""
    return _get_pm2_list()


def _get_pm2_list() -> list:
    """Получить список PM2 процессов."""
    try:
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            procs = json.loads(result.stdout)
            return [
                {
                    "name": p.get("name", ""),
                    "status": p.get("pm2_env", {}).get("status", "unknown"),
                    "restarts": p.get("pm2_env", {}).get("restart_time", 0),
                    "uptime": p.get("pm2_env", {}).get("pm_uptime", 0),
                    "memory": p.get("monit", {}).get("memory", 0),
                    "cpu": p.get("monit", {}).get("cpu", 0),
                }
                for p in procs
            ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return []


# ─── Config (read-only) ───────────────────────────────────────


def get_config_summary() -> dict:
    """Безопасная выдержка из config.json (без секретов)."""
    config_path = PROJECT_PATH / "config.json"
    cfg = _read_json(config_path)
    if not cfg:
        return {"error": "config.json not found"}

    # Только безопасные поля
    safe_keys = [
        "initial_capital", "leverage", "paper_initial_capital",
        "rebalance_threshold_surplus", "rebalance_threshold_deficit",
        "max_drawdown_limit", "equity_trailing_stop_pct",
        "equity_trailing_stop_activation_pct",
        "max_price_velocity_pct", "net_move_block_pct",
        "liquidation_distance_warn_pct", "liquidation_distance_crit_pct",
        "margin_ratio_warning", "margin_ratio_critical",
        "max_bots", "max_replace_per_cycle", "min_cycles_for_rank",
        "use_real_whitelist", "real_whitelist",
    ]
    return {k: cfg.get(k) for k in safe_keys if k in cfg}
