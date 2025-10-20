#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper Trader (Real-Time DB Feed)
--------------------------------
Читает индекс эпизодов из stream_backtest_engine → воспроизводит сделки
в режиме "реального времени" (sleep) или ASAP, подгружая минутки из БД.
Работает ТОЛЬКО с указанной в конфиге моделью (без fallback).

Выход: output/<config_name>/{trades.csv, metrics.json}
Правила проекта: конфиги строго из configs/*.py, даты — UTC, суммы — USDT.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Protocol, Any

import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from tqdm import tqdm
from config import MasterConfig
from utils import load_config as load_master_config

# ------------------------------ Utils ------------------------------

def _load_py_module(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить конфиг: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _to_utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = dtparser.isoparse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Index должен быть DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

# ------------------------------ Config -----------------------------

@dataclass
class ExecParams:
    base_capital_usdt: float
    risk_per_trade_pct: float
    fee_bps: float
    slippage_bps: float
    max_concurrent: int

@dataclass
class PTParams:
    mode: str  # "realtime" | "asap"
    cap_windows_per_symbol: int

@dataclass
class InferenceParams:
    policy_loader: str
    checkpoint_path: str
    strict: bool

@dataclass
class Cfg:
    config_name: str
    db_provider_path: str
    index_csv: str
    exec: ExecParams
    pt: PTParams
    inf: InferenceParams

def _load_cfg(cfg_path: str) -> Cfg:
    mod = _load_py_module(cfg_path)
    if not hasattr(mod, "data") or not isinstance(mod.data, dict):
        raise RuntimeError("В конфиге нужен dict `data`.")
    data = mod.data
    # обязательные части (см. SYSTEM_PROMPT.md / README)
    dbp = data["db_provider"]
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
    index_csv = os.path.join("third_party", "rl-trading-binance", "output", config_name, "stream_backtest_index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Не найден индекс эпизодов: {index_csv}")
    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
    ex_d = data.get("exec", {})
    inf_d = data.get("inference", {})
    execp = ExecParams(
        base_capital_usdt=float(ex_d.get("base_capital_usdt", 10000.0)),
        risk_per_trade_pct=float(ex_d.get("risk_per_trade_pct", 1.0)),
        fee_bps=float(ex_d.get("fee_bps", 2.0)),
        slippage_bps=float(ex_d.get("slippage_bps", 1.0)),
        max_concurrent=int(ex_d.get("max_concurrent", 4)),
    )
    ptp = PTParams(
        mode=str(pt_d.get("mode", "asap")),
        cap_windows_per_symbol=int(pt_d.get("cap_windows_per_symbol", 0)),
    )
    infp = InferenceParams(
        policy_loader=str(inf_d.get("policy_loader", "")),
        checkpoint_path=str(inf_d.get("checkpoint_path", "")),
        strict=bool(inf_d.get("strict", False)),
    )
    return Cfg(config_name, dbp, index_csv, execp, ptp, infp)

# ------------------------------ Provider ---------------------------

ProviderFn = callable

def _load_db_provider(path: str) -> ProviderFn:
    if ":" not in path:
        raise RuntimeError("`data.db_provider` должен быть 'module:function'.")
    mod_path, fn_name = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, fn_name):
        raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
    return getattr(mod, fn_name)

# ------------------------------ Strategy / Policy -------------------
# Плагинная политика инференса; fallback ОТСУТСТВУЕТ (модель обязательна).

class Policy(Protocol):
    def predict(self, symbol: str, ctx_df: pd.DataFrame) -> str:
        """Вернуть 'BUY' или 'SELL' по данным контекста."""
        ...

def _load_policy(loader_path: str, ckpt_path: str, cfg: MasterConfig) -> Policy | None:
    try:
        if ":" not in loader_path:
            raise RuntimeError("`inference.policy_loader` должен быть 'module:function'.")
        mod_path, fn_name = loader_path.split(":", 1)
        mod = importlib.import_module(mod_path)
        if not hasattr(mod, fn_name):
            raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
        loader = getattr(mod, fn_name)
        policy = loader(ckpt_path, cfg)  # type: ignore
        return policy
    except Exception as e:
        raise RuntimeError(f"[paper_trader] Ошибка загрузки политики '{loader_path}': {e}")

# ------------------------------ Execution helpers -----------------

def _apply_slippage(price: float, bps: float, side: str) -> float:
    # bps = basis points (0.01% = 1 bps). Для покупки повышаем цену, для продажи понижаем.
    delta = price * (bps / 10000.0)
    return price + delta if side == "BUY" else price - delta

def _fees_cost(notional: float, fee_bps: float) -> float:
    return notional * (fee_bps / 10000.0)

def _position_size(capital: float, risk_pct: float, entry: float) -> float:
    risk_usdt = capital * (risk_pct / 100.0)
    qty = max(risk_usdt / max(entry, 1e-12), 0.0)
    return qty

def _compute_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "net_pnl_usdt": 0.0}
    pnl = trades["net_pnl_usdt"].values
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = (wins / max(losses, 1e-12)) if losses > 0 else float("inf")
    win_rate = float((pnl > 0).mean()) * 100.0
    equity = pnl.cumsum()
    peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    dd = (equity - peak)
    max_dd = float(dd.min())
    max_dd_pct = (abs(max_dd) / max(1.0, (np.max(peak) if len(peak) else 1.0))) * 100.0
    # простая минутная дискретизация: std по трейдам как приближение
    sharpe = float((np.mean(pnl) / (np.std(pnl) + 1e-12)) * np.sqrt(252))  # приближение к дневной
    return {
        "trades": int(len(trades)),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(pf, 4),
        "max_dd_pct": round(max_dd_pct, 3),
        "sharpe": round(sharpe, 4),
        "net_pnl_usdt": round(float(pnl.sum()), 2),
    }

# ------------------------------ Main --------------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Использование: python paper_trader.py configs/alpha.py")
        return 2
    cfg_path = argv[1]
    cfg = _load_cfg(cfg_path)
    master_cfg = load_master_config(cfg_path)
    out_dir = os.path.join("third_party", "rl-trading-binance", "output", cfg.config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_trades = os.path.join(out_dir, "trades.csv")
    out_metrics = os.path.join(out_dir, "metrics.json")

    provider = _load_db_provider(cfg.db_provider_path)
    policy = _load_policy(cfg.inf.policy_loader, cfg.inf.checkpoint_path, master_cfg)
    idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])
    # Опциональный "колпак" на окна в день/тикер
    if cfg.pt.cap_windows_per_symbol > 0:
        keep_rows = []
        for sym, g in idx.groupby("symbol"):
            g = g.sort_values("session_start")
            g["d"] = g["session_start"].dt.floor("D")
            g = g.groupby("d").head(cfg.pt.cap_windows_per_symbol).drop(columns=["d"])
            keep_rows.append(g)
        idx = pd.concat(keep_rows, ignore_index=True)

    trades_rows: List[Dict[str, object]] = []
    capital = cfg.exec.base_capital_usdt

    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Paper trading"):
        sym = row["symbol"]
        ctx_start = _to_utc(row["ctx_start"])
        ctx_end = _to_utc(row["ctx_end"])
        ses_start = _to_utc(row["session_start"])
        ses_end = _to_utc(row["session_end"])
        # Подгружаем диапазон с КОНТЕКСТОМ и СЕССИЕЙ (для инференса модели):
        feed = dict(provider([sym], ctx_start.isoformat(), ses_end.isoformat()))
        if sym not in feed or feed[sym].empty:
            continue
        df = _ensure_utc_index(feed[sym]).sort_index()
        # Минимальные проверки покрытия ключевых временных меток
        last_ts = ses_end - pd.Timedelta(minutes=1)
        if df.index[0] > ctx_start or df.index[-1] < last_ts:
            # неполное покрытие — пропустим окно
            continue
        first_px = float(df.loc[ses_start:ses_start].iloc[0]["close"])
        # Выбор стороны: ТОЛЬКО модель (никаких fallback).
        ctx_slice = df.loc[ctx_start:ctx_end]
        if len(ctx_slice) != master_cfg.seq.agent_history_len:
            # Пропускаем, если окно контекста неполное
            continue
        try:
            side = str(policy.predict(sym, ctx_slice))
        except Exception as e:
            raise RuntimeError(f"[paper_trader] policy.predict() error for {sym} @ {ctx_end}: {e}")
        if side not in ("BUY", "SELL"):
            raise RuntimeError(f"[paper_trader] policy returned invalid action: {side!r}")
        # Исполнение
        entry_raw = first_px
        entry_px = _apply_slippage(entry_raw, cfg.exec.slippage_bps, side)
        qty = _position_size(capital, cfg.exec.risk_per_trade_pct, entry_px)
        # Выход по ПОСЛЕДНЕЙ минуте сессии (исключаем бар, начинающийся в session_end)
        last_px = float(df.loc[last_ts:last_ts].iloc[-1]["close"])
        exit_px = _apply_slippage(last_px, cfg.exec.slippage_bps, "SELL" if side=="BUY" else "BUY")
        notional_entry = qty * entry_px
        notional_exit = qty * exit_px
        gross = (notional_exit - notional_entry) if side == "BUY" else (notional_entry - notional_exit)
        fees = _fees_cost(notional_entry, cfg.exec.fee_bps) + _fees_cost(notional_exit, cfg.exec.fee_bps)
        net = gross - fees
        trades_rows.append({
            "symbol": sym,
            "entry_time": ses_start.isoformat(),
            "exit_time": ses_end.isoformat(),
            "side": side,
            "qty": round(qty, 8),
            "entry_price": round(entry_px, 8),
            "exit_price": round(exit_px, 8),
            "gross_pnl_usdt": round(gross, 2),
            "fees_usdt": round(fees, 2),
            "net_pnl_usdt": round(net, 2),
        })
        # ASAP vs realtime: в этой версии не делаем sleep — режим "realtime" можно включить позже

    trades = pd.DataFrame(trades_rows)
    trades.to_csv(out_trades, index=False)
    metrics = _compute_metrics(trades)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[paper_trader] trades: {len(trades)}  net_pnl: {metrics.get('net_pnl_usdt',0):.2f} USDT  PF: {metrics.get('profit_factor',0)}  WinRate: {metrics.get('win_rate',0)}%")
    print(f"[paper_trader] saved: {out_trades}")
    print(f"[paper_trader] saved: {out_metrics}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
