#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper Trader (Real-Time DB Feed)
--------------------------------
Читает индекс эпизодов из stream_backtest_engine → воспроизводит сделки
в режиме "реального времени" (sleep) или ASAP, подгружая минутки из БД
только на период сессии. Стратегия по умолчанию: Follow-Context.

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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Callable, Protocol, Any
import inspect

import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from tqdm import tqdm
from utils import find_spike_windows, calculate_normalization_stats # детектор + нормировка

# Глобальная переменная для инъекции MasterConfig в тестах
_MASTER_CFG: Optional[Any] = None

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


def _get_close_near(df: pd.DataFrame, ts: pd.Timestamp, how: str = "pad") -> float:
    """Честно получить close на/до ts (UTC, минутные бары).
    how: "pad" → берём последний бар ≤ ts; "nearest" оставлен для совместимости."""
    if ts in df.index:
        return float(df.loc[ts, "close"])
    method = how if how in ("pad", "nearest", "backfill") else "pad"
    i = df.index.get_indexer([ts], method=method)[0]
    return float(df.iloc[i]["close"])

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
    policy_loader: Optional[str]
    checkpoint_path: Optional[str]
    strict: bool = False

@dataclass
class Cfg:
    config_name: str
    db_provider_path: str
    index_csv: str
    exec: ExecParams
    pt: PTParams
    inference: InferenceParams
    # --- расширения для потокового построения индекса ---
    build_index_from_db: bool
    time_start_utc: Optional[datetime]
    time_end_utc: Optional[datetime]
    ctx_minutes: int
    session_minutes: int
    # режим построения индекса: "spike" | "sliding"
    index_mode: str
    sliding_stride_minutes: int
    # детектор
    det_context: int
    det_window: int
    det_abs_change_pct: float
    det_contrast_min: float
    det_cooldown: int
    det_use_lookahead: bool
    # опционально: список тикеров для сканирования
    symbols: List[str]

def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
    mod = _load_py_module(cfg_path)
    if not hasattr(mod, "data") or not isinstance(mod.data, dict):
        raise RuntimeError("В конфиге нужен dict `data`.")
    data = mod.data
    # master_cfg НЕ обязателен: попробуем найти cfg / MasterConfig(), иначе оставим None
    master_cfg = getattr(mod, "cfg", None)
    if master_cfg is None and hasattr(mod, "MasterConfig"):
        try:
            master_cfg = mod.MasterConfig()
        except Exception:
            master_cfg = None

    # обязательные части (см. SYSTEM_PROMPT.md / README)
    dbp = data["db_provider"]
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
    index_csv = os.path.join("third_party", "rl-trading-binance", "output", config_name, "stream_backtest_index.csv")
    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
    ex_d = data.get("exec", {})
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
    # ---- inference ----
    inf_d = data.get("inference", {})
    inf = InferenceParams(
        policy_loader=inf_d.get("policy_loader"),
        checkpoint_path=inf_d.get("checkpoint_path"),
        strict=bool(inf_d.get("strict", False)),
    )
    # --- доп. поля для индекса и детектора ---
    tr = data.get("time_range", {})
    t_start = tr.get("start_utc")
    t_end = tr.get("end_utc")
    t_start = t_start if t_start is None else _to_utc(t_start)
    t_end = t_end if t_end is None else _to_utc(t_end)
    ctx_m = int(data.get("ctx_minutes", 30))
    sess_m = int(data.get("session_minutes", 10))
    index_mode = str(data.get("index_mode", "spike"))
    sliding_stride = int(data.get("sliding_stride_minutes", 1))
    det = data.get("detector", {})
    det_ctx = int(det.get("context_minutes", 90))
    det_win = int(det.get("window_minutes", 10))
    det_abs = float(det.get("abs_change_pct", data.get("trigger", {}).get("abs_change_pct", 5.0)))
    det_con = float(det.get("contrast_min", 5.0))
    det_cool = int(det.get("cooldown_minutes", data.get("trigger", {}).get("cooldown_minutes", 60)))
    det_la = bool(det.get("use_lookahead", True))

    # --- NEW: Адаптация под вложенную структуру build_index_from_db ---
    build_idx_cfg = data.get("build_index_from_db", False)
    if isinstance(build_idx_cfg, dict):
        build_idx_enabled = bool(build_idx_cfg.get("enabled", False))
        symbols = list(build_idx_cfg.get("symbols", []))
    else:
        build_idx_enabled = bool(build_idx_cfg)
        symbols = list(data.get("symbols", []))

    paper_trader_cfg = Cfg(config_name, dbp, index_csv, execp, ptp, inf,
               build_idx_enabled,
               t_start, t_end, ctx_m, sess_m,
               index_mode, sliding_stride,
               det_ctx, det_win, det_abs, det_con, det_cool, det_la,
               symbols)
    return paper_trader_cfg, master_cfg

# ------------------------------ Provider ---------------------------

ProviderFn = Callable[[List[str], str, str], Dict[str, pd.DataFrame]]

def _load_db_provider(path: str) -> ProviderFn:
    if ":" not in path:
        raise RuntimeError("`data.db_provider` должен быть 'module:function'.")
    mod_path, fn_name = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, fn_name):
        raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
    return getattr(mod, fn_name)

# ------------------------------ Inference --------------------------
class _Policy(Protocol):
    # Рекомендуемый интерфейс адаптера инференса:
    #  - predict_side(df_ctx: pd.DataFrame) -> str  ("BUY"/"SELL")
    #  - либо predict(df_ctx) -> int (1=BUY, 0/−1=SELL)
    #  - либо __call__(df_ctx) -> ...
    def predict_side(self, df_ctx: pd.DataFrame) -> str: ...

def _load_policy(policy_loader: Optional[str], checkpoint_path: Optional[str], cfg_path: str, stats: Optional[Dict]) -> Optional[_Policy]:
    if not policy_loader:
        return None
    if ":" not in policy_loader:
        raise RuntimeError("`data.inference.policy_loader` должен быть 'module:function'.")
    mod_path, fn_name = policy_loader.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, fn_name):
        raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}` (policy_loader).")
    loader = getattr(mod, fn_name)

    def _resolve_master_cfg():
        try:
            cfg_mod = _load_py_module(cfg_path)
        except Exception:
            return None
        for name in ("cfg", "master_cfg"):
            if hasattr(cfg_mod, name):
                return getattr(cfg_mod, name)
        if hasattr(cfg_mod, "MasterConfig"):
            try:
                return cfg_mod.MasterConfig()
            except Exception:
                pass
        if hasattr(cfg_mod, "data"):
            return cfg_mod.data
        return cfg_mod

    mc = _resolve_master_cfg()
    # Ожидаем, что лоадер принимает 3 аргумента: ckpt, master_cfg, stats
    return loader(checkpoint_path, mc, stats)

def _policy_to_side(policy: _Policy, symbol: str, df_ctx: pd.DataFrame, master_cfg: Any) -> Optional[str]:
    # Универсальный вызов с мягкой деградацией интерфейса
    if hasattr(policy, "predict_side"):
        side = str(policy.predict_side(df_ctx)).upper()
        if side in ("BUY", "SELL"):
            return side
        if side == "HOLD":
            return None
        return None
    if hasattr(policy, "predict"):
        pred = policy.predict(symbol, df_ctx)
        # допускаем как str, так и int
        if isinstance(pred, str):
            up = pred.upper()
            if up in ("BUY", "SELL"):
                return up
            if up == "HOLD":
                return None
            return None
        try:
            pred_i = int(pred)
        except Exception:
            return None
        if pred_i == 1:
            return "BUY"
        if pred_i == 2:
            return "SELL"
        # 0 или иное — трактуем как HOLD/нет сигнала
        return None
    if callable(policy):
        # --- NEW: Добавим отладочный вывод для callable-политик ---
        if master_cfg and hasattr(policy, "model") and hasattr(policy.model, "last_qvals"):
            qvals = getattr(policy.model, "last_qvals", None)
            if qvals is not None:
                adv = qvals - qvals[0]
                act = np.argmax(adv)
                conf = adv[act]
                thr = master_cfg.backtest.long_action_threshold if act == 1 else (master_cfg.backtest.short_action_threshold if act == 2 else 0.0)
                print(f"    [DEBUG] {symbol}: action={act}, confidence={conf:.5f}, threshold={thr:.5f}")

        pred = policy(df_ctx)
        try:
            pred = int(pred)
        except Exception:
            pass
        if pred == 1:
            return "BUY"
        if pred == 2:
            return "SELL"
        return None
    return None

# ------------------------------ Strategy ---------------------------
# (удалено) Follow-Context — заменено на инференс политики

# ------------------------------ Execution helpers -----------------

def _apply_slippage(price: float, bps: float, side: str) -> float:
    # Для тестов паритета: используем slippage из MasterConfig, если он есть
    if _MASTER_CFG is not None and hasattr(_MASTER_CFG, "market"):
        slip_pct = _MASTER_CFG.market.slippage
        return price * (1 + slip_pct) if side == "BUY" else price * (1 - slip_pct)
    # bps = basis points (0.01% = 1 bps). Для покупки повышаем цену, для продажи понижаем.
    delta = price * (bps / 10000.0)
    return price + delta if side == "BUY" else price - delta

def _fees_cost(notional: float, fee_bps: float) -> float:
    # Для тестов паритета: используем transaction_fee из MasterConfig, если он есть
    if _MASTER_CFG is not None and hasattr(_MASTER_CFG, "market"):
        return notional * _MASTER_CFG.market.transaction_fee
    return notional * (fee_bps / 10000.0)

def _position_size(capital: float, risk_pct: float, entry: float) -> float:
    # Для тестов паритета: используем position_fraction из MasterConfig, если он есть
    if _MASTER_CFG is not None and hasattr(_MASTER_CFG, "backtest"):
        position_fraction = _MASTER_CFG.backtest.position_fraction
        return (capital * position_fraction) / max(entry, 1e-12)
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

def _build_sliding_windows(
    df: pd.DataFrame,
    ctx_minutes: int,
    session_minutes: int,
    stride_minutes: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """
    Строит эпизоды по КАЖДОЙ минуте с шагом stride_minutes.
    Возвращает список (ctx_start, ctx_end, ses_start, ses_end, abs_change_pct=nan).
    """
    if df.empty:
        return []
    df = _ensure_utc_index(df).sort_index()
    ts = df.index.unique().sort_values()
    wins: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
    # требуем полное покрытие минут в [ctx_start, ses_end)
    for i in range(ctx_minutes, len(ts) - session_minutes, stride_minutes):
        ses_start = ts[i]
        ctx_start = ses_start - pd.Timedelta(minutes=ctx_minutes)
        ctx_end   = ses_start
        ses_end   = ses_start + pd.Timedelta(minutes=session_minutes)
        full_range = pd.date_range(ctx_start, ses_end - pd.Timedelta(minutes=1), freq="min", tz="UTC")
        slice_df = df.loc[(df.index >= full_range[0]) & (df.index <= full_range[-1])]
        if len(slice_df) == len(full_range):
            wins.append((ctx_start, ctx_end, ses_start, ses_end, float("nan")))
    return wins

# ------------------------------ Main --------------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Использование: python paper_trader.py configs/alpha.py")
        return 2
    cfg_path = argv[1]
    cfg, master_cfg = _load_cfg(cfg_path)
    out_dir = os.path.join("third_party", "rl-trading-binance", "output", cfg.config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_trades = os.path.join(out_dir, "trades.csv")
    out_metrics = os.path.join(out_dir, "metrics.json")
    stats_path = os.path.join(out_dir, "norm_stats.json")

    provider = _load_db_provider(cfg.db_provider_path)

    # --- Статистики нормализации ---
    stats = None
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    if stats is None:
        print(f"ERROR: Normalization stats not found at '{stats_path}'")
        print("Please run backtest_engine.py first to generate the stats file:")
        print(f"  python third_party\\rl-trading-binance\\backtest_engine.py {cfg_path}")
        return 1 # Завершаем с ошибкой

    # Загружаем модель (если указана)
    policy: Optional[_Policy] = _load_policy(
        cfg.inference.policy_loader,
        cfg.inference.checkpoint_path,
        cfg_path,
        stats
    )
    need_build = cfg.build_index_from_db or not os.path.exists(cfg.index_csv)
    if need_build:
        if cfg.time_start_utc is None or cfg.time_end_utc is None or not cfg.symbols:
            raise RuntimeError(
                "Для построения индекса из БД нужны data.time_range{start_utc,end_utc} и data.symbols[]. "
                "Либо выключите build_index_from_db=False и подготовьте stream_backtest_index.csv офлайн."
            )
        rows = []
        print("Starting index generation...")
        total_wins = 0
        for sym in cfg.symbols:
            print(f"--> Processing symbol: {sym}")
            feed = dict(provider([sym], cfg.time_start_utc.isoformat(), cfg.time_end_utc.isoformat()))
            if sym not in feed or feed[sym].empty:
                continue
            df = _ensure_utc_index(feed[sym]).sort_index()
            # --- Построение окон: режим "sliding" (каждый минутный бар) или "spike" (детектор всплесков) ---
            if getattr(cfg, "index_mode", "spike") == "sliding":
                wins = _build_sliding_windows(
                    df=df,
                    ctx_minutes=cfg.ctx_minutes,
                    session_minutes=cfg.session_minutes,
                    stride_minutes=max(1, int(getattr(cfg, "sliding_stride_minutes", 1))),
                )
            else:
                wins = find_spike_windows(
                    df,
                    context_minutes=cfg.det_context,
                    window_minutes=cfg.det_window,
                    abs_change_threshold_pct=cfg.det_abs_change_pct,
                    contrast_min=cfg.det_contrast_min,
                    cooldown_minutes=cfg.det_cooldown,
                    use_lookahead=cfg.det_use_lookahead,
                )
            for (ctx_start, ctx_end, ses_start, ses_end, abs_chg) in wins:
                # Приводим торговую сессию к длине из конфига (напр., 10 минут в демо)
                ses_end_adj = ses_start + timedelta(minutes=cfg.session_minutes)
                rows.append({
                    "symbol": sym,
                    "ctx_start": ctx_start.isoformat(),
                    "ctx_end": ctx_end.isoformat(),
                    "session_start": ses_start.isoformat(),
                    "session_end": ses_end_adj.isoformat(),
                    "abs_change_pct": abs_chg,
                })
            print(f"    -> windows found: {len(wins)}")
            total_wins += len(wins)
        idx = pd.DataFrame(rows)
        out_dir = os.path.dirname(cfg.index_csv)
        os.makedirs(out_dir, exist_ok=True)
        # Сразу храним UTC-датавремена и используем их же ниже
        for col in ["ctx_start","ctx_end","session_start","session_end"]:
            idx[col] = pd.to_datetime(idx[col], utc=True)
        idx.to_csv(cfg.index_csv, index=False)
        print(f"Index saved: {cfg.index_csv}  | total windows: {total_wins}")
    else:
        idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])

    # === NEW: Prefetch & cache per-symbol minute feed for the whole time-span ===
    # Сокращаем кол-во DB-коннектов: вместо запроса на КАЖДОЕ окно берём всё разом по символу
    # и далее работаем только срезами по временам окна.
    symbols = sorted(idx["symbol"].unique())
    # по каждому символу берём минимальный ctx_start и максимальный session_end для охвата всех окон
    sym_ranges = {
        sym: (
            pd.to_datetime(idx.loc[idx["symbol"] == sym, "ctx_start"].min(), utc=True),
            pd.to_datetime(idx.loc[idx["symbol"] == sym, "session_end"].max(), utc=True),
        )
        for sym in symbols
    }
    FEED_CACHE: Dict[str, pd.DataFrame] = {}
    for sym, (t_min, t_max) in sym_ranges.items():
        try:
            feed = dict(provider([sym], t_min.isoformat(), t_max.isoformat()))
            if sym in feed and not feed[sym].empty:
                FEED_CACHE[sym] = _ensure_utc_index(feed[sym]).sort_index()
            else:
                FEED_CACHE[sym] = pd.DataFrame()
        except Exception as e:
            print(f"[warn] provider failed for {sym}: {e}")
            FEED_CACHE[sym] = pd.DataFrame()
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
        last_ts = ses_end - pd.Timedelta(minutes=1)
        # Берём минутные бары из кэша (загружены один раз на символ)
        df = FEED_CACHE.get(sym, pd.DataFrame())
        if df.empty:
            continue
        if df.index[0] > ctx_start or df.index[-1] < last_ts:
            # неполное покрытие — пропустим окно
            continue
        
        # Направление определяет модель по контексту
        if policy is None:
            if cfg.inference.strict:
                continue  # строгий режим: без политики окно пропускаем
            else:
                raise RuntimeError("Policy не загружена, а inf_strict=False запрещает эвристику.")
        
        # Контекст для модели должен иметь длину agent_history_len
        ctx_end_ts = pd.Timestamp(ctx_end)
        # NB: ctx_start из индекса может быть шире, чем нужно модели.
        # Отрезаем окно нужной длины agent_history_len от конца контекста.
        ctx_start_for_model = ctx_end_ts - pd.Timedelta(minutes=master_cfg.seq.agent_history_len)
        df_ctx = df.loc[ctx_start_for_model : ctx_end_ts - pd.Timedelta(minutes=1)]
        
        # --- NEW: Передаем master_cfg в _policy_to_side для отладки ---
        # Это позволит нам видеть пороги прямо в логе.
        side = _policy_to_side(policy, sym, df_ctx, master_cfg)

        if side not in ("BUY", "SELL"):
            if cfg.inference.strict:
                # В строгом режиме просто пропускаем, если модель решила не торговать (HOLD)
                if side is None:
                    pass # Отладочный вывод теперь делается в _policy_to_side
                continue
            else:
                raise RuntimeError(f"predict_side вернул некорректное значение: {side}")

        # Первая цена сессии / последняя цена сессии (с pad-защитой)
        first_px = _get_close_near(df, pd.Timestamp(ses_start), how="pad")
        # Исполнение
        entry_raw = first_px
        entry_px = _apply_slippage(entry_raw, cfg.exec.slippage_bps, side)
        qty = _position_size(capital, cfg.exec.risk_per_trade_pct, entry_px)
        # Выход в конце сессии
        last_px = _get_close_near(df, pd.Timestamp(ses_end), how="pad")
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