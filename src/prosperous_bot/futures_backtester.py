"""futures_backtester.py – Hedge‑mode back‑test
=================================================
* Leverage 10×, TP +1 %, SL ‑4 %, taker fee 0.05 % (taker side assumed)
* Читает/создаёт CSV `<SYMBOL>_data.csv` в `C:\Python\Prosperous_Bot\graphs`
* Выводит: `results.csv` (все сделки) и `dashboard.html` (equity‑кривые + KPI)
* KPI теперь расширены: Sharpe, Max DD %(equity), win‑rate %, total trades,
  TP/SL‑count, max time‑in‑trade (час)
* Запуск без аргументов — символы берутся из `config_signal.json` → `binance_pairs`
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as pplot

from .config import Config
from .data_loader import DataLoader
from .signal_generator import process_all_data

# ---------------- настройки стратегии -----------------------
LEVERAGE = 10
TAKER_FEE = 0.0005
TP_PCT = 0.015   # +1.5 %
SL_PCT   = -0.04  # −4 %
INITIAL_BALANCE = 100.0  # USDT
CSV_SUFFIX = "_data.csv"
GRAPH_DIR = Path(r"C:\Python\Prosperous_Bot\graphs"); GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- модели --------------------------
class Position:
    """Описывает одну сторону хедж‑позиции (long или short)."""

    def __init__(self, side: str, entry: float, qty: float, ts: pd.Timestamp):
        self.side = side
        self.entry = entry
        self.qty = qty
        self.open_ts = ts
        self.tp = entry * (1 + TP_PCT if side == 'long' else 1 + SL_PCT)
        self.sl = entry * (1 + SL_PCT if side == 'long' else 1 + TP_PCT)
        self.is_open = True
        self.close_px: float | None = None
        self.close_ts: pd.Timestamp | None = None
        self.result: str | None = None  # TP / SL / Signal / End

    def check_exit(self, hi: float, lo: float):
        if not self.is_open:
            return None
        if self.side == 'long':
            if lo <= self.sl:
                return self.sl, 'SL'
            if hi >= self.tp:
                return self.tp, 'TP'
        else:
            if hi >= self.sl:
                return self.sl, 'SL'
            if lo <= self.tp:
                return self.tp, 'TP'
        return None

    def close(self, px: float, ts: pd.Timestamp, tag: str):
        self.close_px, self.close_ts, self.result, self.is_open = px, ts, tag, False

    def pnl(self):
        sign = 1 if self.side == 'long' else -1
        return sign * (self.close_px - self.entry) * self.qty

    def fee(self):
        return 2 * self.entry * self.qty * TAKER_FEE

    def duration(self):
        return (self.close_ts - self.open_ts).total_seconds() if self.close_ts else 0


class Backtester:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.cfg = Config('config_signal.json')
        self.loader = DataLoader(self.cfg)
        self.trades: List[Dict] = []  # raw trade dicts
        self.results: Dict[str, Dict] = {}  # equity curves

    # ------------- загрузка/кэш данных ----------------------
    async def _df(self, symbol: str):
        csv = GRAPH_DIR / f"{symbol}{CSV_SUFFIX}"
        if csv.exists():
            try:
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                return df[~df.index.duplicated(keep='first')]
            except Exception as e:
                print(f"CSV read error {symbol}: {e}")
        df = await self.loader.get_realtime_data(symbol, self.cfg.get('interval'), self.cfg.get('lookback_period'))
        if df is not None and not df.empty:
            df.to_csv(csv)
        return df

    # ------------- обработка символа ------------------------
    async def _run_symbol(self, symbol: str):
        df = await self._df(symbol)
        if df is None or df.empty:
            print(f"No data for {symbol}"); return
        sig_df, _, signals = process_all_data({symbol: df}, self.cfg.config, {symbol: {}})[symbol]

        long: Optional[Position] = None
        short: Optional[Position] = None
        equity = INITIAL_BALANCE
        curve = []
        sig_map = {t: s for s, t in signals}

        for ts, row in sig_df.iterrows():
            px, hi, lo = row['close'], row['high'], row['low']
            # --- TP/SL
            for pos in (long, short):
                if pos and pos.is_open:
                    res = pos.check_exit(hi, lo)
                    if res:
                        ex_px, tag = res
                        pos.close(ex_px, ts, tag)
                        equity += pos.pnl() - pos.fee()
                        self.trades.append(self._rec(symbol, pos))
                        # --- сигнал
            sig = sig_map.get(ts)
            if sig == 'buy':
                # открываем long, не трогая возможный short
                if not (long and long.is_open):
                    long = Position('long', px, (equity*LEVERAGE)/px, ts)
            elif sig == 'sell':
                # открываем short, не трогая возможный long
                if not (short and short.is_open):
                    short = Position('short', px, (equity*LEVERAGE)/px, ts)
            curve.append(dict(ts=ts, equity=equity))

        # --- закрываем остатки ---
        last_px, last_ts = sig_df.iloc[-1]['close'], sig_df.index[-1]
        for pos in (long, short):
            if pos and pos.is_open:
                pos.close(last_px, last_ts, 'End'); equity += pos.pnl()-pos.fee(); self.trades.append(self._rec(symbol, pos))
        self.results[symbol] = dict(final=equity, curve=pd.DataFrame(curve))

    # --------------------------------------------------------
    def _rec(self, sym, pos):
        return dict(symbol=sym, side=pos.side, open_ts=pos.open_ts, open_px=pos.entry,
                    close_ts=pos.close_ts, close_px=pos.close_px, result=pos.result,
                    pnl=pos.pnl(), fee=pos.fee(), duration=pos.duration())

    # ---------------- KPI per‑symbol ------------------------
    @staticmethod
    def _max_dd(eq: pd.Series):
        roll = eq.cummax(); return ((eq-roll)/roll).min()*100

    @staticmethod
    def _sharpe(ret: pd.Series):
        return 0 if ret.std()==0 else (ret.mean()/ret.std())*np.sqrt(len(ret))

    def _kpi(self, trades: pd.DataFrame):
        res = {}
        for sym, g in trades.groupby('symbol'):
            wins=(g['pnl']>0).mean()*100
            tp=(g['result']=='TP').sum(); sl=(g['result']=='SL').sum()
            longest=g['duration'].max()/3600
            curve=self.results[sym]['curve']
            md=self._max_dd(curve['equity'])
            sr=self._sharpe(curve['equity'].pct_change().fillna(0))
            res[sym]=dict(trades=len(g),win_rate=round(wins,2),tp=tp,sl=sl,
                           max_dd=round(md,2),sharpe=round(sr,2),longest_trade=round(longest,2))
        return res

    # ---------------- сохранение ----------------------------
    def _save(self):
        trades=pd.DataFrame(self.trades); trades.to_csv(GRAPH_DIR/'results.csv',index=False)
        stats=pd.DataFrame(self._kpi(trades)).T
        curves=[go.Scatter(x=r['curve']['ts'], y=r['curve']['equity'], mode='lines', name=sym) for sym,r in self.results.items()]
        html_curve=pplot(go.Figure(curves,layout=go.Layout(title='Equity Curves', xaxis_title='Time', yaxis_title='USDT')), include_plotlyjs='cdn', output_type='div')
        (GRAPH_DIR/'dashboard.html').write_text(f"<html><body><h1>Backtest Dashboard</h1>{html_curve}<h2>Performance</h2>{stats.to_html(border=0)}</body></html>", encoding='utf-8')
        print(f"Saved results to {GRAPH_DIR}\n")

    async def _runner(self):
        """Async wrapper to run back‑test and then save outputs."""
        await asyncio.gather(*(self._run_symbol(s) for s in self.symbols))
        self._save()

    # ---------------- public API ----------------
    def run(self):
        """Blocking call: start event‑loop, back‑test all symbols, write dashboard."""
        asyncio.run(self._runner())


# ------------------------ CLI -------------------------
if __name__ == "__main__":
    import sys

    cfg = Config("config_signal.json")
    if len(sys.argv) == 1:
        syms = [s.upper() for s in cfg.get("binance_pairs", [])]
        if not syms:
            print("config_signal.json missing 'binance_pairs' → nothing to test"); sys.exit(0)
        print("No symbols specified → using binance_pairs from config_signal.json")
    else:
        syms = [s.upper() for s in sys.argv[1:]]

    Backtester(syms).run()
