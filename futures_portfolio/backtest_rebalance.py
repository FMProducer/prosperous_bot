import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
import aiohttp
import time
import json
import os
import math
import logging
import argparse
import traceback
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class LimitOrderSimulator:
    """Симулятор лимитных ордеров для учета рыночных фрикций."""
    def __init__(self, commission_pct: float = 0.0004, offset_pct: float = 0.1):
        self.commission_pct = Decimal(str(commission_pct))
        self.offset_pct = Decimal(str(offset_pct))
        self.stats: Dict[str, Any] = {"attempted": 0, "filled": 0, "fallback": 0}

    def simulate_limit_execution(self, side: str, qty: Decimal, mid_price: Decimal,
                                  candle_high: Decimal, candle_low: Decimal) -> Tuple[bool, Decimal, Decimal, str]:
        self.stats["attempted"] += 1
        if side == "SELL":
            limit_price = mid_price * (1 + self.offset_pct / 100)
            if candle_high >= limit_price:
                self.stats["filled"] += 1
                return True, qty, max(limit_price, mid_price), "LIMIT_FILLED"
            self.stats["fallback"] += 1
            return True, qty, candle_low, "FALLBACK_MARKET"
        else: # BUY
            limit_price = mid_price * (1 - self.offset_pct / 100)
            if candle_low <= limit_price:
                self.stats["filled"] += 1
                return True, qty, min(limit_price, mid_price), "LIMIT_FILLED"
            self.stats["fallback"] += 1
            return True, qty, candle_high, "FALLBACK_MARKET"

    def get_summary(self) -> Dict[str, Any]:
        return self.stats

class PortfolioState:
    """Управление состоянием портфеля во время бэктеста."""
    def __init__(self, initial_capital: float, commission: float, targets: Dict[str, Any], threshold: float,
                 siphoning_threshold_pct: float = 0.0, reinvestment_ratio: float = 0.0, min_notional: float = 6.0):
        self.initial_capital = Decimal(str(initial_capital))
        self.commission = Decimal(str(commission))
        self.targets = targets
        self.threshold = Decimal(str(threshold))
        self.siphoning_threshold_pct = Decimal(str(siphoning_threshold_pct))
        self.reinvestment_ratio = Decimal(str(reinvestment_ratio))
        self.min_notional = Decimal(str(min_notional))

        self.real_equity = self.initial_capital
        self.siphoning_reserve = Decimal('0.0')

        self.l_target = Decimal(str(targets["BASE_LONG"]["share"]))
        self.s_target = Decimal(str(targets["BASE_SHORT"]["share"]))
        self.v_target = Decimal(str(targets["VIRTUAL"]["share"]))
        self.l_lev = Decimal(str(targets["BASE_LONG"]["leverage"]))
        self.s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"]))

        self.positions: Dict[str, Decimal] = {"LONG": Decimal('0.0'), "SHORT": Decimal('0.0')}
        self.entry_prices: Dict[str, Decimal] = {"LONG": Decimal('0.0'), "SHORT": Decimal('0.0')}
        
        self.virt_qty = Decimal('0.0')

        self.history: List[float] = []
        self.rebalance_log: List[Dict[str, Any]] = []
        self.cycles: int = 0
        self.skipped_expansions_counter: int = 0
        self.liquidations_counter: int = 0

    def init_state(self, price: float) -> None:
        """Инициализация начальных позиций."""
        dec_price = Decimal(str(price))
        # Initial allocation based on target shares of initial capital
        self.positions["LONG"] = (self.initial_capital * self.l_target * self.l_lev) / dec_price
        self.positions["SHORT"] = (self.initial_capital * self.s_target * self.s_lev) / dec_price
        self.entry_prices["LONG"] = dec_price
        self.entry_prices["SHORT"] = dec_price

        # Virtual leg as physical coins
        self.virt_qty = (self.initial_capital * self.v_target) / dec_price

        # Initial commission and setup
        # Wallet Balance = Initial Capital - Cost of Virtual (since it's spot-like) - entry commissions
        virt_cost = self.virt_qty * dec_price
        init_comm = (abs(self.positions["LONG"] * dec_price) + abs(self.positions["SHORT"] * dec_price)) * self.commission

        self.real_equity -= init_comm
        self.real_equity -= virt_cost

    def update(self, price: float, high: float, low: float, sim: Optional[LimitOrderSimulator] = None) -> float:
        dec_price = Decimal(str(price))
        dec_high = Decimal(str(high))
        dec_low = Decimal(str(low))

        # 1. MTM PnL (Futures)
        mtm_pnl_l = self.positions["LONG"] * (dec_price - self.entry_prices["LONG"])
        mtm_pnl_s = self.positions["SHORT"] * (self.entry_prices["SHORT"] - dec_price)

        # 2. Liquidation Check & NAV Calculation
        # For MTM Pure Isolation: NAV = (PositionValue / Leverage) + UnrealizedPnL
        val_l = ((abs(self.positions["LONG"]) * self.entry_prices["LONG"] / self.l_lev) + mtm_pnl_l) if self.positions["LONG"] != 0 else Decimal('0')
        val_s = ((abs(self.positions["SHORT"]) * self.entry_prices["SHORT"] / self.s_lev) + mtm_pnl_s) if self.positions["SHORT"] != 0 else Decimal('0')
        
        # [EDGE CASE] Leg-Specific Liquidation
        if val_l <= 0 and self.positions["LONG"] != 0:
            loss = (abs(self.positions["LONG"]) * self.entry_prices["LONG"] / self.l_lev)
            self.real_equity -= loss
            self.positions["LONG"] = Decimal('0')
            self.liquidations_counter += 1
            mtm_pnl_l = Decimal('0')
            val_l = Decimal('0')
            logger.warning(f"LONG leg liquidated at price {price}! Loss: {loss:.2f} USDT")

        if val_s <= 0 and self.positions["SHORT"] != 0:
            loss = (abs(self.positions["SHORT"]) * self.entry_prices["SHORT"] / self.s_lev)
            self.real_equity -= loss
            self.positions["SHORT"] = Decimal('0')
            self.liquidations_counter += 1
            mtm_pnl_s = Decimal('0')
            val_s = Decimal('0')
            logger.warning(f"SHORT leg liquidated at price {price}! Loss: {loss:.2f} USDT")

        val_v = self.virt_qty * dec_price

        # 3. TPV = Wallet Balance + Futures PnL + Virtual Value
        tpv = self.real_equity + mtm_pnl_l + mtm_pnl_s + val_v
        
        # total_tpv includes reserve for tracking/siphoning
        total_tpv_with_reserve = tpv + self.siphoning_reserve

        if total_tpv_with_reserve <= 0:
            self.history.append(0.0)
            return 0.0

        # 4. Current Shares
        share_l = val_l / tpv if tpv > 0 else Decimal('0')
        share_s = val_s / tpv if tpv > 0 else Decimal('0')
        share_v = val_v / tpv if tpv > 0 else Decimal('0')
        share_c = (tpv - (val_l + val_s + val_v)) / tpv if tpv > 0 else Decimal('0')

        # 5. Rebalance Check
        dev_l = share_l - self.l_target
        dev_s = share_s - self.s_target
        dev_v = share_v - self.v_target

        if any(abs(d) > self.threshold for d in [dev_l, dev_s, dev_v]):
            self.cycles += 1
            log_entry = {
                "step": len(self.history) + 1,
                "price": float(dec_price),
                "tpv": float(tpv),
                "shares": {"L": float(share_l), "S": float(share_s), "V": float(share_v), "C": float(share_c)},
                "actions": []
            }

            actions = [
                {"key": "LONG", "dev": dev_l, "lev": self.l_lev, "val": val_l},
                {"key": "SHORT", "dev": dev_s, "lev": self.s_lev, "val": val_s},
                {"key": "VIRTUAL", "dev": dev_v, "lev": Decimal('1.0'), "val": val_v}
            ]
            # Order of execution: Surplus FIRST to liberate cash
            actions.sort(key=lambda x: x['dev'], reverse=True)

            for act in actions:
                if act['key'] == "LONG":
                    target_notional = self.l_target * tpv * self.l_lev
                    current_notional = abs(self.positions["LONG"]) * dec_price
                elif act['key'] == "SHORT":
                    target_notional = self.s_target * tpv * self.s_lev
                    current_notional = abs(self.positions["SHORT"]) * dec_price
                else: # VIRTUAL
                    target_notional = self.v_target * tpv * Decimal('1.0')
                    current_notional = self.virt_qty * dec_price

                diff_usdt = target_notional - current_notional

                # [EDGE CASE] Death Spiral / Min Notional check
                if abs(diff_usdt) < self.min_notional and act['key'] != "VIRTUAL":
                    if abs(target_notional/act['lev'] - act['val']) / (tpv + Decimal('1e-9')) > Decimal('3') * self.threshold:
                        logger.warning(f"BLOCK: Rebalance for {act['key']} blocked by min_notional! Deviation exceeds 3x threshold")
                    continue

                # Trade side mapping
                if act['key'] == "SHORT":
                    side = "SELL" if diff_usdt > 0 else "BUY"
                else:
                    side = "BUY" if diff_usdt > 0 else "SELL"

                f_price = dec_price
                if sim and act['key'] != "VIRTUAL":
                    _, _, f_price, _ = sim.simulate_limit_execution(side, abs(diff_usdt)/dec_price, dec_price, dec_high, dec_low)

                # Expansion constraint: only if we have enough real_equity (Wallet Balance)
                if diff_usdt > 0:
                    needed_margin = diff_usdt / act['lev']
                    # Budget for commission (all legs including Virtual have commission now)
                    est_comm = abs(diff_usdt) * self.commission
                    if needed_margin + est_comm > self.real_equity:
                        self.skipped_expansions_counter += 1
                        # Scale down
                        max_allowed_margin = max(Decimal('0'), self.real_equity - est_comm)
                        if max_allowed_margin * act['lev'] < (self.min_notional if act['key'] != "VIRTUAL" else Decimal('0.1')):
                            continue # Cannot even afford min order
                        diff_usdt = max_allowed_margin * act['lev']

                if act['key'] == "LONG":
                    pnl = self.positions["LONG"] * (f_price - self.entry_prices["LONG"])
                    comm = abs(diff_usdt) * self.commission
                    self.real_equity += pnl - comm
                    self.positions["LONG"] = (self.positions["LONG"] * f_price + diff_usdt) / f_price
                    self.entry_prices["LONG"] = f_price
                elif act['key'] == "SHORT":
                    pnl = self.positions["SHORT"] * (self.entry_prices["SHORT"] - f_price)
                    comm = abs(diff_usdt) * self.commission
                    self.real_equity += pnl - comm
                    self.positions["SHORT"] = (self.positions["SHORT"] * f_price + diff_usdt) / f_price
                    self.entry_prices["SHORT"] = f_price
                elif act['key'] == "VIRTUAL":
                    comm = abs(diff_usdt) * self.commission # Virtual now pays commission
                    self.real_equity -= (diff_usdt + comm)
                    self.virt_qty = (self.virt_qty * dec_price + diff_usdt) / dec_price

                log_entry["actions"].append(f"{act['key']} {side} {float(abs(diff_usdt)):.2f}")

            self.rebalance_log.append(log_entry)

            # Re-calculate TPV after rebalance for history
            mtm_pnl_l = self.positions["LONG"] * (dec_price - self.entry_prices["LONG"])
            mtm_pnl_s = self.positions["SHORT"] * (self.entry_prices["SHORT"] - dec_price)
            tpv = self.real_equity + mtm_pnl_l + mtm_pnl_s + (self.virt_qty * dec_price)
            total_tpv_with_reserve = tpv + self.siphoning_reserve

        # 6. SAFE Siphoning logic (Runs every cycle)
        total_surplus = total_tpv_with_reserve - self.initial_capital
        siphoning_threshold_abs = self.initial_capital * (self.siphoning_threshold_pct / 100)

        if total_surplus > self.siphoning_reserve + max(Decimal('0.1'), siphoning_threshold_abs):
            new_profit = total_surplus - self.siphoning_reserve
            siphon_amount = new_profit * (1 - self.reinvestment_ratio)
            if siphon_amount > Decimal('0.1'):
                self.siphoning_reserve += siphon_amount
                self.real_equity -= siphon_amount
                total_tpv_with_reserve -= siphon_amount

        self.history.append(float(total_tpv_with_reserve))
        return float(total_tpv_with_reserve)

async def download_live_data(symbol: str, data_dir: str, days: float = 2.0) -> str:
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    logger.info(f"Downloading live data for {symbol}...")
    params = {"symbol": symbol, "interval": "1m", "limit": 1500}
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            if resp.status == 200:
                all_data = await resp.json()
            else:
                raise Exception(f"Binance API error: {resp.status}")

    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in df.columns:
        df[col] = df[col].astype(float)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, f"{symbol}_live.feather")
    df.to_feather(file_path)
    return file_path

async def run_backtest(config_path: str, data_dir: str, live_mode: bool = False, ticker_override: Optional[str] = None,
                        days: float = 2.0, commission: float = 0.0004, use_limit_orders: bool = False,
                        limit_offset_pct: float = 0.1, limit_timeout_sec: int = 30, quiet: bool = False,
                        threshold_override: Optional[float] = None, capital_override: Optional[float] = None) -> Optional[Dict[str, Any]]:
    try:
        if quiet:
            logger.setLevel(logging.WARNING)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        base_ticker: str = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

        if live_mode:
            file_path = await download_live_data(base_ticker, data_dir, days)
        else:
            possible_files: List[str] = [
                os.path.join(data_dir, f"{base_ticker}_live.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_10d.feather"),
                os.path.join(data_dir, f"{base_ticker}_live_48h.feather")
            ]
            file_path = next((p for p in possible_files if os.path.exists(p)), None)

        if not file_path or not os.path.exists(file_path):
            logger.error(f"Data file not found for {base_ticker}")
            return None

        df: pd.DataFrame = pd.read_feather(file_path)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        portfolio_cfg: Dict[str, Any] = config["portfolios"][0]
        initial_capital: float = capital_override if capital_override is not None else portfolio_cfg.get("initial_capital", 1000.0)
        targets: Dict[str, Any] = portfolio_cfg["targets"]
        ticker_thresholds: Dict[str, float] = portfolio_cfg.get("ticker_thresholds", {})
        
        if threshold_override is not None:
            threshold = threshold_override
        else:
            threshold = ticker_thresholds.get(base_ticker, portfolio_cfg.get("rebalance_threshold", 0.02))
            
        siphoning_threshold_pct: float = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio: float = portfolio_cfg.get("reinvestment_ratio", 0.0)
        min_notional: float = config.get("min_notional_usdt", 6.0)
        
        close_prices: npt.NDArray[np.float64] = df['close'].values.astype(np.float64)
        high_prices: npt.NDArray[np.float64] = df['high'].values.astype(np.float64)
        low_prices: npt.NDArray[np.float64] = df['low'].values.astype(np.float64)

        limit_enabled: bool = config.get("limit_order_enabled", use_limit_orders)
        sim: Optional[LimitOrderSimulator] = LimitOrderSimulator(commission_pct=commission, offset_pct=limit_offset_pct) if limit_enabled else None

        state = PortfolioState(initial_capital, commission, targets, threshold, siphoning_threshold_pct, reinvestment_ratio, min_notional)
        state.init_state(close_prices[0])

        for i in range(len(df)):
            state.update(close_prices[i], high_prices[i], low_prices[i], sim)

        equity_curve: npt.NDArray[np.float64] = np.array(state.history)
        profit_pct: float = (equity_curve[-1] / (initial_capital + 1e-9) - 1) * 100
        max_eq: npt.NDArray[np.float64] = np.maximum.accumulate(equity_curve)
        dd: npt.NDArray[np.float64] = (max_eq - equity_curve) / (max_eq + 1e-9)
        max_dd_pct: float = np.max(dd) * 100 if len(dd) > 0 else 0.0
        asset_chg_pct: float = (close_prices[-1] / (close_prices[0] + 1e-9) - 1) * 100

        if not quiet:
            logger.info(f"Iterative Backtest for {base_ticker}:")
            logger.info(f"Profit: {profit_pct:+.2f}% | MaxDD: {max_dd_pct:.2f}% | Cycles: {state.cycles}")
            logger.info(f"Asset Change: {asset_chg_pct:+.2f}% | SAFE Reserve: {state.siphoning_reserve:.2f} USDT")
            logger.info(f"Skipped Expansions: {state.skipped_expansions_counter} | Liquidations: {state.liquidations_counter}")
            
            if state.rebalance_log:
                logger.info("\nRebalance Log:")
                for log in state.rebalance_log:
                    shares_str = f"L:{log['shares']['L']*100:.1f}% S:{log['shares']['S']*100:.1f}% V:{log['shares']['V']*100:.1f}% C:{log['shares']['C']*100:.1f}%"
                    actions_str = " | ".join(log['actions'])
                    logger.info(f"Cycle #{log['step']}: Price {log['price']:.6g} | Shares: {shares_str} | TPV: {log['tpv']:.2f} | Actions: {actions_str}")

            if sim:
                s = sim.get_summary()
                logger.info(f"Limit Orders: {s['filled']}/{s['attempted']} filled ({s['fallback']} fallbacks)")

        return {
            "profit_pct": float(profit_pct),
            "max_dd_pct": float(max_dd_pct),
            "cycles": int(state.cycles),
            "asset_chg_pct": float(asset_chg_pct),
            "siphoning_reserve": float(state.siphoning_reserve),
            "skipped_expansions": int(state.skipped_expansions_counter),
            "liquidations": int(state.liquidations_counter)
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    asyncio.run(run_backtest(args.config, data_dir, args.live, args.ticker, args.days, capital_override=args.capital, threshold_override=args.threshold))
