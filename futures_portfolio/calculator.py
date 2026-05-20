import logging
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

logger = logging.getLogger(__name__)

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_qty: float, 
                 long_entry_price: float = 0.0, short_entry_price: float = 0.0,
                 virt_entry_price: float = 0.0, virt_debt: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0,
                 last_rebalance_price: float = 0.0,
                 min_notional: float = 6.0) -> None:
        
        # Convert all inputs to Decimal for precision
        self.positions = {k: Decimal(str(v)) for k, v in positions.items()}
        self.price = Decimal(str(spot_price))
        self.base_ticker = base_ticker
        self.siphoning_reserve = Decimal(str(siphoning_reserve))
        self.initial_capital = Decimal(str(initial_capital))
        self.real_equity = Decimal(str(real_equity)) # Wallet Balance (Cash + Margin, NO PnL)
        self.virt_qty = Decimal(str(virt_qty))
        self.virt_debt = Decimal(str(virt_debt))
        self.targets = targets or {}
        self.last_rebalance_price = Decimal(str(last_rebalance_price)) if last_rebalance_price > 0 else self.price
        self.min_notional = Decimal(str(min_notional))

        # 1. PnL contributions for Heartbeat (RELATIVE to last rebalance)
        l_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        s_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))

        self.pnl_l = (l_qty * (self.price - self.last_rebalance_price)) if l_qty > 0 else Decimal('0')
        self.pnl_s = (s_qty * (self.last_rebalance_price - self.price)) if s_qty > 0 else Decimal('0')
        self.pnl_v = (self.virt_qty * (self.price - self.last_rebalance_price)) if self.virt_qty > 0 else Decimal('0')

        # Real MTM PnL (for TPV) relative to entry prices for futures
        mtm_pnl_l = (l_qty * (self.price - Decimal(str(long_entry_price)))) if l_qty > 0 else Decimal('0')
        mtm_pnl_s = (s_qty * (Decimal(str(short_entry_price)) - self.price)) if s_qty > 0 else Decimal('0')
        
        self.virt_value = self.virt_qty * self.price
        if virt_entry_price > 0:
            self.pnl_v_mtm = (self.price - Decimal(str(virt_entry_price))) * self.virt_qty
        elif self.virt_debt > 0:
            self.pnl_v_mtm = self.virt_value - self.virt_debt
        else:
            v_target_share = Decimal(str(self.targets.get("VIRTUAL", {}).get("share", 0.35)))
            self.pnl_v_mtm = self.virt_value - (self.initial_capital * v_target_share)

        # [SSOT] TPV calculation
        self.tpv = self.real_equity + mtm_pnl_l + mtm_pnl_s + self.virt_value
        if self.tpv <= 0: self.tpv = Decimal('1e-9')
        self.total_tpv = self.tpv + self.siphoning_reserve

        self.total_pnl = self.tpv - self.initial_capital
        self.total_pnl_pct = ((self.tpv / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else Decimal('0')

        # 4. Calculate NOTIONAL Values for core logic
        self.val_long_notional = (l_qty * self.price)
        self.val_short_notional = (s_qty * self.price)
        self.val_virt_notional = self.virt_value
        
        # Internal raw shares for rebalancing
        self.share_long_raw = self.val_long_notional / self.tpv
        self.share_short_raw = self.val_short_notional / self.tpv
        self.share_virt_raw = self.val_virt_notional / self.tpv

        # 5. DISPLAY SHARES (EQUITY-BASED) for Heartbeat readability
        l_lev = Decimal(str(self.targets.get("BASE_LONG", {}).get("leverage", 5)))
        s_lev = Decimal(str(self.targets.get("BASE_SHORT", {}).get("leverage", 5)))
        
        self.l_margin_equity = (l_qty * Decimal(str(long_entry_price)) / l_lev) + mtm_pnl_l if l_qty > 0 else Decimal('0')
        self.s_margin_equity = (s_qty * Decimal(str(short_entry_price)) / s_lev) + mtm_pnl_s if s_qty > 0 else Decimal('0')
        
        self.display_share_long = (self.l_margin_equity / self.tpv * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.display_share_short = (self.s_margin_equity / self.tpv * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.display_share_virt = (self.val_virt_notional / self.tpv * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.display_share_cash = (Decimal('100.00') - self.display_share_long - self.display_share_short - self.display_share_virt)

    def calculate_rebalance(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> Dict:
        """Calculate rebalance actions and return summary for display."""
        actions = self.calculate_deviations(targets, threshold, ignore_limits)

        return {
            "actions": actions,
            "share_long_pct": float(self.display_share_long),
            "share_short_pct": float(self.display_share_short),
            "share_virt_pct": float(self.display_share_virt),
            "share_cash_pct": float(self.display_share_cash),
            "val_long": float(self.l_margin_equity),
            "val_short": float(self.s_margin_equity),
            "val_virt": float(self.val_virt_notional),
            "val_cash": float(self.tpv - (self.l_margin_equity + self.s_margin_equity + self.val_virt_notional)),
            "tpv": float(self.tpv),
            "total_tpv": float(self.total_tpv),
            "pnl_l": float(self.pnl_l),
            "pnl_s": float(self.pnl_s),
            "pnl_v": float(self.pnl_v),
            "total_pnl": float(self.total_pnl),
            "total_pnl_pct": float(self.total_pnl_pct)
        }

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """Core logic: Rebalance based on NOTIONAL exposure."""
        dec_threshold = Decimal(str(threshold))
        min_notional = self.min_notional
        
        surplus_actions = []
        deficit_actions = []
        
        shares = {
            "BASE_LONG": self.share_long_raw,
            "BASE_SHORT": self.share_short_raw,
            "VIRTUAL": self.share_virt_raw
        }

        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            lev = Decimal(str(targets[key].get("leverage", 1)))
            target_notional_share = Decimal(str(targets[key]["share"])) * lev
            current_notional_share = shares[key]
            
            diff_share = current_notional_share - target_notional_share
            
            if not ignore_limits and abs(diff_share) < dec_threshold:
                continue

            diff_usdt = (target_notional_share - current_notional_share) * self.tpv
            diff_equity = diff_usdt / lev

            if abs(diff_usdt) < Decimal('1.0'): continue

            action = {
                "key": key,
                "type": "VIRTUAL_ORDER" if key == "VIRTUAL" else "ORDER",
                "symbol": "VIRTUAL" if key == "VIRTUAL" else f"{self.base_ticker}_{key.replace('BASE_', '')}",
                "base_symbol": "VIRTUAL" if key == "VIRTUAL" else self.base_ticker,
                "position_side": "BOTH" if key == "VIRTUAL" else key.replace('BASE_', ''),
                "diff_usdt": float(diff_usdt),
                "diff_equity": float(diff_equity),
                "leverage": float(lev),
                "is_reduction": diff_usdt < 0
            }

            if action["is_reduction"]:
                surplus_actions.append(action)
            else:
                deficit_actions.append(action)

        final_actions = []
        total_proceeds = Decimal('0')
        for act in surplus_actions:
            if abs(Decimal(str(act["diff_usdt"]))) >= min_notional or ignore_limits:
                act["priority"] = 0
                final_actions.append(act)
                total_proceeds += abs(Decimal(str(act["diff_equity"])))

        # Available cash = Current Cash (MTM based) + Proceeds from sells
        l_lev = Decimal(str(targets["BASE_LONG"].get("leverage", 5)))
        s_lev = Decimal(str(targets["BASE_SHORT"].get("leverage", 5)))
        l_margin = (self.val_long_notional / l_lev)
        s_margin = (self.val_short_notional / s_lev)
        current_cash = self.tpv - (l_margin + s_margin + self.val_virt_notional)
        available_funds = current_cash + total_proceeds
        
        deficit_actions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)
        for act in deficit_actions:
            lev = Decimal(str(act["leverage"]))
            needed_equity = abs(Decimal(str(act["diff_equity"])))
            
            if available_funds <= Decimal('0') and not ignore_limits: continue
            
            actual_buy_equity = min(needed_equity, available_funds) if not ignore_limits else needed_equity
            actual_buy_usdt = actual_buy_equity * lev

            if actual_buy_usdt >= min_notional or ignore_limits:
                act["diff_usdt"] = float(abs(actual_buy_usdt))
                act["priority"] = 2
                final_actions.append(act)
                available_funds -= actual_buy_equity

        return final_actions
