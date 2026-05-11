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
                 virt_entry_price: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0) -> None:
        
        # Convert all inputs to Decimal for precision
        self.positions = {k: Decimal(str(v)) for k, v in positions.items()}
        self.price = Decimal(str(spot_price))
        self.base_ticker = base_ticker
        self.siphoning_reserve = Decimal(str(siphoning_reserve))
        self.initial_capital = Decimal(str(initial_capital))
        self.real_equity = Decimal(str(real_equity)) # Includes unrealized PnL and collateral of L/S legs
        self.virt_qty = Decimal(str(virt_qty))
        self.virt_entry = Decimal(str(virt_entry_price)) if virt_entry_price > 0 else self.price
        
        # 1. Calculate Virtual Leg Current Market Value
        self.val_virt = self.virt_qty * self.price
        
        # 2. Total Working TPV (Real Account Value + Virtual Spot Value)
        # TPV represents the total liquidation value of the entire swarm unit.
        self.tpv = self.real_equity + self.val_virt
        
        if self.tpv <= 0:
            self.tpv = Decimal('1e-9')
            
        self.total_tpv = self.tpv + self.siphoning_reserve

        # 3. Calculate Equity Value of Real Legs (L + S)
        # Equity = (Notional / Leverage) + Unrealized PnL
        long_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        short_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))
        
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"])) if targets and "BASE_LONG" in targets else Decimal('5.0')
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"])) if targets and "BASE_SHORT" in targets else Decimal('5.0')

        # Use entry prices to separate Collateral from PnL
        l_entry = Decimal(str(long_entry_price)) if long_entry_price > 0 else self.price
        s_entry = Decimal(str(short_entry_price)) if short_entry_price > 0 else self.price

        # The core of Equity rebalancing: position value changes with price
        self.val_long = (long_qty * l_entry / l_lev) + (long_qty * (self.price - l_entry)) if long_qty > 0 else Decimal('0')
        self.val_short = (short_qty * s_entry / s_lev) + (short_qty * (s_entry - self.price)) if short_qty > 0 else Decimal('0')

        # 4. Shares calculation (Capital weights)
        # We calculate raw fractions first to avoid rounding errors accumulation
        self.share_long_raw = (self.val_long / self.tpv)
        self.share_short_raw = (self.val_short / self.tpv)
        self.share_virt_raw = (self.val_virt / self.tpv)

        # Cash is the uninvested capital remainder (Strict Invariant)
        # We allow it to be negative if the sum of other legs > 100% (e.g. commission drain)
        self.share_cash_raw = Decimal('1.0') - self.share_long_raw - self.share_short_raw - self.share_virt_raw

        # Convert to percentages for display and rebalancing logic
        self.share_long_pct = (self.share_long_raw * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_short_pct = (self.share_short_raw * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_virt_pct = (self.share_virt_raw * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_cash_pct = (self.share_cash_raw * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)

        # Final correction to ensure exactly 100.00%
        total_pct = self.share_long_pct + self.share_short_pct + self.share_virt_pct + self.share_cash_pct
        if total_pct != Decimal('100.00'):
            diff = Decimal('100.00') - total_pct
            self.share_cash_pct += diff # Adjust cash by the sub-penny difference

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Rebalance based on CAPITAL (Equity) deviations. 
        This allows the portfolio to harvest volatility profit.
        """
        dec_threshold = Decimal(str(threshold))
        actions: List[Dict] = []
        
        shares: Dict[str, Decimal] = {
            "BASE_LONG": self.share_long_pct / 100,
            "BASE_SHORT": self.share_short_pct / 100,
            "VIRTUAL": self.share_virt_pct / 100
        }

        # Check if rebalance is triggered by any leg
        any_exceeded: bool = dec_threshold < 0
        if not any_exceeded:
            for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
                target_share = Decimal(str(targets[key]["share"]))
                if abs(shares[key] - target_share) > dec_threshold:
                    any_exceeded = True
                    break

        if not any_exceeded:
            return []

        # Rebalance ALL legs to restore Target Equity shares
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share = Decimal(str(targets[key]["share"]))
            current_share = shares[key]
            diff_share = current_share - target_share # Positive if surplus

            if key == "VIRTUAL":
                # For Virtual, diff_usdt is the amount to move to/from Cash
                diff_usdt = diff_share * self.tpv
                actions.append({
                    "type": "VIRTUAL_ORDER",
                    "symbol": "VIRTUAL",
                    "base_symbol": "VIRTUAL",
                    "position_side": "BOTH",
                    "diff_usdt": float(-diff_usdt), # Negative means sell virtual to cash
                    "priority": 1 if diff_share > 0 else 3
                })
            else:
                lev = Decimal(str(targets[key]["leverage"]))
                pos_side: str = "LONG" if key == "BASE_LONG" else "SHORT"
                pos_key: str = f"{self.base_ticker}_{pos_side}"

                # To restore Equity share by X%, we must change Notional volume by (X% * Leverage)
                diff_usdt = -diff_share * self.tpv * lev

                if not ignore_limits:
                    limit = self.tpv * Decimal('0.5') * lev
                    if diff_usdt > limit: diff_usdt = limit
                    if diff_usdt < -limit: diff_usdt = -limit

                is_reduction: bool = diff_usdt < 0

                actions.append({
                    "type": "ORDER",
                    "symbol": pos_key,
                    "base_symbol": self.base_ticker,
                    "position_side": pos_side,
                    "diff_usdt": float(diff_usdt),
                    "priority": 0 if is_reduction else 2
                })

        actions.sort(key=lambda x: x["priority"])
        return actions
