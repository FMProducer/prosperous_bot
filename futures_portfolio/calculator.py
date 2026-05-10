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
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0) -> None:
        
        # Convert all inputs to Decimal for precision
        self.positions = {k: Decimal(str(v)) for k, v in positions.items()}
        self.price = Decimal(str(spot_price))
        self.base_ticker = base_ticker
        self.siphoning_reserve = Decimal(str(siphoning_reserve))
        self.initial_capital = Decimal(str(initial_capital))
        self.real_equity = Decimal(str(real_equity))
        self.virt_qty = Decimal(str(virt_qty))
        
        # 1. Calculate Virtual Leg Current Notional Value
        self.notional_virt = self.virt_qty * self.price
        
        # 2. Total Working TPV (Real Equity + Virtual Market Value)
        # Note: real_equity includes unrealized PnL and current collateral of L/S legs.
        # TPV represents the total liquidity if all positions were closed.
        self.tpv = self.real_equity + self.notional_virt
        
        if self.tpv <= 0:
            self.tpv = Decimal('1e-9')
            
        self.total_tpv = self.tpv + self.siphoning_reserve

        # 3. Calculate Notional Value of Real Legs (L + S)
        long_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        short_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))
        
        self.notional_long = long_qty * self.price
        self.notional_short = short_qty * self.price

        # 4. Target-normalized Shares calculation (NOTIONAL / (TPV * LEVERAGE))
        # This reflects the intended capital allocation (Equity weight) in a Notional-neutral world.
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"])) if targets and "BASE_LONG" in targets else Decimal('5.0')
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"])) if targets and "BASE_SHORT" in targets else Decimal('5.0')

        self.share_long_pct = (self.notional_long / (self.tpv * l_lev) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        self.share_short_pct = (self.notional_short / (self.tpv * s_lev) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        self.share_virt_pct = (self.notional_virt / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        
        # Cash is the remainder of the intended capital allocation
        self.share_cash_pct = (Decimal('100.0') - self.share_long_pct - self.share_short_pct - self.share_virt_pct)
        if self.share_cash_pct < 0: self.share_cash_pct = Decimal('0')

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Rebalance based on NOTIONAL deviations to maintain Delta Neutrality.
        """
        dec_threshold = Decimal(str(threshold))
        actions: List[Dict] = []
        
        # Current "Capital-equivalent" shares based on Notional
        shares: Dict[str, Decimal] = {
            "BASE_LONG": self.share_long_pct / 100,
            "BASE_SHORT": self.share_short_pct / 100,
            "VIRTUAL": self.share_virt_pct / 100
        }

        # Check if any leg exceeded the threshold
        any_exceeded: bool = dec_threshold < 0
        if not any_exceeded:
            for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
                target_share = Decimal(str(targets[key]["share"]))
                if abs(shares[key] - target_share) > dec_threshold:
                    any_exceeded = True
                    break

        if not any_exceeded:
            return []

        # Rebalance ALL legs to their Target Notional
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share = Decimal(str(targets[key]["share"]))
            lev = Decimal(str(targets[key]["leverage"])) if "leverage" in targets[key] else Decimal('1.0')
            
            # Target Notional Value = TPV * Target Share * Leverage
            target_notional = self.tpv * target_share * lev
            
            if key == "VIRTUAL":
                current_notional = self.notional_virt
                diff_usdt = target_notional - current_notional
                
                actions.append({
                    "type": "VIRTUAL_ORDER",
                    "symbol": "VIRTUAL",
                    "base_symbol": "VIRTUAL",
                    "position_side": "BOTH",
                    "diff_usdt": float(diff_usdt), # Positive means BUY (add to virtual, remove from cash)
                    "priority": 1 if diff_usdt < 0 else 3 # Sell first to free up cash
                })
            else:
                pos_side: str = "LONG" if key == "BASE_LONG" else "SHORT"
                pos_key: str = f"{self.base_ticker}_{pos_side}"
                current_notional = self.notional_long if pos_side == "LONG" else self.notional_short
                
                # diff_usdt > 0 means we need MORE notional (Buy for Long, Sell for Short)
                diff_usdt = target_notional - current_notional

                if not ignore_limits:
                    limit = self.tpv * Decimal('0.5') * lev
                    if diff_usdt > limit: diff_usdt = limit
                    if diff_usdt < -limit: diff_usdt = -limit

                # In Notional mode, "reduction" means moving Notional closer to zero.
                # However, for simplicity and safety, we prioritize "SELL" orders if TPV is tight.
                # executor.py determines side:
                # If Long: diff > 0 -> BUY, diff < 0 -> SELL
                # If Short: diff > 0 -> SELL, diff < 0 -> BUY
                is_sell: bool = (pos_side == "LONG" and diff_usdt < 0) or (pos_side == "SHORT" and diff_usdt > 0)

                actions.append({
                    "type": "ORDER",
                    "symbol": pos_key,
                    "base_symbol": self.base_ticker,
                    "position_side": pos_side,
                    "diff_usdt": float(diff_usdt),
                    "priority": 0 if is_sell else 2
                })

        actions.sort(key=lambda x: x["priority"])
        return actions
