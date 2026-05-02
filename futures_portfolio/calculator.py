import logging
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

logger = logging.getLogger(__name__)

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_basis_price: float, virt_allocated_usdt: float, 
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
        
        # 1. Calculate Virtual Leg Current Value (Recalculated every iteration)
        dec_virt_basis_price = Decimal(str(virt_basis_price))
        dec_virt_allocated_usdt = Decimal(str(virt_allocated_usdt))
        
        if dec_virt_basis_price <= 0: 
            dec_virt_basis_price = self.price
            
        # V_current = Qty_virt * Price_current = (Allocated / Basis) * Price_current
        self.virt_current_value = dec_virt_allocated_usdt * (self.price / dec_virt_basis_price)
        
        # 2. Calculate Total Portfolio Value (TPV)
        # TPV = Real_Equity (exchange balance) + Virtual_PnL
        # Virtual_PnL = Current_Value - Initial_Allocation
        self.virt_pnl = self.virt_current_value - dec_virt_allocated_usdt
        self.tpv = self.real_equity + self.virt_pnl
        
        # Total TPV including siphoned reserve for SAFE check
        self.total_tpv = self.tpv + self.siphoning_reserve

        # 3. Calculate Real Legs (Long/Short) Pro-rata Values
        # We assign the 'real' part of the portfolio (Real_Equity - Virtual_Buffer) to L/S
        # Virtual_Buffer is the dec_virt_allocated_usdt (the capital we simulate as spot)
        val_real_total = self.real_equity - dec_virt_allocated_usdt
        
        long_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        short_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))
        
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"])) if targets and "BASE_LONG" in targets else Decimal('5.0')
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"])) if targets and "BASE_SHORT" in targets else Decimal('5.0')

        self.notional_long = long_qty * self.price
        self.notional_short = short_qty * self.price

        est_l = (self.notional_long / l_lev) if l_lev > 0 else Decimal('0')
        est_s = (self.notional_short / s_lev) if s_lev > 0 else Decimal('0')
        est_sum = est_l + est_s

        if est_sum > 0:
            self.val_long = (est_l / est_sum) * val_real_total
            self.val_short = (est_s / est_sum) * val_real_total
        else:
            # Fallback to target ratios if no positions exist
            l_t = Decimal(str(targets["BASE_LONG"]["share"]))
            s_t = Decimal(str(targets["BASE_SHORT"]["share"]))
            self.val_long = (l_t / (l_t + s_t)) * val_real_total
            self.val_short = (s_t / (l_t + s_t)) * val_real_total

        # 4. Calculate shares (Guaranteed to sum to 100.0%)
        # Share_i = Value_i / TPV
        # Note: val_long + val_short + virt_current_value = (Real_Equity - Alloc) + Virt_Current = Real_Equity + Virt_PnL = TPV.
        if self.tpv > 0:
            self.share_long_pct = (self.val_long / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
            self.share_short_pct = (self.val_short / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
            self.share_virt_pct = (Decimal('100.0') - self.share_long_pct - self.share_short_pct)
        else:
            self.share_long_pct = self.share_short_pct = self.share_virt_pct = Decimal('0')

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Ребалансировка портфеля. Если хоть одна нога превысила порог, пересчитываем всё.
        """
        dec_threshold = Decimal(str(threshold))
        actions: List[Dict] = []
        shares: Dict[str, Decimal] = {
            "BASE_LONG": self.share_long_pct / 100,
            "BASE_SHORT": self.share_short_pct / 100,
            "VIRTUAL": self.share_virt_pct / 100
        }

        # Проверяем, превышен ли порог хотя бы одной ногой
        any_exceeded: bool = dec_threshold < 0

        if not any_exceeded:
            for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
                target_share = Decimal(str(targets[key]["share"]))
                if abs(shares[key] - target_share) > dec_threshold:
                    any_exceeded = True
                    break

        if not any_exceeded:
            return []

        # Ребалансируем ВСЕ ноги
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share = Decimal(str(targets[key]["share"]))
            current_share = shares[key]
            diff_share = current_share - target_share # Положительно при ИЗБЫТКЕ

            if key == "VIRTUAL":
                actions.append({
                    "type": "VIRTUAL_RESET",
                    "symbol": "VIRTUAL",
                    "base_symbol": "VIRTUAL",
                    "position_side": "BOTH",
                    "diff_usdt": float(diff_share * self.tpv),
                    "priority": 1 if diff_share > 0 else 3
                })
            else:
                cfg: Dict = targets[key]
                lev = Decimal(str(cfg["leverage"]))
                pos_side: str = "LONG" if key == "BASE_LONG" else "SHORT"
                pos_key: str = f"{self.base_ticker}_{pos_side}"

                # ПОРТФЕЛЬНАЯ ФОРМУЛА:
                # Чтобы изменить долю капитала на X%, нужно изменить НОМИНАЛ на (X% * Плечо)
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
