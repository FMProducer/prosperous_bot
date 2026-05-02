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
        
        # 1. Calculate Virtual Leg Current Value (Market Value V)
        dec_virt_basis_price = Decimal(str(virt_basis_price))
        dec_virt_allocated_usdt = Decimal(str(virt_allocated_usdt))
        if dec_virt_basis_price <= 0: dec_virt_basis_price = self.price
        
        self.virt_current_value = dec_virt_allocated_usdt * (self.price / dec_virt_basis_price)
        
        # 2. Total Working TPV (Account Value + Virtual Profit)
        # This is our '100%' base for all rebalancing
        virt_pnl = self.virt_current_value - dec_virt_allocated_usdt
        self.tpv = self.real_equity + virt_pnl
        
        if self.tpv <= 0:
            self.tpv = Decimal('1e-9')
            
        self.total_tpv = self.tpv + self.siphoning_reserve

        # 3. Calculate Actual Market Value of Real Legs (L + S)
        long_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        short_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))
        
        l_lev = Decimal(str(targets["BASE_LONG"]["leverage"])) if targets and "BASE_LONG" in targets else Decimal('5.0')
        s_lev = Decimal(str(targets["BASE_SHORT"]["leverage"])) if targets and "BASE_SHORT" in targets else Decimal('5.0')

        # Use entry prices if provided, otherwise assume current price (no PnL)
        l_entry = Decimal(str(long_entry_price)) if long_entry_price > 0 else self.price
        s_entry = Decimal(str(short_entry_price)) if short_entry_price > 0 else self.price

        # Market Value = Collateral Basis + Unrealized PnL
        # Note: Collateral Basis = (Qty * EntryPrice) / Leverage
        self.val_long = (long_qty * l_entry / l_lev) + (long_qty * (self.price - l_entry)) if long_qty > 0 else Decimal('0')
        self.val_short = (short_qty * s_entry / s_lev) + (short_qty * (s_entry - self.price)) if short_qty > 0 else Decimal('0')
        self.val_virt = self.virt_current_value

        # 4. Shares calculation
        self.share_long_pct = (self.val_long / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        self.share_short_pct = (self.val_short / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        self.share_virt_pct = (self.val_virt / self.tpv * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN)
        
        # Cash is the remainder (Uninvested capital)
        self.share_cash_pct = (Decimal('100.0') - self.share_long_pct - self.share_short_pct - self.share_virt_pct)
        if self.share_cash_pct < 0: self.share_cash_pct = Decimal('0')

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
        # Принудительная ребалансировка если threshold < 0
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

                # Чтобы изменить долю капитала на X%, нужно изменить НОМИНАЛ на (X% * Плечо)
                # Если у нас избыток доли (diff_share > 0), нам нужно ОТРИЦАТЕЛЬНОЕ изменение (продажа)
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
