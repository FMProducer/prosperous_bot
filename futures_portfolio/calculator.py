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
        self.real_equity = Decimal(str(real_equity)) # Чистый кэш (Wallet Balance)
        self.virt_qty = Decimal(str(virt_qty))
        self.virt_entry_price = Decimal(str(virt_entry_price))
        self.targets = targets or {}

        # TPV = Кэш + Рыночная стоимость виртуальной позиции
        self.virt_value = self.virt_qty * self.price
        self.tpv = self.real_equity + self.virt_value

        # PnL для логов: разница между текущей стоимостью и ценой входа
        # Если цена входа не задана, PnL = 0
        if self.virt_entry_price > 0 and self.virt_qty > 0:
            self.virt_pnl_val = (self.price - self.virt_entry_price) * self.virt_qty
        else:
            self.virt_pnl_val = Decimal('0')

        if self.tpv <= 0:
            self.tpv = Decimal('1e-9')

        self.total_tpv = self.tpv + self.siphoning_reserve

        # Общий PnL системы для Heartbeat
        self.total_pnl = self.tpv - self.initial_capital

        # Total PnL % for logs
        self.total_pnl_pct = ((self.tpv / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else Decimal('0')

        # 2. Calculate NAV for each leg
        l_lev = Decimal(str(self.targets.get("BASE_LONG", {}).get("leverage", 5)))
        s_lev = Decimal(str(self.targets.get("BASE_SHORT", {}).get("leverage", 5)))

        l_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        s_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))

        # Value = Initial Margin (Ignoring Unrealized PnL for rebalancing basis to keep Cash logs positive)
        self.val_long = (l_qty * Decimal(str(long_entry_price)) / l_lev) if l_qty > 0 else Decimal('0')
        self.val_short = (s_qty * Decimal(str(short_entry_price)) / s_lev) if s_qty > 0 else Decimal('0')
        self.val_virt = self.virt_qty * self.price

        # PnL contributions for transparency
        self.pnl_l = (l_qty * (self.price - Decimal(str(long_entry_price)))) if l_qty > 0 else Decimal('0')
        self.pnl_s = (s_qty * (Decimal(str(short_entry_price)) - self.price)) if s_qty > 0 else Decimal('0')
        self.pnl_v = self.virt_pnl_val

        # 3. Cash is what's left in the futures wallet that isn't tied up in L/S margin/PnL
        # Since TPV = real_equity + val_virt, and real_equity = val_l + val_s + cash
        self.val_cash = self.tpv - (self.val_long + self.val_short + self.val_virt)
        if self.val_cash < 0 and abs(self.val_cash) < 0.1: self.val_cash = Decimal('0') # Rounding protection

        # 4. Shares calculation (Capital weights)
        self.share_long_raw = (self.val_long / self.tpv)
        self.share_short_raw = (self.val_short / self.tpv)
        self.share_virt_raw = (self.val_virt / self.tpv)
        self.share_cash_raw = (self.val_cash / self.tpv)

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

    def calculate_rebalance(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> Dict:
        """
        Calculate rebalance actions and return a summary of the current state.
        """
        actions = self.calculate_deviations(targets, threshold, ignore_limits)

        return {
            "actions": actions,
            "share_long_pct": float(self.share_long_pct),
            "share_short_pct": float(self.share_short_pct),
            "share_virt_pct": float(self.share_virt_pct),
            "share_cash_pct": float(self.share_cash_pct),
            "tpv": float(self.tpv),
            "total_tpv": float(self.total_tpv),
            "siphoning_reserve": float(self.siphoning_reserve),
            "virt_current_value": float(self.val_virt),
            "pnl_l": float(self.pnl_l),
            "pnl_s": float(self.pnl_s),
            "pnl_v": float(self.pnl_v),
            "total_pnl": float(self.total_pnl),
            "total_pnl_pct": float(self.total_pnl_pct)
        }

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Rebalance based on CAPITAL (Equity) deviations. 
        This allows the portfolio to harvest volatility profit.
        """
        dec_threshold = Decimal(str(threshold))
        actions: List[Dict] = []
        
        # Use raw ratios for maximum precision before rebalancing
        shares: Dict[str, Decimal] = {
            "BASE_LONG": self.share_long_raw,
            "BASE_SHORT": self.share_short_raw,
            "VIRTUAL": self.share_virt_raw
        }

        # Rebalance ONLY legs that actually breached the threshold
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share = Decimal(str(targets[key]["share"]))
            current_share = shares[key]
            diff_share = current_share - target_share # Positive if surplus

            # Only rebalance legs that actually breached the threshold
            if not ignore_limits and abs(diff_share) < dec_threshold:
                if abs(diff_share) > 0:
                    logger.debug(f"Trigger: {key} deviation {diff_share*100:+.2f}% (below threshold)")
                continue

            if abs(diff_share) > 0:
                # Log the trigger reason (will be captured by main.py)
                logger.debug(f"Trigger: {key} deviation {diff_share*100:+.2f}% targets {target_share*100}%")

            if key == "VIRTUAL":
                # Знак: если target > current (deficit), diff_usdt > 0 (BUY)
                # diff_share = current - target. If deficit, diff_share < 0.
                # So BUY is -diff_share * tpv.
                diff_usdt = -diff_share * self.tpv

                # Dust Guard: игнорируем сделки меньше 1 USDT
                if abs(diff_usdt) < Decimal('1.0'):
                    continue

                actions.append({
                    "type": "VIRTUAL_ORDER",
                    "symbol": "VIRTUAL",
                    "base_symbol": "VIRTUAL",
                    "position_side": "BOTH",
                    "diff_usdt": float(diff_usdt),
                    "priority": 1 if diff_usdt > 0 else 3
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
