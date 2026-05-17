import logging
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

logger = logging.getLogger(__name__)

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class PortfolioCalculator:
    # Caching common constants to prevent redundant Decimal instantiations
    _DEC_ZERO = Decimal('0')
    _DEC_ONE = Decimal('1.0')
    _DEC_MIN_NOTIONAL = Decimal('6.1')
    _DEC_100 = Decimal('100')
    _DEC_100_00 = Decimal('100.00')

    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_qty: float, 
                 long_entry_price: float = 0.0, short_entry_price: float = 0.0,
                 virt_entry_price: float = 0.0, virt_debt: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0,
                 last_rebalance_price: float = 0.0) -> None:
        
        # Convert all inputs to Decimal for precision
        self.positions = {k: Decimal(str(v)) for k, v in positions.items()}
        self.price = Decimal(str(spot_price))
        self.base_ticker = base_ticker
        self.siphoning_reserve = Decimal(str(siphoning_reserve))
        self.initial_capital = Decimal(str(initial_capital))
        self.real_equity = Decimal(str(real_equity)) # Wallet Balance (Cash + Margin, NO PnL)
        self.virt_qty = Decimal(str(virt_qty))
        self.virt_debt = Decimal(str(virt_debt))

        # Pre-convert targets to Decimal once
        self.targets = {}
        if targets:
            for k, v in targets.items():
                self.targets[k] = {
                    "share": Decimal(str(v.get("share", 0))),
                    "leverage": Decimal(str(v.get("leverage", 1)))
                }

        self.last_rebalance_price = Decimal(str(last_rebalance_price)) if last_rebalance_price > 0 else self.price

        # 1. PnL contributions for Heartbeat (RELATIVE to last rebalance)
        # This ensures the sign of $ PnL always matches the sign of % deviation.
        l_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", self._DEC_ZERO))
        s_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", self._DEC_ZERO))

        self.pnl_l = (l_qty * (self.price - self.last_rebalance_price)) if l_qty > 0 else self._DEC_ZERO
        self.pnl_s = (s_qty * (self.last_rebalance_price - self.price)) if s_qty > 0 else self._DEC_ZERO
        self.pnl_v = (self.virt_qty * (self.price - self.last_rebalance_price)) if self.virt_qty > 0 else self._DEC_ZERO

        # Real MTM PnL (for TPV) relative to entry prices for futures
        mtm_pnl_l = (l_qty * (self.price - Decimal(str(long_entry_price)))) if l_qty > 0 else self._DEC_ZERO
        mtm_pnl_s = (s_qty * (Decimal(str(short_entry_price)) - self.price)) if s_qty > 0 else self._DEC_ZERO
        
        # 1. Рассчитываем ЧИСТЫЙ PnL виртуальной ноги (MTM)
        # Если есть цена входа, считаем по ней. Если нет — по дельте от долга.
        self.virt_value = self.virt_qty * self.price
        if virt_entry_price > 0:
            self.pnl_v = (self.price - Decimal(str(virt_entry_price))) * self.virt_qty
        elif self.virt_debt > 0:
            self.pnl_v = self.virt_value - self.virt_debt
        else:
            v_target_share = self.targets.get("VIRTUAL", {}).get("share", Decimal('0.35'))
            self.pnl_v = self.virt_value - (self.initial_capital * v_target_share)

        # [SSOT] TPV = Доступный баланс (кэш) + PnL фьючерсов + Рыночная стоимость виртуальной ноги
        # real_equity должен передаваться как "Чистый кэш" (Wallet Balance - virt_debt)
        self.tpv = self.real_equity + mtm_pnl_l + mtm_pnl_s + self.virt_value

        if self.tpv <= 0:
            self.tpv = Decimal('1e-9')

        self.total_tpv = self.tpv + self.siphoning_reserve

        # Total PnL = Sum of all contributions
        self.total_pnl = self.tpv - self.initial_capital

        # Total PnL % = (TPV / initial_capital - 1) * 100
        self.total_pnl_pct = ((self.tpv / self.initial_capital) - 1) * self._DEC_100 if self.initial_capital > 0 else self._DEC_ZERO

        # 4. Calculate NAV for each leg
        l_lev = self.targets.get("BASE_LONG", {}).get("leverage", Decimal('5'))
        s_lev = self.targets.get("BASE_SHORT", {}).get("leverage", Decimal('5'))

        # Value = Initial Margin + MTM PnL (Actual Liquidation Equity of the position)
        self.val_long = ((l_qty * Decimal(str(long_entry_price)) / l_lev) + mtm_pnl_l) if l_qty > 0 else self._DEC_ZERO
        self.val_short = ((s_qty * Decimal(str(short_entry_price)) / s_lev) + mtm_pnl_s) if s_qty > 0 else self._DEC_ZERO
        self.val_virt = self.virt_value
        
        # 5. Cash is the remaining liquidity (Free Wallet Balance)
        self.val_cash = self.tpv - (self.val_long + self.val_short + self.val_virt)
        if self.val_cash < 0 and abs(self.val_cash) < 0.1: self.val_cash = self._DEC_ZERO # Rounding protection

        # 6. Shares calculation (Capital weights against true TPV)
        self.share_long_raw = (self.val_long / self.tpv)
        self.share_short_raw = (self.val_short / self.tpv)
        self.share_virt_raw = (self.val_virt / self.tpv)
        self.share_cash_raw = (self.val_cash / self.tpv)

        # Convert to percentages for display and rebalancing logic
        self.share_long_pct = (self.share_long_raw * self._DEC_100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_short_pct = (self.share_short_raw * self._DEC_100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_virt_pct = (self.share_virt_raw * self._DEC_100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        self.share_cash_pct = (self.share_cash_raw * self._DEC_100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)

        # Final correction to ensure exactly 100.00%
        total_pct = self.share_long_pct + self.share_short_pct + self.share_virt_pct + self.share_cash_pct
        if total_pct != self._DEC_100_00:
            diff = self._DEC_100_00 - total_pct
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
            "val_long": float(self.val_long),
            "val_short": float(self.val_short),
            "val_virt": float(self.val_virt),
            "val_cash": float(self.val_cash),
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
        
        # Actions split by intent
        surplus_actions: List[Dict] = []
        deficit_actions: List[Dict] = []
        
        # Use raw ratios for maximum precision
        shares: Dict[str, Decimal] = {
            "BASE_LONG": self.share_long_raw,
            "BASE_SHORT": self.share_short_raw,
            "VIRTUAL": self.share_virt_raw
        }

        # 1. Gather all legs that reached the threshold
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            # Use pre-converted targets if available
            target_data = self.targets.get(key)
            if target_data:
                target_share = target_data["share"]
                lev = target_data["leverage"]
            else:
                target_share = Decimal(str(targets[key]["share"]))
                lev = Decimal(str(targets[key].get("leverage", 1)))

            current_share = shares[key]
            diff_share = current_share - target_share # Positive if surplus (actual > target)
            
            if not ignore_limits and abs(diff_share) < dec_threshold:
                continue

            # Calculate theoretical diff_usdt
            # diff_usdt = -diff_share * self.tpv * leverage
            # Surplus (+) -> Negative diff_usdt (SELL/Reduction)
            # Deficit (-) -> Positive diff_usdt (BUY/Expansion)
            diff_usdt = -diff_share * self.tpv * lev
            diff_equity = -diff_share * self.tpv # Real cash (margin) movement

            if abs(diff_usdt) < self._DEC_ONE: # Fundamental rounding filter
                continue

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

        # 2. Final actions list starts with all SELLs (Priority 0)
        final_actions: List[Dict] = []
        
        # Proceeds must be calculated in pure Equity (Cash) terms
        total_proceeds = self._DEC_ZERO
        for act in surplus_actions:
            if abs(Decimal(str(act["diff_usdt"]))) >= self._DEC_MIN_NOTIONAL or ignore_limits:
                act["priority"] = 0
                final_actions.append(act)
                total_proceeds += abs(Decimal(str(act["diff_equity"])))

        # 3. Calculate available funds for BUYs (Strict Cash Accounting)
        available_funds = self.val_cash + total_proceeds
        
        # 4. Process Deficits with Priority (VIRTUAL first)
        deficit_actions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)
        
        for act in deficit_actions:
            lev = Decimal(str(act["leverage"]))
            needed_equity = abs(Decimal(str(act["diff_equity"])))
            
            if available_funds <= self._DEC_ZERO and not ignore_limits:
                continue

            # Cap purchasing power by available cash (Equity)
            if needed_equity > available_funds and not ignore_limits:
                actual_buy_equity = available_funds
            else:
                actual_buy_equity = needed_equity
                
            # Convert approved Equity back to Notional for the executor
            actual_buy_usdt = actual_buy_equity * lev

            if actual_buy_usdt >= self._DEC_MIN_NOTIONAL or ignore_limits:
                act["diff_usdt"] = float(abs(actual_buy_usdt))
                act["priority"] = 2
                final_actions.append(act)
                available_funds -= actual_buy_equity

        return final_actions
