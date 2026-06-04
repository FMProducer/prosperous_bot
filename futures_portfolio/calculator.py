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
        # This ensures the sign of $ PnL always matches the sign of % deviation.
        l_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", Decimal('0')))
        s_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", Decimal('0')))

        self.pnl_l = (l_qty * (self.price - self.last_rebalance_price)) if l_qty > 0 else Decimal('0')
        self.pnl_s = (s_qty * (self.last_rebalance_price - self.price)) if s_qty > 0 else Decimal('0')
        self.pnl_v = (self.virt_qty * (self.price - self.last_rebalance_price)) if self.virt_qty > 0 else Decimal('0')

        # Real MTM PnL (for TPV) relative to entry prices for futures
        mtm_pnl_l = (l_qty * (self.price - Decimal(str(long_entry_price)))) if l_qty > 0 else Decimal('0')
        mtm_pnl_s = (s_qty * (Decimal(str(short_entry_price)) - self.price)) if s_qty > 0 else Decimal('0')
        
        # 1. Рассчитываем ЧИСТЫЙ PnL виртуальной ноги (MTM)
        # Если есть цена входа, считаем по ней. Если нет — по дельте от долга.
        self.virt_value = self.virt_qty * self.price
        if virt_entry_price > 0:
            self.pnl_v = (self.price - Decimal(str(virt_entry_price))) * self.virt_qty
        elif self.virt_debt > 0:
            self.pnl_v = self.virt_value - self.virt_debt
        else:
            v_target_share = Decimal(str(self.targets.get("VIRTUAL", {}).get("share", 0.35)))
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
        self.total_pnl_pct = ((self.tpv / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else Decimal('0')

        # 4. Calculate NAV for each leg
        l_lev = Decimal(str(self.targets.get("BASE_LONG", {}).get("leverage", 5)))
        s_lev = Decimal(str(self.targets.get("BASE_SHORT", {}).get("leverage", 5)))

        # Value = Initial Margin + MTM PnL (Actual Liquidation Equity of the position)
        self.val_long = ((l_qty * Decimal(str(long_entry_price)) / l_lev) + mtm_pnl_l) if l_qty > 0 else Decimal('0')
        self.val_short = ((s_qty * Decimal(str(short_entry_price)) / s_lev) + mtm_pnl_s) if s_qty > 0 else Decimal('0')
        self.val_virt = self.virt_value
        
        # 5. Cash is the remaining liquidity (Free Wallet Balance)
        self.val_cash = self.tpv - (self.val_long + self.val_short + self.val_virt)
        if self.val_cash < 0 and abs(self.val_cash) < 0.1: self.val_cash = Decimal('0') # Rounding protection

        # 6. Shares calculation (Capital weights against true TPV)
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

    def calculate_rebalance(self, targets: Dict[str, Dict], threshold_surplus: float, threshold_deficit: float, 
                           ignore_limits: bool = False, allow_surplus_sell: bool = True, force_block: bool = False,
                           current_equity: float = None) -> Dict:
        """
        Calculate rebalance actions and return a summary of the current state.
        current_equity: if provided, use this value for order sizing instead of self.initial_capital.
                        This allows dynamic scaling of order sizes with account growth.
        allow_surplus_sell: if False, surplus (profit-taking) actions are blocked.
        force_block: if True, ALL actions are blocked (Net Move Guard active).
        """
        dev_res = self.calculate_deviations(targets, threshold_surplus, threshold_deficit, 
                                           ignore_limits, allow_surplus_sell, force_block, current_equity)
        # calculate_deviations возвращает dict с actions, available_funds, tpv, share_*
        actions = dev_res["actions"]

        return {
            "actions": actions,
            "available_funds": dev_res["available_funds"],
            "share_long_pct": dev_res["share_long_pct"],
            "share_short_pct": dev_res["share_short_pct"],
            "share_virt_pct": dev_res["share_virt_pct"],
            "share_cash_pct": dev_res["share_cash_pct"],
            "val_long": float(self.val_long),
            "val_short": float(self.val_short),
            "val_virt": float(self.val_virt),
            "val_cash": float(self.val_cash),
            "tpv": dev_res["tpv"],
            "total_tpv": dev_res["total_tpv"],
            "siphoning_reserve": float(self.siphoning_reserve),
            "virt_current_value": float(self.val_virt),
            "pnl_l": float(self.pnl_l),
            "pnl_s": float(self.pnl_s),
            "pnl_v": float(self.pnl_v),
            "total_pnl": float(self.total_pnl),
            "total_pnl_pct": float(self.total_pnl_pct)
        }

    def calculate_deviations(self, targets: Dict[str, Dict], threshold_surplus: float, threshold_deficit: float, 
                            ignore_limits: bool = False, allow_surplus_sell: bool = True, force_block: bool = False,
                            current_equity: float = None) -> Dict:
        """
        Rebalance based on CAPITAL (Equity) deviations.
        current_equity: if provided, use this value instead of self.initial_capital for order sizing.
                        This allows dynamic scaling of order sizes with account growth.
        """
        if force_block:
            return {
                "actions": [],
                "available_funds": float(max(Decimal('0'), self.val_cash)),
                "share_long_pct": float(self.share_long_pct),
                "share_short_pct": float(self.share_short_pct),
                "share_virt_pct": float(self.share_virt_pct),
                "share_cash_pct": float(self.share_cash_pct),
                "tpv": float(self.tpv),
                "total_tpv": float(self.total_tpv),
            }

        # Use dynamic equity for order sizing instead of static initial_capital
        equity_base = Decimal(str(current_equity)) if current_equity is not None else self.initial_capital

        dec_threshold_surplus = Decimal(str(threshold_surplus))
        dec_threshold_deficit = Decimal(str(threshold_deficit))
        min_notional = self.min_notional
        
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
            target_share = Decimal(str(targets[key]["share"]))
            current_share = shares[key]
            diff_share = current_share - target_share # Positive if surplus (actual > target)
            
            if not ignore_limits:
                if diff_share > 0 and abs(diff_share) < dec_threshold_surplus:
                    continue
                elif diff_share < 0 and abs(diff_share) < dec_threshold_deficit:
                    continue

            # Calculate theoretical diff_usdt
            # Use dynamic equity_base (current_equity or initial_capital) for order sizing
            # This ensures positions scale with account growth
            lev = Decimal(str(targets[key].get("leverage", 1)))
            diff_usdt = -diff_share * equity_base * lev
            diff_equity = -diff_share * equity_base

            if abs(diff_usdt) < Decimal('1.0'): # Fundamental rounding filter
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

        # 2. Assemble final actions with Anti-Churn FUSE logic
        final_actions: List[Dict] = []
        total_proceeds = Decimal('0')

        # Last rebalance price for FUSE checks
        last_reb = self.last_rebalance_price
        # Cold start check: if last_reb is current price or ignore_limits is True, bypass FUSE
        bypass_fuse = ignore_limits or last_reb == self.price

        # Phase 1: Process SURPLUS (Priority 0)
        # PnL PROTECTION: If allow_surplus_sell is False, skip all surplus actions.
        # This prevents selling "winners" while the portfolio is in drawdown,
        # which would lock in profits on one side while losses accumulate on the other.
        for act in surplus_actions:
            if not allow_surplus_sell:
                logger.debug(f"🛡️ PnL GUARD: Surplus {act['key']} blocked (portfolio PnL < 0)")
                continue

            if abs(Decimal(str(act["diff_usdt"]))) < min_notional and not ignore_limits:
                continue

            act["priority"] = 0
            is_valid = True

            if not bypass_fuse:
                is_inverse = (act["position_side"] == "SHORT")
                if not is_inverse:
                    limit_price = last_reb * (1 + dec_threshold_surplus)
                    is_valid = self.price >= limit_price
                else:
                    limit_price = last_reb * (1 - dec_threshold_surplus)
                    is_valid = self.price <= limit_price

            if is_valid:
                final_actions.append(act)
                total_proceeds += abs(Decimal(str(act["diff_equity"])))
            else:
                logger.debug(f"🚫 FUSE (SURPLUS): {act['key']} blocked. Price {self.price:.6g} vs Limit {limit_price:.6g}")

        # 3. Calculate available funds for BUYs (Strict Cash Accounting)
        # Use only bot's internal cash (val_cash = TPV - positions).
        # val_cash is NOT exchange margin — it's the bot's free equity reserve.
        # Exception: first startup (ignore_limits=True, no positions) to build initial positions.
        if ignore_limits and total_proceeds == 0:
            available_funds = self.val_cash  # Initial capital for first position building
        else:
            available_funds = max(Decimal('0'), self.val_cash)  # Bot cash reserve only
        
        # 4. Process Deficits with Priority (VIRTUAL first, Priority 2)
        deficit_actions.sort(key=lambda x: 0 if x["key"] == "VIRTUAL" else 1)
        
        for act in deficit_actions:
            lev = Decimal(str(act["leverage"]))
            needed_equity = abs(Decimal(str(act["diff_equity"])))
            
            if available_funds <= Decimal('0') and not ignore_limits:
                continue

            # FUSE Check for Deficits — DISABLED.
            # Deficit purchases are driven by cash reserve (val_cash), not margin.
            # Buying deficit legs immediately restores balance; waiting for a
            # further price move (FUSE) only lets the imbalance grow.
            is_valid = True

            # Cap purchasing power by available cash (Equity)
            if needed_equity > available_funds and not ignore_limits:
                actual_buy_equity = available_funds
            else:
                actual_buy_equity = needed_equity
                
            # Convert approved Equity back to Notional for the executor
            actual_buy_usdt = actual_buy_equity * lev

            if actual_buy_usdt >= min_notional or ignore_limits:
                act["diff_usdt"] = float(abs(actual_buy_usdt))
                act["priority"] = 2
                final_actions.append(act)
                available_funds -= actual_buy_equity

        return {
            "actions": final_actions,
            "available_funds": float(available_funds),
            "tpv": float(self.tpv),
            "total_tpv": float(self.total_tpv),
            "share_long_pct": float(self.share_long_pct),
            "share_short_pct": float(self.share_short_pct),
            "share_virt_pct": float(self.share_virt_pct),
            "share_cash_pct": float(self.share_cash_pct)
        }
