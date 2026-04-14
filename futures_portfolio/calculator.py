import logging
from typing import Dict, List
import math

logger = logging.getLogger(__name__)

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_basis_price: float, virt_allocated_usdt: float):
        self.positions = positions
        self.price = spot_price
        self.real_equity = real_equity
        
        if virt_basis_price <= 0: virt_basis_price = spot_price
        price_change = spot_price / virt_basis_price
        self.virt_current_value = virt_allocated_usdt * price_change
        
        # TPV
        self.tpv = real_equity + (self.virt_current_value - virt_allocated_usdt)
        
        # Защита от NaN
        if math.isnan(self.tpv) or self.tpv <= 0:
            self.tpv = real_equity if real_equity > 0 else 1e-9

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float) -> List[Dict]:
        deviations = []
        
        # Проверка Long
        long_notional = abs(self.positions.get("BTCUSDT_LONG", 0.0)) * self.price
        share_long = (long_notional / targets["BTCUSDT_LONG"]["leverage"]) / self.tpv if self.tpv > 0 else 0
        
        # Проверка Short
        short_notional = abs(self.positions.get("BTCUSDT_SHORT", 0.0)) * self.price
        share_short = (short_notional / targets["BTCUSDT_SHORT"]["leverage"]) / self.tpv if self.tpv > 0 else 0
        
        # Проверка Virtual
        share_virt = self.virt_current_value / self.tpv if self.tpv > 0 else 0
        
        # Максимальное отклонение
        max_dev = max(abs(share_long - targets["BTCUSDT_LONG"]["share"]),
                      abs(share_short - targets["BTCUSDT_SHORT"]["share"]),
                      abs(share_virt - targets["BTC_VIRTUAL"]["share"]))
        
        if max_dev > threshold:
            for key in ["BTCUSDT_LONG", "BTCUSDT_SHORT"]:
                cfg = targets[key]
                target_notional = self.tpv * cfg["share"] * cfg["leverage"]
                current_notional = abs(self.positions.get(key, 0.0)) * self.price
                diff_usdt = target_notional - current_notional
                
                # Ограничение: не меняем более чем на 50% от TPV за один раз (защита от прыжков)
                max_change = self.tpv * 0.5 * cfg["leverage"]
                if abs(diff_usdt) > max_change:
                    diff_usdt = math.copysign(max_change, diff_usdt)

                deviations.append({
                    "symbol": key,
                    "current_share": share_long if "LONG" in key else share_short,
                    "target_share": cfg["share"],
                    "diff_usdt": diff_usdt
                })
        
        return deviations
