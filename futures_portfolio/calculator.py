import logging
from typing import Dict, List
import math

logger = logging.getLogger(__name__)

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_basis_price: float, virt_allocated_usdt: float, 
                 long_entry_price: float = 0.0, short_entry_price: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None):
        self.positions = positions
        self.price = spot_price
        self.base_ticker = base_ticker
        self.siphoning_reserve = siphoning_reserve
        
        # Виртуальная доля
        if virt_basis_price <= 0: virt_basis_price = spot_price
        price_change = spot_price / virt_basis_price
        self.virt_current_value = virt_allocated_usdt * price_change
        
        # Общий TPV (включая накопленный резерв)
        self.total_tpv = real_equity + (self.virt_current_value - virt_allocated_usdt)
        
        # Активный TPV для расчетов (без резерва)
        self.tpv = self.total_tpv - self.siphoning_reserve
        
        # Защита от NaN
        if math.isnan(self.tpv) or self.tpv <= 0:
            self.tpv = 1e-9

        # Текущие доли (Share %)
        long_notional = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0)) * self.price
        short_notional = abs(self.positions.get(f"{self.base_ticker}_SHORT", 0.0)) * self.price
        
        # Динамическое получение плеча для логирования
        l_lev = targets["BASE_LONG"]["leverage"] if targets and "BASE_LONG" in targets else 5.0
        s_lev = targets["BASE_SHORT"]["leverage"] if targets and "BASE_SHORT" in targets else 5.0

        # Сохраняем для логирования (в целых числах процентов для красоты)
        self.share_long_pct = round((long_notional / l_lev) / self.tpv * 100) if self.tpv > 0 else 0
        self.share_short_pct = round((short_notional / s_lev) / self.tpv * 100) if self.tpv > 0 else 0
        self.share_virt_pct = round(self.virt_current_value / self.tpv * 100) if self.tpv > 0 else 0

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        deviations = []

        # Проверка Long
        long_notional = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0)) * self.price
        share_long = (long_notional / targets["BASE_LONG"]["leverage"]) / self.tpv if self.tpv > 0 else 0

        # Проверка Short
        short_notional = abs(self.positions.get(f"{self.base_ticker}_SHORT", 0.0)) * self.price
        share_short = (short_notional / targets["BASE_SHORT"]["leverage"]) / self.tpv if self.tpv > 0 else 0

        # Проверка Virtual
        share_virt = self.virt_current_value / self.tpv if self.tpv > 0 else 0

        # Максимальное отклонение
        max_dev = max(abs(share_long - targets["BASE_LONG"]["share"]),
                      abs(share_short - targets["BASE_SHORT"]["share"]),
                      abs(share_virt - targets["VIRTUAL"]["share"]))

        if max_dev > threshold:
            for key in ["BASE_LONG", "BASE_SHORT"]:
                cfg = targets[key]
                target_notional = self.tpv * cfg["share"] * cfg["leverage"]

                # Сопоставление ключей конфига с позициями
                pos_key = f"{self.base_ticker}_LONG" if key == "BASE_LONG" else f"{self.base_ticker}_SHORT"
                current_notional = abs(self.positions.get(pos_key, 0.0)) * self.price
                diff_usdt = target_notional - current_notional
                
                # Ограничение: не меняем более чем на 50% от TPV за один раз (пропускаем, если ignore_limits=True)
                if not ignore_limits:
                    max_change = self.tpv * 0.5 * cfg["leverage"]
                    if abs(diff_usdt) > max_change:
                        diff_usdt = math.copysign(max_change, diff_usdt)

                deviations.append({
                    "symbol": pos_key,
                    "current_share": share_long if "LONG" in pos_key else share_short,
                    "target_share": cfg["share"],
                    "diff_usdt": diff_usdt
                })

        return deviations
