import logging
from typing import Dict, List
import math

logger = logging.getLogger(__name__)

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_basis_price: float, virt_allocated_usdt: float, 
                 long_entry_price: float = 0.0, short_entry_price: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0):
        self.positions = positions
        self.price = spot_price
        self.base_ticker = base_ticker
        self.siphoning_reserve = siphoning_reserve
        self.initial_capital = initial_capital
        
        # Виртуальная доля
        if virt_basis_price <= 0: virt_basis_price = spot_price
        price_change = spot_price / virt_basis_price
        self.virt_current_value = virt_allocated_usdt * price_change
        
        # Общий TPV (включая накопленный резерв)
        self.total_tpv = real_equity + (self.virt_current_value - virt_allocated_usdt) + self.siphoning_reserve
        
        # Активный TPV для расчетов: если мы в просадке, используем SAFE для маржи
        if self.total_tpv < self.initial_capital:
            self.tpv = self.total_tpv # Recovery mode: используем всё
        else:
            self.tpv = self.total_tpv - self.siphoning_reserve
        
        # Защита от NaN
        if math.isnan(self.tpv) or self.tpv <= 0:
            self.tpv = 1e-9

        # Расчет стоимости позиций (Allocated Capital + PnL)
        # Для Long: Value = (Initial_Notional / Lev) + (Current_Notional - Initial_Notional)
        # Для Short: Value = (Initial_Notional / Lev) + (Initial_Notional - Current_Notional)
        
        long_qty = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0))
        short_qty = abs(self.positions.get(f"{self.base_ticker}_SHORT", 0.0))
        
        # Используем цену входа для расчета базы, если она есть, иначе текущую
        l_entry = long_entry_price if long_entry_price > 0 else spot_price
        s_entry = short_entry_price if short_entry_price > 0 else spot_price
        
        l_lev = targets["BASE_LONG"]["leverage"] if targets and "BASE_LONG" in targets else 5.0
        s_lev = targets["BASE_SHORT"]["leverage"] if targets and "BASE_SHORT" in targets else 5.0

        # Актуальная стоимость Long (Доля капитала + PnL)
        val_long = (long_qty * l_entry / l_lev) + (long_qty * (spot_price - l_entry))
        # Актуальная стоимость Short (Доля капитала + PnL)
        val_short = (short_qty * s_entry / s_lev) + (short_qty * (s_entry - spot_price))
        # Стоимость Виртуальной части
        val_virt = self.virt_current_value

        # Сохраняем для логирования (теперь сумма будет ~100%, и Short будет расти при падении цены)
        self.share_long_pct = round(val_long / self.tpv * 100, 1) if self.tpv > 0 else 0
        self.share_short_pct = round(val_short / self.tpv * 100, 1) if self.tpv > 0 else 0
        self.share_virt_pct = round(val_virt / self.tpv * 100, 1) if self.tpv > 0 else 0

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Ребалансировка портфеля. Если хоть одна нога превысила порог, пересчитываем всё.
        """
        actions = []
        shares = {
            "BASE_LONG": self.share_long_pct / 100,
            "BASE_SHORT": self.share_short_pct / 100,
            "VIRTUAL": self.share_virt_pct / 100
        }

        # Проверяем, превышен ли порог хотя бы одной ногой
        any_exceeded = False
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            if abs(shares[key] - targets[key]["share"]) > threshold:
                any_exceeded = True
                break

        if not any_exceeded:
            return []

        # Ребалансируем ВСЕ ноги
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share = targets[key]["share"]
            current_share = shares[key]
            diff_share = current_share - target_share # Положительно при ИЗБЫТКЕ

            if key == "VIRTUAL":
                actions.append({
                    "type": "VIRTUAL_RESET",
                    "symbol": "VIRTUAL",
                    "diff_usdt": diff_share * self.tpv,
                    "priority": 1 if diff_share > 0 else 3
                })
            else:
                cfg = targets[key]
                pos_key = f"{self.base_ticker}_LONG" if key == "BASE_LONG" else f"{self.base_ticker}_SHORT"

                # ПОРТФЕЛЬНАЯ ФОРМУЛА:
                # Чтобы изменить долю капитала на X%, нужно изменить НОМИНАЛ на (X% * Плечо)
                # Если у нас избыток доли (diff_share > 0), нам нужно ОТРИЦАТЕЛЬНОЕ изменение (продажа)
                diff_usdt = -diff_share * self.tpv * cfg["leverage"]

                if not ignore_limits:
                    max_change = self.tpv * 0.5 * cfg["leverage"]
                    if abs(diff_usdt) > max_change:
                        diff_usdt = math.copysign(max_change, diff_usdt)

                # Reduction (продажа излишка) если diff_usdt < 0
                is_reduction = diff_usdt < 0

                actions.append({
                    "type": "ORDER",
                    "symbol": pos_key,
                    "diff_usdt": diff_usdt,
                    "priority": 0 if is_reduction else 2
                })

        actions.sort(key=lambda x: x["priority"])
        return actions
