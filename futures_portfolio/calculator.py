import logging
from typing import Dict, List
import math

logger = logging.getLogger(__name__)

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_price: float, real_equity: float, 
                 virt_basis_price: float, virt_allocated_usdt: float, 
                 long_entry_price: float = 0.0, short_entry_price: float = 0.0,
                 base_ticker: str = "BTCUSDT", siphoning_reserve: float = 0.0,
                 targets: Dict[str, Dict] = None, initial_capital: float = 10000.0) -> None:
        self.positions: Dict[str, float] = positions
        self.price: float = spot_price
        self.base_ticker: str = base_ticker
        self.siphoning_reserve: float = siphoning_reserve
        self.initial_capital: float = initial_capital
        
        # Виртуальная доля
        if virt_basis_price <= 0: virt_basis_price = spot_price
        price_change: float = spot_price / virt_basis_price
        self.virt_current_value: float = virt_allocated_usdt * price_change
        
        # Общий TPV (включая накопленный резерв)
        self.total_tpv: float = real_equity + (self.virt_current_value - virt_allocated_usdt) + self.siphoning_reserve
        
        # Логика TPV Cap (Siphoning) + Recovery Mode
        # ПРАВИЛО: Активный TPV не может превышать initial_capital.
        # Если total_tpv > initial_capital, излишек уходит в reserve.
        # Если total_tpv < initial_capital, reserve используется для поддержания маржи (Recovery Mode).

        if self.total_tpv > self.initial_capital:
            self.tpv = self.initial_capital
            self.siphoning_reserve = self.total_tpv - self.initial_capital
        else:
            self.tpv = self.total_tpv
            self.siphoning_reserve = 0.0 # Все ушло на поддержку маржи
        
        # Защита от NaN
        if math.isnan(self.tpv) or self.tpv <= 0:
            self.tpv = 1e-9

        # Расчет стоимости позиций (Allocated Capital + PnL)
        
        long_qty: float = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0))
        short_qty: float = abs(self.positions.get(f"{self.base_ticker}_SHORT", 0.0))
        
        # Используем цену входа для расчета базы, если она есть, иначе текущую
        l_entry: float = long_entry_price if long_entry_price > 0 else spot_price
        s_entry: float = short_entry_price if short_entry_price > 0 else spot_price
        
        l_lev: float = targets["BASE_LONG"]["leverage"] if targets and "BASE_LONG" in targets else 5.0
        s_lev: float = targets["BASE_SHORT"]["leverage"] if targets and "BASE_SHORT" in targets else 5.0

        # Актуальная стоимость Long (Доля капитала + PnL)
        val_long: float = (long_qty * l_entry / l_lev) + (long_qty * (spot_price - l_entry))
        # Актуальная стоимость Short (Доля капитала + PnL)
        val_short: float = (short_qty * s_entry / s_lev) + (short_qty * (s_entry - spot_price))
        # Стоимость Виртуальной части
        val_virt: float = self.virt_current_value

        # Сохраняем для логирования
        self.long_entry_price = long_entry_price
        self.short_entry_price = short_entry_price
        self.share_long_pct: float = round(val_long / self.tpv * 100, 1) if self.tpv > 0 else 0.0
        self.share_short_pct: float = round(val_short / self.tpv * 100, 1) if self.tpv > 0 else 0.0
        self.share_virt_pct: float = round(val_virt / self.tpv * 100, 1) if self.tpv > 0 else 0.0

    def calculate_deviations(self, targets: Dict[str, Dict], threshold: float, ignore_limits: bool = False) -> List[Dict]:
        """
        Ребалансировка портфеля. Если хоть одна нога превысила порог, пересчитываем всё.
        """
        actions: List[Dict] = []
        shares: Dict[str, float] = {
            "BASE_LONG": self.share_long_pct / 100,
            "BASE_SHORT": self.share_short_pct / 100,
            "VIRTUAL": self.share_virt_pct / 100
        }

        # Проверяем порог. Если threshold < 0 (force), сразу any_exceeded = True
        any_exceeded: bool = threshold < 0.0

        if not any_exceeded:
            for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
                if not math.isclose(shares[key], targets[key]["share"], abs_tol=max(0.0, threshold)):
                    any_exceeded = True
                    break

        if not any_exceeded:
            return []

        # Ребалансируем ВСЕ ноги
        for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
            target_share: float = targets[key]["share"]
            current_share: float = shares[key]
            diff_share: float = current_share - target_share # Положительно при ИЗБЫТКЕ

            if key == "VIRTUAL":
                actions.append({
                    "type": "VIRTUAL_RESET",
                    "symbol": "VIRTUAL",
                    "diff_usdt": diff_share * self.tpv,
                    "priority": 1 if diff_share > 0 else 3
                })
            else:
                cfg: Dict = targets[key]
                pos_key: str = f"{self.base_ticker}_LONG" if key == "BASE_LONG" else f"{self.base_ticker}_SHORT"

                # ПОРТФЕЛЬНАЯ ФОРМУЛА:
                # Чтобы изменить долю капитала на X%, нужно изменить НОМИНАЛ на (X% * Плечо)
                # Если у нас избыток доли (diff_share > 0), нам нужно ОТРИЦАТЕЛЬНОЕ изменение (продажа)
                diff_usdt: float = -diff_share * self.tpv * cfg["leverage"]

                if not ignore_limits:
                    max_change: float = self.tpv * 0.5 * cfg["leverage"]
                    if abs(diff_usdt) > max_change:
                        diff_usdt = math.copysign(max_change, diff_usdt)

                # Reduction (продажа излишка) если diff_usdt < 0
                is_reduction: bool = diff_usdt < 0

                actions.append({
                    "type": "ORDER",
                    "symbol": pos_key,
                    "diff_usdt": diff_usdt,
                    "priority": 0 if is_reduction else 2
                })

        actions.sort(key=lambda x: x["priority"])
        return actions
