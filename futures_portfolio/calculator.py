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
        self.real_equity: float = real_equity
        
        # Виртуальная доля
        if virt_basis_price <= 0: virt_basis_price = spot_price
        price_change: float = spot_price / virt_basis_price
        self.virt_current_value: float = virt_allocated_usdt * price_change
        
        # Общий TPV (включая накопленный резерв)
        self.total_tpv: float = real_equity + (self.virt_current_value - virt_allocated_usdt) + self.siphoning_reserve

        # Логика TPV Cap (Siphoning) + Recovery Mode
        if self.total_tpv > self.initial_capital:
            # ПРАВИЛО: Активный TPV не может превышать initial_capital.
            # Излишек считается потенциальным резервом для сифонинга в main.py
            self.tpv: float = self.initial_capital
            # Обновляем reserve для соответствия total_tpv (для тестов и логов)
            self.siphoning_reserve = self.total_tpv - self.initial_capital
        else:
            # Если мы в просадке, используем SAFE для поддержания маржи (Recovery Mode)
            self.tpv: float = self.total_tpv

        
        # Защита от NaN
        if math.isnan(self.tpv) or self.tpv <= 0:
            self.tpv = 1e-9

        # Расчет стоимости позиций (Allocated Capital + PnL)
        
        long_qty: float = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0))
        short_qty: float = abs(self.positions.get(f"{self.base_ticker}_SHORT", 0.0))
        
        l_lev: float = targets["BASE_LONG"]["leverage"] if targets and "BASE_LONG" in targets else 5.0
        s_lev: float = targets["BASE_SHORT"]["leverage"] if targets and "BASE_SHORT" in targets else 5.0

        # Упрощенный расчет долей: (Номинал / Плечо) / TPV
        self.notional_long = long_qty * spot_price
        self.notional_short = short_qty * spot_price

        val_long: float = (self.notional_long / l_lev) if l_lev > 0 else 0.0
        val_short: float = (self.notional_short / s_lev) if s_lev > 0 else 0.0
        val_virt: float = self.virt_current_value

        # Сохраняем для логирования и проверок отклонений
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

        # Проверяем, превышен ли порог хотя бы одной ногой
        any_exceeded: bool = threshold < 0.0

        if not any_exceeded:
            for key in ["BASE_LONG", "BASE_SHORT", "VIRTUAL"]:
                if not math.isclose(shares[key], targets[key]["share"], abs_tol=threshold):
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
                    "base_symbol": "VIRTUAL",
                    "position_side": "BOTH",
                    "diff_usdt": diff_share * self.tpv,
                    "priority": 1 if diff_share > 0 else 3
                })
            else:
                cfg: Dict = targets[key]
                pos_side: str = "LONG" if key == "BASE_LONG" else "SHORT"
                pos_key: str = f"{self.base_ticker}_{pos_side}"

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
                    "base_symbol": self.base_ticker,
                    "position_side": pos_side,
                    "diff_usdt": diff_usdt,
                    "priority": 0 if is_reduction else 2
                })

        actions.sort(key=lambda x: x["priority"])
        return actions
