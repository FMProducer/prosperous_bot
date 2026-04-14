"""Расчёт стоимости портфеля, долей и проверка отклонений."""
from typing import Dict, List

class PortfolioCalculator:
    def __init__(self, positions: Dict[str, float], spot_prices: Dict[str, float], free_balance: float):
        self.positions = positions
        self.spot_prices = spot_prices
        self.free_balance = free_balance

    def calculate_position_value(self, symbol: str) -> float:
        """Стоимость позиции в USDT: qty * spot_price."""
        qty = self.positions.get(symbol, 0.0)
        price = self.spot_prices.get(symbol, 0.0)
        return qty * price

    def total_portfolio_value(self) -> float:
        """Общая стоимость портфеля + свободный баланс."""
        position_values = sum(
            self.calculate_position_value(sym) for sym in self.positions.keys()
        )
        return position_values + self.free_balance

    def current_shares(self) -> Dict[str, float]:
        """Текущие доли в процентах."""
        total_value = self.total_portfolio_value()
        shares = {}
        for symbol in self.positions.keys():
            if total_value > 0:
                shares[symbol] = self.calculate_position_value(symbol) / total_value
        return shares

    def calculate_deviations(self, targets: Dict[str, float], threshold: float) -> List[Dict]:
        """
        Проверка отклонений от целевых долей.
        Возвращает список словарей с полями:
        symbol, deviation, direction, current_value, target_value
        """
        total_value = self.total_portfolio_value()
        current_shares = self.current_shares()
        deviations = []

        for symbol, target_share in targets.items():
            current_share = current_shares.get(symbol, 0.0)
            deviation = abs(current_share - target_share)
            if deviation > threshold:
                direction = "EXCESS" if current_share > target_share else "DEFICIT"
                deviations.append({
                    "symbol": symbol,
                    "deviation": deviation,
                    "direction": direction,
                    "current_share": current_share,
                    "target_share": target_share,
                    "current_value": self.calculate_position_value(symbol),
                    "target_value": target_share * total_value,
                })
        return deviations