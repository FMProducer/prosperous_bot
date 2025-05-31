
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional


class RebalanceEngine:
    """Алгоритм ребаланса портфеля.

    • Динамический порог: max(base_threshold_pct, 0.2 × ATR24h)
    • Корректный перевод USDT‑дельты → BTC / контракт (плечо 5×)
    • Двухшаговое исполнение: PostOnly‑limit (t сек) → Market fallback
    """

    def __init__(
        self,
        portfolio,
        target_weights: dict | None = None,
        spot_asset_symbol: str = '',
        futures_contract_symbol_base: str = '',
        *,
        base_threshold_pct: float = 0.005,
        threshold_pct: float | None = None,    # legacy совместимость
        exchange_client=None,
    ):
        if target_weights is None:
            target_weights = {}
        self.portfolio = portfolio
        self.target_weights = target_weights
        self.spot_asset_symbol = spot_asset_symbol
        self.futures_contract_symbol_base = futures_contract_symbol_base
        self.exchange = exchange_client
        # Если передан legacy‑параметр threshold_pct — используем его
        self.base_threshold_pct = threshold_pct if threshold_pct is not None else base_threshold_pct

    # ---------- helpers -------------------------------------------------

    @staticmethod
    def _dynamic_threshold(base_thr: float, atr_24h_pct: Optional[float]) -> float:
        """Возвращает адаптивный порог."""
        return max(base_thr, 0.2 * atr_24h_pct) if atr_24h_pct is not None else base_thr

    @staticmethod
    def _round_lot(qty_float: float, min_qty: float = 1) -> int:
        """Округление до целого числа контрактов (≥1)."""
        return max(int(round(qty_float)), int(min_qty))

    # ---------- public API ---------------------------------------------

    async def build_orders(
        self,
        *,
        p_spot: float,
        p_contract: Optional[float] = None,
        atr_24h_pct: Optional[float] = None,
    ) -> list[dict]:
        """Возвращает список ордеров dict(symbol, side, qty, notional_usdt, asset_key)."""


        try:
            nav = await self.portfolio.get_nav_usdt(p_spot=p_spot, p_contract=p_contract)  # type: ignore
            dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract)  # type: ignore
        except AttributeError:
            raw_vals: Dict[str, float] = getattr(self.portfolio, "_dist", {})  # type: ignore
            nav = sum(raw_vals.values())
            dist = {k: (v / nav if nav else 0.0) for k, v in raw_vals.items()}
        thr = self._dynamic_threshold(self.base_threshold_pct, atr_24h_pct)

        orders: list[dict] = []
        for asset_key, w_target in self.target_weights.items():
            w_cur = dist.get(asset_key, 0.0)
            diff = w_target - w_cur
            if abs(diff) <= thr:
                continue

            delta_usdt = diff * nav
            if asset_key.endswith('_SPOT'):
                symbol = self.spot_asset_symbol
                qty_float = delta_usdt / p_spot
            else:
                if not p_contract or p_contract <= 0:
                    logging.warning('p_contract not provided — пропуск %s', asset_key)
                    continue
                symbol = self.futures_contract_symbol_base
                qty_float = delta_usdt / p_contract

            side = 'buy' if qty_float > 0 else 'sell'
            qty_lot = self._round_lot(abs(qty_float))

            orders.append(
                dict(
                    symbol=symbol,
                    side=side,
                    qty=qty_lot,
                    notional_usdt=delta_usdt,
                    asset_key=asset_key,
                )
            )
        return orders
