import asyncio
import logging
from datetime import datetime, timedelta


class RebalanceEngine:
    """
    Алгоритм ребаланса портфеля.
    Динамический порог: max(base_threshold_pct, 0.2 * ATR24h)
    Корректный перевод USDT-дельты -> BTC/контракты (плечо 5)
    Двухшаговое исполнение: PostOnly-limit (t seconds) -> Market fallback
    """

    def __init__(
        self,
        portfolio,
        target_weights: dict,
        spot_asset_symbol: str,
        futures_contract_symbol_base: str,
        *,
        base_threshold_pct: float = 0.005,
        exchange_client=None,
    ):
        self.portfolio = portfolio
        self.target_weights = target_weights
        self.spot_asset_symbol = spot_asset_symbol
        self.futures_contract_symbol_base = futures_contract_symbol_base
        self.base_threshold_pct = base_threshold_pct
        self.exchange = exchange_client

    # ---------- helpers -------------------------------------------------

    @staticmethod
    def _dynamic_threshold(base_thr: float, atr_24h_pct: float | None) -> float:
        """Возвращает адаптивный порог."""
        return max(base_thr, 0.2 * atr_24h_pct) if atr_24h_pct is not None else base_thr

    @staticmethod
    def _round_lot(qty_float: float, min_qty: float = 1) -> int:
        """Округление до целого числа контрактов (>= 1)."""
        return max(int(round(qty_float)), int(min_qty))

    # ---------- public API ---------------------------------------------

    async def build_orders(
        self,
        *,
        p_spot: float,
        p_contract: float | None = None,
        atr_24h_pct: float | None = None,
    ) -> list[dict]:
        """
        Формирует список словарей-ордеров:
          {symbol, side, qty, notional_usdt, asset_key}
        """
        nav = await self.portfolio.get_nav_usdt(p_spot=p_spot, p_contract=p_contract)
        dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract)
        thr = self._dynamic_threshold(self.base_threshold_pct, atr_24h_pct)

        orders = []
        for asset_key, w_target in self.target_weights.items():
            w_cur = dist.get(asset_key, 0.0)
            diff = w_target - w_cur
            if abs(diff) <= thr:
                continue

            delta_usdt = diff * nav
            if asset_key.endswith("_SPOT"):
                symbol = self.spot_asset_symbol
                qty_float = delta_usdt / p_spot
            else:
                if p_contract is None or p_contract <= 0:
                    logging.warning("p_contract not provided — пропуск %s", asset_key)
                    continue
                symbol = self.futures_contract_symbol_base
                qty_float = delta_usdt / p_contract

            side = "buy" if qty_float > 0 else "sell"
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

    async def execute(
        self,
        *,
        orders: list[dict],
        timeout_sec: int = 5,
        post_only: bool = True,
    ) -> list[dict]:
        """
        Исполняет ордера через exchange_client:
        PostOnly-limit -> ожидание timeout_sec -> Market fallback.
        Возвращает список exec_log.
        """
        if self.exchange is None:
            raise RuntimeError("exchange_client must be passed to RebalanceEngine")

        exec_log = []
        for o in orders:
            status, price_exec, commission = "failed", None, 0.0
            symbol, side, qty = o["symbol"], o["side"], o["qty"]
            try:
                if post_only:
                    ord_obj = await self.exchange.post_only_limit(symbol, side, qty)
                    t0 = datetime.utcnow()
                    while not ord_obj.filled and datetime.utcnow() - t0 < timedelta(seconds=timeout_sec):
                        await asyncio.sleep(0.5)
                        ord_obj = await self.exchange.get_order(ord_obj.id)
                    if ord_obj.filled:
                        price_exec, commission, status = ord_obj.price, ord_obj.commission, "filled_limit"
                    else:
                        await self.exchange.cancel_order(ord_obj.id)

                if status != "filled_limit":  # fallback
                    ord_obj = await self.exchange.market_order(symbol, side, qty)
                    price_exec, commission, status = ord_obj.price, ord_obj.commission, "filled_market"

                await self.portfolio.apply_execution(
                    symbol=symbol, side=side, qty=qty, price=price_exec, commission=commission
                )
            except Exception as exc:
                logging.exception("EXEC ERROR %s %s %s: %s", side, qty, symbol, exc)
                status = "error"

            exec_log.append(
                dict(symbol=symbol, side=side, qty=qty, price_exec=price_exec, commission=commission, status=status)
            )
        return exec_log
