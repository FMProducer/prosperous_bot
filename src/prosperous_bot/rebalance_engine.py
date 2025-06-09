import asyncio
import copy
from .logging_config import configure_root # This will be adjusted by hand later if patch fails
configure_root()
import logging
from datetime import datetime, timedelta
from typing import Optional

# ── Helper: substitute {main_asset_symbol} recursively ──────────────
def _subst_symbol(obj, sym):
    if isinstance(obj, dict):
        return { _subst_symbol(k, sym): _subst_symbol(v, sym) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _subst_symbol(x, sym) for x in obj ]
    if isinstance(obj, str) and "{main_asset_symbol}" in obj:
        return obj.replace("{main_asset_symbol}", sym)
    return obj

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
        target_weights: dict | None = None,
        spot_asset_symbol: str = "",
        futures_contract_symbol_base: str = "",
        *,
        base_threshold_pct: float = 0.005,
        threshold_pct: float | None = None,
        exchange_client=None,
        params: Optional[dict] = None, # Added params
    ):
        if target_weights is None: # Keep this for backward compatibility if needed
            target_weights = {}

        # Process params with _subst_symbol first
        if params is None:
            # Fallback or error if params is critical and not provided
            # For now, let's assume it might be optional or handled later if None
            # Or, initialize a default minimal structure if appropriate
            _params = {}
            logging.warning("RebalanceEngine initialized without 'params'. Using default empty dict.")
        else:
            sym = params.get("main_asset_symbol", "BTC").upper()
            _params = _subst_symbol(copy.deepcopy(params), sym)

        self.params = _params # Store processed params

        # ── Debounce: минимальный интервал между попытками ребаланса ─────
        self.min_rebalance_interval_minutes: int = self.params.get(
            "min_rebalance_interval_minutes", 0
        )
        self._last_rebalance_attempt_ts: Optional[datetime] = None

        # Initialize attributes from processed self.params
        # Ensure default values are applied if keys are missing from self.params
        self.portfolio = portfolio # portfolio is passed directly, not from params
        self.exchange = exchange_client # exchange_client is passed directly

        # Target weights can come from params or direct argument
        # Direct argument target_weights takes precedence if provided for flexibility
        self.target_weights = target_weights if target_weights else self.params.get('target_weights_normal', {})
        if not self.target_weights: # Further fallback or error for target_weights
            self.target_weights = self.params.get('target_weights', {}) # Legacy fallback
            if self.target_weights:
                logging.warning("Using legacy 'target_weights' from params for RebalanceEngine.")
            else:
                # Decide if this is a critical error or if empty target_weights is acceptable
                logging.warning("RebalanceEngine: 'target_weights_normal' (or 'target_weights') not found in params or direct args. Using empty target_weights.")
                self.target_weights = {}


        # Spot and futures symbols from params, with fallbacks if needed
        self.spot_asset_symbol = self.params.get('spot_asset_symbol', spot_asset_symbol if spot_asset_symbol else "BTCUSDT") # Example default
        self.futures_contract_symbol_base = self.params.get('futures_contract_symbol_base',
            futures_contract_symbol_base if futures_contract_symbol_base else "BTCUSDT_PERP") # Example default

        # Threshold from params or direct argument
        # Direct threshold_pct (legacy) or base_threshold_pct argument takes precedence
        if threshold_pct is not None:
            self.base_threshold_pct = threshold_pct
            logging.info(f"RebalanceEngine: Using direct legacy 'threshold_pct': {threshold_pct}")
        elif base_threshold_pct is not None and 'base_threshold_pct' not in self.params: # direct base_threshold_pct but not in params
            self.base_threshold_pct = base_threshold_pct
            logging.info(f"RebalanceEngine: Using direct 'base_threshold_pct': {base_threshold_pct}")
        else: # Fallback to params or the default of the direct argument if not in params
            self.base_threshold_pct = self.params.get('rebalance_threshold', base_threshold_pct)
            logging.info(f"RebalanceEngine: Using 'rebalance_threshold' from params or default: {self.base_threshold_pct}")


        # ---------- helpers -------------------------------------------------

    @staticmethod
    def _dynamic_threshold(base_thr: float, atr_24h_pct: Optional[float]) -> float:
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
        p_contract: Optional[float] = None,
        atr_24h_pct: Optional[float] = None,
    ) -> list[dict]:
        """ # noqa: D205
        Формирует список словарей-ордеров:
          {symbol, side, qty, notional_usdt, asset_key}
        """
        # --- дебаунс -----------------------------------------------------
        now_ts = datetime.utcnow()
        if (
            self.min_rebalance_interval_minutes > 0
            and self._last_rebalance_attempt_ts is not None
            and (now_ts - self._last_rebalance_attempt_ts)
            < timedelta(minutes=self.min_rebalance_interval_minutes)
        ):
            return []  # слишком рано; пропускаем попытку

        # записываем момент последней успешной проверки
        self._last_rebalance_attempt_ts = now_ts

        leverage = self.params.get("futures_leverage", 5.0)
        effective_leverage = leverage if leverage > 0 else 1e-9
        p_contract_adjusted = p_contract * effective_leverage if p_contract else None
        if hasattr(self.portfolio, "get_nav_usdt"):
            nav = await self.portfolio.get_nav_usdt(p_spot=p_spot, p_contract=p_contract)
        else:  # stub-портфели в tests
            try:
                dist_abs = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract_adjusted, leverage=leverage)
            except TypeError:
                dist_abs = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract_adjusted)
                nav = sum(dist_abs.values())
        try:
            dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract_adjusted, leverage=leverage)
        except TypeError:
            dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract_adjusted)
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
            # пропускаем ордера меньше заданного порога
            min_ord = self.params.get("min_order_notional_usdt", 10.0)
            if abs(delta_usdt) < min_ord:
                continue

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