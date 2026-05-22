"""Исполнение ордеров на Binance Futures (Hedge Mode)."""
import logging
import asyncio
from typing import Dict, Optional, Tuple, Any, List, TYPE_CHECKING
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

if TYPE_CHECKING:
    from connector import BinanceConnector

logger = logging.getLogger(__name__)

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class PortfolioExecutor:
    def __init__(self, connector: 'BinanceConnector', base_ticker: str = "BTCUSDT", max_orders_per_second: int = 10):
        self.connector = connector
        self.base_ticker = base_ticker
        self.semaphore = asyncio.Semaphore(max_orders_per_second)

    def calculate_order_size(self, target_share: float, current_value: float, total_value: float, spot_price: float) -> Decimal:
        """Расчёт размера ордера в контрактах."""
        dec_target_share = Decimal(str(target_share))
        dec_current_value = Decimal(str(current_value))
        dec_total_value = Decimal(str(total_value))
        dec_spot_price = Decimal(str(spot_price))
        
        target_value = dec_target_share * dec_total_value
        delta_value = target_value - dec_current_value
        order_qty = delta_value / dec_spot_price
        return order_qty

    def round_quantity(self, qty: Decimal, step_size: Decimal) -> Decimal:
        """Precision-perfect rounding using exchange stepSize mask."""
        if step_size <= 0:
            return qty.quantize(Decimal('1.000'), rounding="ROUND_FLOOR").normalize()
        
        # Normalize step_size to avoid float representation artifacts
        mask = step_size.normalize()
        # Round DOWN to the nearest allowed step to ensure it fits within balance/limits
        return qty.quantize(mask, rounding="ROUND_FLOOR").normalize()

    async def execute_market_order(self, symbol: str, qty: Any, side: str, step_size: Any = Decimal('0'), reduce_only: bool = False, position_side: str = "BOTH", min_notional: Any = Decimal('6.0'), price: Any = Decimal('0')) -> Dict:
        """
        Executes an AGGRESSIVE LIMIT order (Mark Price +/- 1.0%) to simulate market execution
        with slippage protection.
        """
        qty = Decimal(str(qty))
        step_size = Decimal(str(step_size))
        min_notional = Decimal(str(min_notional))
        price = Decimal(str(price))

        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": "Округлилось до нуля"}

        try:
            # 1. Fetch current Mark Price for slippage anchor
            try:
                prices = await self.connector.get_mark_prices([symbol])
                mark_price = Decimal(str(prices.get(symbol, 0)))
            except Exception as e:
                logger.warning(f"Price fetch failed for {symbol}, using fallback: {e}")
                mark_price = Decimal('0')

            if mark_price <= 0:
                # Fallback to provided price if API fails
                mark_price = price if price > 0 else Decimal('0')
            
            if mark_price <= 0:
                raise Exception("Invalid mark price for slippage protection")

            # Dust Guard check
            if (abs(qty) * mark_price) < min_notional:
                msg = f"Order too small: {float(abs(qty) * mark_price):.2f} USDT < {float(min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}

            # 2. Apply 1.0% aggressive offset (Safe ceiling/floor)
            offset = mark_price * Decimal('0.01')
            exec_price = (mark_price + offset) if side == "BUY" else (mark_price - offset)
            
            # [SAFETY] Format values strictly for Binance API
            str_qty = "{:f}".format(abs(qty).normalize())
            str_price = "{:f}".format(exec_price.normalize())

            params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "GTC", # Fill as much as possible at aggressive price
                "quantity": str_qty,
                "price": str_price,
                "positionSide": position_side
            }
            
            result = await asyncio.to_thread(
                self.connector.futures_client.futures_create_order,
                **params
            )

            order_id = result.get("orderId")
            executed_qty = Decimal(str(result.get("executedQty", "0.0")))
            avg_price = Decimal(str(result.get("avgPrice", "0.0")))
            
            # [POLLING] Aggressive limit might take a few ms to fill
            if executed_qty == 0 and order_id:
                logger.info(f"Aggressive limit order {order_id} initial fill is 0. Polling status...")
                for _ in range(5):
                    await asyncio.sleep(0.3)
                    status = await self.connector.get_order_status(symbol, order_id)
                    executed_qty = Decimal(str(status.get("executedQty", "0.0")))
                    avg_price = Decimal(str(status.get("avgPrice", "0.0")))
                    if executed_qty > 0: break

            if avg_price == 0 and executed_qty > 0:
                avg_price = mark_price

            logger.info(f"Order EXECUTED on Binance: {side} {str_qty} {symbol} ({position_side}) | Fact Qty: {executed_qty}")

            if executed_qty == 0:
                return {"status": "ERROR", "message": f"Binance executed 0.0 contracts for order {order_id}", "result": result}

            # --- REAL TRADE DATA SYNC ---
            real_pnl = Decimal('0.0')
            real_commission = Decimal('0.0')
            try:
                await asyncio.sleep(0.5)
                trades = await self.connector.get_order_trades(symbol, order_id)
                for t in trades:
                    real_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                    real_commission += Decimal(str(t.get("commission", "0.0")))
                logger.info(f"Realized data for {symbol} Order {order_id}: PnL={float(real_pnl):+.4f}, Commission={float(real_commission):.4f}")
            except Exception as e:
                logger.warning(f"Could not fetch trade details for {symbol} {order_id}: {e}")

            return {
                "status": "SUCCESS",
                "result": result,
                "executed_qty": float(executed_qty),
                "qty": float(executed_qty),
                "avg_price": float(avg_price),
                "price": float(avg_price),
                "realized_pnl": float(real_pnl),
                "trade_pnl": float(real_pnl),
                "commission": float(real_commission)
            }
        except Exception as e:
            import traceback
            logger.error(f"Order FAILED for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {"status": "ERROR", "message": str(e) if str(e) else f"Unknown error of type {type(e).__name__}"}

    async def execute_limit_with_fallback(self, symbol: str, qty: Any, side: str,
                                           step_size: Any = Decimal('0'), reduce_only: bool = False,
                                           position_side: str = "BOTH",
                                           offset_pct: Any = Decimal('0.2'),
                                           timeout_sec: int = 30,
                                           min_notional: Any = Decimal('6.0'),
                                           price: Any = Decimal('0')) -> Dict:
        """
        Limit + Fallback with improved aggressive limit protection.
        """
        qty = Decimal(str(qty))
        step_size = Decimal(str(step_size))
        offset_pct = Decimal(str(offset_pct))
        min_notional = Decimal(str(min_notional))
        price = Decimal(str(price))

        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": "Округлилось до нуля"}

        try:
            order_book = await self.connector.get_order_book(symbol, limit=5)
            best_bid = Decimal(str(order_book["bids"][0][0]))
            best_ask = Decimal(str(order_book["asks"][0][0]))
            mid_price = (best_bid + best_ask) / 2

            if (abs(qty) * mid_price) < min_notional:
                msg = f"Limit order too small: {float(abs(qty) * mid_price):.2f} USDT < {float(min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}

            if side == "SELL":
                limit_price = mid_price * (1 + offset_pct / 100)
                if limit_price > best_ask: limit_price = best_ask
            else:
                limit_price = mid_price * (1 - offset_pct / 100)
                if limit_price < best_bid: limit_price = best_bid

            params = {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(limit_price),
                "position_side": position_side
            }
            
            order = await self.connector.place_limit_maker_order(**params)
            order_id = order["orderId"]

            filled_qty = Decimal('0.0')
            avg_fill_price = Decimal('0.0')

            for _ in range(timeout_sec):
                await asyncio.sleep(1)
                status = await self.connector.get_order_status(symbol, order_id)
                filled_qty = Decimal(str(status.get("executedQty", 0)))
                avg_fill_price = Decimal(str(status.get("avgPrice", 0))) if status.get("avgPrice") else avg_fill_price

                if status["status"] == "FILLED":
                    real_pnl = Decimal('0.0')
                    real_commission = Decimal('0.0')
                    try:
                        await asyncio.sleep(0.5)
                        trades = await self.connector.get_order_trades(symbol, order_id)
                        for t in trades:
                            real_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                            real_commission += Decimal(str(t.get("commission", "0.0")))
                    except: pass

                    return {
                        "status": "SUCCESS_LIMIT",
                        "executed_qty": float(filled_qty),
                        "qty": float(filled_qty),
                        "avg_price": float(avg_fill_price),
                        "price": float(avg_fill_price),
                        "realized_pnl": float(real_pnl),
                        "trade_pnl": float(real_pnl),
                        "commission": float(real_commission),
                        "order_type": "LIMIT_MAKER"
                    }

            logger.warning(f"Limit order timeout ({timeout_sec}s). Cancelling and fallback to aggressive market limit.")
            await self.connector.cancel_order(symbol, order_id)

            remaining_qty = qty - filled_qty
            market_pnl = Decimal('0.0')
            market_comm = Decimal('0.0')
            total_executed = filled_qty
            
            if remaining_qty > 0:
                # Use AGGRESSIVE LIMIT for fallback instead of blind MARKET
                market_result = await self.execute_market_order(
                    symbol=symbol, qty=remaining_qty, side=side, step_size=Decimal('0'),
                    position_side=position_side, min_notional=min_notional, price=mid_price
                )
                if market_result["status"] == "SUCCESS":
                    total_executed += Decimal(str(market_result["executed_qty"]))
                    market_pnl = Decimal(str(market_result["realized_pnl"]))
                    market_comm = Decimal(str(market_result["commission"]))

            limit_pnl = Decimal('0.0')
            limit_comm = Decimal('0.0')
            if filled_qty > 0:
                try:
                    await asyncio.sleep(0.3)
                    trades = await self.connector.get_order_trades(symbol, order_id)
                    for t in trades:
                        limit_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                        limit_comm += Decimal(str(t.get("commission", "0.0")))
                except: pass

            return {
                "status": "SUCCESS_FALLBACK",
                "executed_qty": float(total_executed),
                "qty": float(total_executed),
                "realized_pnl": float(limit_pnl + market_pnl),
                "trade_pnl": float(limit_pnl + market_pnl),
                "commission": float(limit_comm + market_comm),
                "price": float(price)
            }

        except Exception as e:
            logger.error(f"Limit+Fallback FAILED: {e}")
            return await self.execute_market_order(
                symbol=symbol, qty=qty, side=side, step_size=step_size,
                reduce_only=reduce_only, position_side=position_side,
                min_notional=min_notional, price=price
            )

    async def execute_actions(self, actions: List[Dict[str, Any]], price: float, paper_mode: bool = True, portfolio_cfg: Optional[Dict[str, Any]] = None, step_sizes: Optional[Dict[str, float]] = None, paper_state: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Executes actions concurrently. Reductions before expansions.
        """
        if not actions: return []
        
        reductions = [a for a in actions if a.get("is_reduction")]
        expansions = [a for a in actions if not a.get("is_reduction")]
        results_map = {}

        for group in [reductions, expansions]:
            if not group: continue
            group_action_ids = [id(a) for a in group]
            tasks = [
                self._execute_single_action(action, price, paper_mode, portfolio_cfg or {}, step_sizes or {}, paper_state)
                for action in group
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            for action_id, r in zip(group_action_ids, group_results):
                results_map[action_id] = r if not isinstance(r, Exception) else {"status": "ERROR", "message": str(r)}

        return [results_map[id(a)] for a in actions]

    async def _execute_single_action(self, action: Dict[str, Any], price: float, paper_mode: bool, portfolio_cfg: Dict[str, Any], step_sizes: Dict[str, float], paper_state: Optional[Dict] = None) -> Dict[str, Any]:
        if action["type"] == "VIRTUAL_ORDER":
            return {"type": "VIRTUAL_ORDER", "status": "SUCCESS", "diff_usdt": action["diff_usdt"]}

        symbol = action["symbol"]
        base_symbol = action.get("base_symbol", symbol.split('_')[0])
        pos_side = action.get("position_side", "LONG")
        diff_usdt = Decimal(str(action["diff_usdt"]))
        dec_price = Decimal(str(price))
        side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
        order_qty = abs(diff_usdt / dec_price)
        reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")
        step_size = Decimal(str(step_sizes.get(base_symbol, 0.0)))
        min_notional = Decimal(str(portfolio_cfg.get("min_notional_usdt", 7.0)))

        if paper_mode:
            qty_rounded = self.round_quantity(order_qty, step_size)
            if qty_rounded <= 0: return {"status": "SKIPPED", "message": "Rounded to zero"}
            if (qty_rounded * dec_price) < min_notional: return {"status": "SKIPPED", "message": "Too small"}
            
            # Simulated Profit/Loss logic
            trade_pnl = Decimal('0.0')
            if reduce_only and paper_state:
                old_entry = Decimal(str(paper_state.get("long_entry_price" if pos_side == "LONG" else "short_entry_price", price)))
                trade_pnl = qty_rounded * (dec_price - old_entry) if pos_side == "LONG" else qty_rounded * (old_entry - dec_price)
            
            logger.info(f"PAPER ORDER EXECUTED: {side} {float(qty_rounded):.6g} {base_symbol} ({pos_side}) at {float(dec_price):.6g} | PnL: {float(trade_pnl):+.4f}$")

            return {
                "status": "SUCCESS", "type": pos_side, "side": side, "qty": float(qty_rounded), "executed_qty": float(qty_rounded),
                "price": float(dec_price), "trade_pnl": float(trade_pnl), "commission": float(qty_rounded * dec_price * Decimal('0.0004')),
                "reduce_only": reduce_only
            }
        else:
            async with self.semaphore:
                res = await self.execute_market_order(
                    symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
                    reduce_only=reduce_only, position_side=pos_side, min_notional=min_notional, price=dec_price
                )
                if res["status"] in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]:
                    res["type"] = pos_side
                    res["reduce_only"] = reduce_only
                return res
