
import asyncio

class PortfolioManager:
    """Мини‑обёртка над spot_api и futures_api, используемая в тестах."""
    def __init__(self, spot_api, futures_api):
        self.spot_api = spot_api
        self.futures_api = futures_api

    async def get_value_distribution_usdt(self, *, p_spot: float, p_contract: float | None = None):
        # поддерживаем как sync‑, так и async‑моки
        accs = self.spot_api.spot.get_account_detail()
        accounts = await accs if asyncio.iscoroutine(accs) else accs

        spot_qty = 0.0
        for acc in accounts:
            if getattr(acc, 'currency', None) == 'BTC':
                spot_qty += float(acc.available)
        spot_val = spot_qty * p_spot

        pos_resp = self.futures_api.futures.list_positions()
        positions = await pos_resp if asyncio.iscoroutine(pos_resp) else pos_resp
        long_val = short_val = 0.0
        for pos in positions:
            size = float(pos.size)
            margin = float(getattr(pos, 'margin', 0.0))
            if size > 0:
                long_val += margin
            elif size < 0:
                short_val += margin

        total = spot_val + long_val + short_val or 1.0
        return {
            "BTC_SPOT": spot_val / total,
            "BTC_LONG5X": long_val / total,
            "BTC_SHORT5X": short_val / total,
        }

    # sync‑proxy для property‑тестов
    def get_value_distribution_sync(self, p_spot: float, p_contract: float):
        return asyncio.run(self.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract))
