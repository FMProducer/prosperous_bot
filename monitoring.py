
from fastapi import FastAPI
from adaptive_agent.portfolio_manager import PortfolioManager
from adaptive_agent.rebalance_engine import RebalanceEngine

app = FastAPI()
portfolio = None
engine = None

TARGET_WEIGHTS = {
    "BTC_SPOT": 0.65,
    "BTC_SHORT5X": 0.24,
    "BTC_LONG5X": 0.11
}

@app.on_event("startup")
async def startup():
    global portfolio, engine
    portfolio = PortfolioManager(spot_api=..., fut_api=...)
    engine = RebalanceEngine(portfolio, threshold_pct=0.01)

@app.get("/metrics")
async def metrics(p_spot: float):
    values = await portfolio.get_value_distribution_usdt(p_spot, p_contract=None)
    positions = await portfolio.get_positions_with_margin()
    try:
        pos = next(p for p in positions if p.contract == "BTC_USDT" and abs(float(p.size)) >= 1)
        p_contract = float(pos.margin) / abs(float(pos.size))
    except StopIteration:
        p_contract = p_spot

    total = sum(values.values())
    weights = {k: round(v / total, 6) for k, v in values.items()}
    deltas = {k: round(weights[k] - TARGET_WEIGHTS[k], 6) for k in weights}
    rebalance_required = any(abs(d) > engine.threshold_pct for d in deltas.values())

    return {
        "weights": weights,
        "deltas": deltas,
        "rebalance_required": rebalance_required
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
