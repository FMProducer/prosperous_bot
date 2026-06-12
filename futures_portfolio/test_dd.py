import asyncio, json, os
from backtest_rebalance import run_backtest

async def test():
    cfg = json.load(open('config.json'))
    
    for dd_limit in [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]:
        cfg['max_drawdown_limit'] = dd_limit
        tmp = f'_dd_{dd_limit}.json'
        with open(tmp, 'w') as f:
            json.dump(cfg, f)
        
        result = await run_backtest(tmp, 'data', ticker_override='HMSTRUSDT', days=1, quiet=True)
        os.unlink(tmp)
        
        print(f'max_drawdown_limit={dd_limit:>5.1f}%: profit={result["profit_pct"]:>10.2f}%  max_dd={result["max_dd_pct"]:>6.2f}%  stops={result["trailing_stops"]:>3d}  sortino={result.get("sortino_ratio", 0):.4f}')

asyncio.run(test())
