import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.path.join(os.getcwd(), "futures_portfolio"))

async def test():
    from aggregator import StatusAggregator
    agg = StatusAggregator(config_path="futures_portfolio/config.json")
    print("Aggregator instantiated")
    await agg.init_services()
    print("Services initialized")
    await agg.close_services()
    print("Services closed")

if __name__ == "__main__":
    asyncio.run(test())
