import asyncio
import aiohttp

async def main():
    base_url = "https://fapi.binance.com"
    
    # Import the actual fetch logic from rank_tickers to test
    import sys, os
    sys.path.insert(0, r'C:\Python\Prosperous_Bot\futures_portfolio')
    
    # Create a test scanner to use its exact fetch method
    from rank_tickers import TickerScanner
    
    scanner = TickerScanner(concurrent_requests=20, rebalance_threshold=0.01, scanner_period_days=0.125)
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        print("Testing ticker/24hr...")
        r1 = await scanner.fetch(session, "/fapi/v1/ticker/24hr")
        print(f"  Result: {type(r1)}, len={len(r1) if r1 else 'None'}")
        
        print("Testing premiumIndex...")
        r2 = await scanner.fetch(session, "/fapi/v1/premiumIndex")
        print(f"  Result: {type(r2)}, len={len(r2) if r2 else 'None'}")
        
        if r1 and r2:
            print("SUCCESS - both endpoints returned data")
        else:
            print("FAILURE - one or both returned None")

asyncio.run(main())
