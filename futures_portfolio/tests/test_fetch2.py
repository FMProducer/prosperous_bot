import asyncio
import aiohttp
import functools

async def main():
    base_url = "https://fapi.binance.com"
    semaphore = asyncio.Semaphore(15)
    
    # Exact same fetch logic as rank_tickers
    async def fetch(session, endpoint, params=None):
        async with semaphore:
            try:
                # This is the exact call from rank_tickers
                async with session.get(f"{base_url}{endpoint}", params=params, timeout=20, proxy=None) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)
                        return await fetch(session, endpoint, params)
                    if response.status != 200:
                        print(f"  Non-200 status: {response.status} for {endpoint}")
                        return None
                    return await response.json()
            except Exception as e:
                print(f"  Exception in fetch: {type(e).__name__}: {e}")
                return None
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        # Test basic fetch
        print("Test 1: Basic fetch (same as rank_tickers)")
        r1 = await fetch(session, "/fapi/v1/ticker/24hr")
        print(f"  Result: {type(r1).__name__}, {'len=' + str(len(r1)) if r1 else 'None'}")
        
        r2 = await fetch(session, "/fapi/v1/premiumIndex")
        print(f"  Result: {type(r2).__name__}, {'len=' + str(len(r2)) if r2 else 'None'}")

asyncio.run(main())
