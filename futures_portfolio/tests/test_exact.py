import asyncio
import aiohttp

async def main():
    base_url = "https://fapi.binance.com"
    
    semaphore = asyncio.Semaphore(20)
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with semaphore:
            try:
                async with session.get(f"{base_url}/fapi/v1/ticker/24hr", timeout=20, proxy=None) as response:
                    print(f"ticker status: {response.status}")
                    if response.status == 429:
                        print(f"Retry-After: {response.headers.get('Retry-After', 'N/A')}")
                    elif response.status == 200:
                        data = await response.json()
                        print(f"ticker count: {len(data)}")
                    else:
                        body = await response.text()
                        print(f"Response body: {body[:200]}")
            except Exception as e:
                print(f"ticker exception: {type(e).__name__}: {e}")
        
        async with semaphore:
            try:
                async with session.get(f"{base_url}/fapi/v1/premiumIndex", timeout=20, proxy=None) as response:
                    print(f"premium status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"premium count: {len(data)}")
                    else:
                        body = await response.text()
                        print(f"Response body: {body[:200]}")
            except Exception as e:
                print(f"premium exception: {type(e).__name__}: {e}")

asyncio.run(main())
