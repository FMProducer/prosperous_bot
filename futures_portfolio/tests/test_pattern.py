import asyncio
import aiohttp
import functools

def retry_on_network_error(retries=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        print(f"Max retries reached for {func.__name__}: {e}")
            return None
        return wrapper
    return decorator

class TestScanner:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.semaphore = asyncio.Semaphore(15)
    
    @retry_on_network_error(retries=3)
    async def fetch(self, session, endpoint, params=None):
        async with self.semaphore:
            try:
                async with session.get(f"{self.base_url}{endpoint}", params=params, timeout=20, proxy=None) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)
                        return await self.fetch(session, endpoint, params)
                    if response.status != 200:
                        return None
                    return await response.json()
            except Exception as e:
                return None

async def main():
    scanner = TestScanner()
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        print("Fetching ticker/24hr...")
        r1 = await scanner.fetch(session, "/fapi/v1/ticker/24hr")
        print(f"Result: {type(r1).__name__}, {'len=' + str(len(r1)) if r1 else 'None'}")
        
        print("Fetching premiumIndex...")
        r2 = await scanner.fetch(session, "/fapi/v1/premiumIndex")
        print(f"Result: {type(r2).__name__}, {'len=' + str(len(r2)) if r2 else 'None'}")

asyncio.run(main())
