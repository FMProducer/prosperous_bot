import asyncio, aiohttp, traceback

async def test():
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                async with session.get('https://fapi.binance.com/fapi/v1/ticker/24hr', timeout=20, proxy=None) as resp:
                    print('ticker status:', resp.status)
                    data = await resp.json()
                    print('ticker count:', len(data))
            except Exception as e:
                print('ticker error:', e)
                traceback.print_exc()
            
            try:
                async with session.get('https://fapi.binance.com/fapi/v1/premiumIndex', timeout=20, proxy=None) as resp:
                    print('premium status:', resp.status)
                    data = await resp.json()
                    print('premium count:', len(data))
            except Exception as e:
                print('premium error:', e)
                traceback.print_exc()
    except Exception as e:
        print('session error:', e)
        traceback.print_exc()

asyncio.run(test())
