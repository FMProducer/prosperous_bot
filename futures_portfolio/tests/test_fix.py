import asyncio
import sys
import os

# Fix for aiohttp on Windows with ProactorEventLoop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

from rank_tickers import main

try:
    result = asyncio.run(main(quiet=False, min_volume=20_000_000))
    print(f"\nTotal results: {len(result) if result else 0}")
except Exception as e:
    import traceback
    traceback.print_exc()
