#!/usr/bin/env python3
"""Wrapper to run rank_tickers with debug output."""
import asyncio
import sys
import os

# Monkey-patch to add debug logging
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

# Add a debug wrapper around fetch
import aiohttp
_orig_init = aiohttp.ClientSession.__init__

async def main():
    # Patch the TickerScanner.fetch to add logging
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import after patching
    from rank_tickers import TickerScanner, main as orig_main
    import rank_tickers
    
    # Override fetch to add logging
    orig_fetch = rank_tickers.TickerScanner.fetch
    
    async def debug_fetch(self, session, endpoint, params=None):
        result = await orig_fetch(self, session, endpoint, params)
        if result is None:
            print(f"  [DEBUG] fetch returned None for {endpoint} with params={params}")
        return result
    
    rank_tickers.TickerScanner.fetch = debug_fetch
    
    # Run
    await rank_tickers.main(quiet=False, min_volume=20_000_000)

asyncio.run(main())
