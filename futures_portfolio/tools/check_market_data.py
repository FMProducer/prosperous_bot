import os
import json
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

def check():
    # Load tickers from config.json
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        tickers = config.get('tickers', [])
    except Exception as e:
        print(f"Error loading config.json: {e}")
        tickers = ['VICUSDT', 'SAGAUSDT']

    if not tickers:
        print("No tickers found in config.json")
        return

    c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))
    print(f"--- Real-time Market Snapshot for {len(tickers)} tickers ---")
    for t in tickers:
        try:
            # Spread
            d = c.futures_order_book(symbol=t, limit=5)
            b = float(d['bids'][0][0])
            a = float(d['asks'][0][0])
            s = (a - b) / b * 100
            
            # 1m Klines for Velocity
            k = c.futures_klines(symbol=t, interval='1m', limit=1)
            p_open = float(k[0][1])
            p_close = float(k[0][4])
            p_high = float(k[0][2])
            p_low = float(k[0][3])
            
            v_range = (p_high - p_low) / p_low * 100
            v_change = (p_close - p_open) / p_open * 100
            
            print(f"\n[{t}]")
            print(f"  Best Bid: {b:.6f} | Best Ask: {a:.6f}")
            print(f"  Current Spread:   {s:.4f}%")
            print(f"  1m Net Change:    {v_change:+.4f}%")
            print(f"  1m High-Low Range: {v_range:.4f}% (Internal Volatility)")
            print(f"  Current Price:    {p_close}")
        except Exception as e:
            print(f"  {t}: Error {e}")

if __name__ == '__main__':
    check()
