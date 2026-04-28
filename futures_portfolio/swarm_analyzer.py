import json
import glob
import os

def analyze_swarm():
    # Загрузка конфига для динамических параметров
    try:
        with open('config.json', 'r') as c:
            config = json.load(c)
            initial_per_bot = config['portfolios'][0].get('initial_capital', 39.0)
            max_bots = config.get('max_bots', 10)
            working_capital = initial_per_bot * max_bots
            active_tickers = config.get('tickers', [])
    except Exception as e:
        print(f"Error loading config.json: {e}")
        initial_per_bot = 39.0
        working_capital = 390.0
        active_tickers = []

    total_pnl = 0
    files = glob.glob("paper_state_*.json")
    
    results = []
    
    for f in files:
        try:
            with open(f, 'r') as j:
                data = json.load(j)
                balance = data.get('balance', initial_per_bot)
                pnl = balance - initial_per_bot
                total_pnl += pnl
                # Извлекаем тикер из имени файла или из данных
                ticker_from_file = os.path.basename(f).replace('paper_state_', '').replace('.json', '')
                ticker = data.get('base_ticker', ticker_from_file)
                
                results.append({
                    "ticker": ticker,
                    "balance": balance,
                    "pnl": pnl
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Сортировка по профиту
    results.sort(key=lambda x: x['pnl'], reverse=True)
    
    print(f"\n{'Ticker':<15} | {'Balance':<10} | {'PnL (USDT)':<10} | {'Status':<10}")
    print("-" * 60)
    for r in results:
        status = "Active" if r['ticker'] in active_tickers else "Removed"
        print(f"{r['ticker']:<15} | {r['balance']:<10.2f} | {r['pnl']:<10.2f} | {status:<10}")
        
    print("-" * 60)
    print(f"Total Swarm Net Profit: {total_pnl:.2f} USDT")
    print(f"Overall ROI: {(total_pnl / working_capital) * 100:.2f}% (Relative to {working_capital} USDT deposit)")
    
    # Анализ "мертвых" vs "живых"
    inactive_pnl = sum(r['pnl'] for r in results if r['ticker'] not in active_tickers)
    active_pnl = sum(r['pnl'] for r in results if r['ticker'] in active_tickers)
    print(f"Active Bots PnL: {active_pnl:.2f} USDT")
    print(f"Removed (Historical) PnL: {inactive_pnl:.2f} USDT")
    print(f"Files analyzed: {len(results)}")

if __name__ == "__main__":
    analyze_swarm()
