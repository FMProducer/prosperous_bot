import json
import glob
import os

def analyze_swarm():
    # Загрузка конфига для динамических параметров
    try:
        with open('config.json', 'r') as c:
            config = json.load(c)
            # Берем initial_capital из первого портфеля (обычно 39000)
            initial_per_bot = config['portfolios'][0].get('initial_capital', 39000.0)
            max_bots = config.get('max_bots', 10)
            working_capital = initial_per_bot * max_bots
            active_tickers = config.get('tickers', [])
            live_swarm = config.get('live_swarm', [])
    except Exception as e:
        print(f"Error loading config.json: {e}")
        initial_per_bot = 39000.0
        working_capital = 390000.0
        active_tickers = []
        live_swarm = []

    total_net_pnl = 0
    total_safe = 0
    files = glob.glob("paper_state_*.json")
    
    results = []
    
    for f in files:
        try:
            with open(f, 'r') as j:
                data = json.load(j)
                balance = data.get('balance', initial_per_bot)
                
                # Извлекаем тикер
                ticker_from_file = os.path.basename(f).replace('paper_state_', '').replace('.json', '')
                ticker = data.get('base_ticker', ticker_from_file)
                
                # Ищем соответствующий state_*.json для получения SAFE
                safe_reserve = 0.0
                state_file = f"state_{ticker}.json"
                if os.path.exists(state_file):
                    with open(state_file, 'r') as sj:
                        state_data = json.load(sj)
                        safe_reserve = state_data.get('siphoning_reserve', 0.0)
                
                # Расчет PnL: (Текущий баланс - Начальный) + То что ушло в SAFE
                working_pnl = balance - initial_per_bot
                total_ticker_pnl = working_pnl + safe_reserve
                
                total_net_pnl += total_ticker_pnl
                total_safe += safe_reserve
                
                results.append({
                    "ticker": ticker,
                    "balance": balance,
                    "safe": safe_reserve,
                    "pnl": total_ticker_pnl,
                    "mode": "REAL" if ticker in live_swarm else "PAPER"
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Сортировка по общему профиту
    results.sort(key=lambda x: x['pnl'], reverse=True)
    
    print(f"\n{'Ticker':<15} | {'Balance':<10} | {'SAFE':<10} | {'Total PnL':<10} | {'Mode':<7} | {'Status':<10}")
    print("-" * 85)
    for r in results:
        status = "Active" if r['ticker'] in active_tickers else "Removed"
        print(f"{r['ticker']:<15} | {r['balance']:<10.2f} | {r['safe']:<10.2f} | {r['pnl']:<10.2f} | {r['mode']:<7} | {status:<10}")
        
    print("-" * 85)
    print(f"Total Swarm Net Profit: {total_net_pnl:.2f} USDT (including SAFE)")
    print(f"Total SAFE Reserve:     {total_safe:.2f} USDT")
    print(f"Overall ROI: {(total_net_pnl / working_capital) * 100:.2f}% (Relative to {working_capital} USDT deposit)")
    
    # Анализ "мертвых" vs "живых"
    inactive_pnl = sum(r['pnl'] for r in results if r['ticker'] not in active_tickers)
    active_pnl = sum(r['pnl'] for r in results if r['ticker'] in active_tickers)
    print(f"Active Bots Total PnL:   {active_pnl:.2f} USDT")
    print(f"Removed Bots Total PnL:  {inactive_pnl:.2f} USDT")
    print(f"Files analyzed: {len(results)}")

if __name__ == "__main__":
    analyze_swarm()
