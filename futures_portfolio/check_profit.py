import glob
import json
import os

states = glob.glob('paper_state_*.json')
total_tpv = 0
total_pnl = 0
initial = 260.0
count = 0

print(f"Analyzing {len(states)} bots...")
print(f"{'Ticker':<15} | {'TPV':<10} | {'PnL':<10}")
print("-" * 40)

for sf in states:
    try:
        with open(sf, 'r') as f:
            data = json.load(f)
            tpv = data.get('tpv', 0)
            pnl = data.get('total_pnl', 0)
            ticker = os.path.basename(sf).replace('paper_state_', '').replace('.json', '')
            total_tpv += tpv
            total_pnl += pnl
            count += 1
            print(f"{ticker:<15} | {tpv:<10.2f} | {pnl:<10.2f}")
    except Exception as e:
        print(f"Error reading {sf}: {e}")

if count > 0:
    avg_tpv = total_tpv / count
    total_initial = count * initial
    portfolio_roi = (total_tpv / total_initial - 1) * 100
    
    print("-" * 40)
    print(f"Total Bots:       {count}")
    print(f"Aggregated TPV:   {total_tpv:.2f}$")
    print(f"Aggregated PnL:   {total_pnl:.2f}$")
    print(f"Average TPV:      {avg_tpv:.2f}$ (Initial: {initial:.2f}$)")
    print(f"Portfolio ROI:    {portfolio_roi:.4f}%")
else:
    print("No states found.")
