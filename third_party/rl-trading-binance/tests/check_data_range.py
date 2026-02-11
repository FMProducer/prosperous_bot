import json
from pathlib import Path
from datetime import datetime, timezone

def main():
    # Setup paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_dir = project_root / "user_data" / "data" / "binance" / "futures"
    
    print(f"Checking data in: {data_dir}")
    files = list(data_dir.glob("*-1m-futures.json"))
    
    if not files:
        print("❌ No data files found!")
        return

    print(f"{'File':<35} | {'Start (UTC)':<20} | {'End (UTC)':<20} | {'Candles':<8}")
    print("-" * 90)
    
    # Check first 10 files
    for f in files[:10]:
        try:
            with open(f, 'r') as d:
                data = json.load(d)
                if not data:
                    print(f"{f.name[:35]:<35} | {'EMPTY':<20} | {'EMPTY':<20} | 0")
                    continue
                
                # Freqtrade JSON: [timestamp, open, high, low, close, volume]
                start_ts = data[0][0]
                end_ts = data[-1][0]
                
                start_dt = datetime.fromtimestamp(start_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                end_dt = datetime.fromtimestamp(end_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                
                print(f"{f.name[:35]:<35} | {start_dt:<20} | {end_dt:<20} | {len(data)}")
                
                # Check for aux files (funding/mark) required for futures
                base_name = f.name.replace("-1m-futures.json", "").replace("-1m.json", "")
                funding_name = f"{base_name}-1h-funding_rate.json"
                mark_name = f"{base_name}-1h-mark.json"
                
                if not (data_dir / funding_name).exists():
                     print(f"   ⚠️ Missing: {funding_name}")
                if not (data_dir / mark_name).exists():
                     print(f"   ⚠️ Missing: {mark_name}")
        except Exception as e:
            print(f"{f.name[:35]:<35} | ERROR: {e}")

if __name__ == "__main__":
    main()
