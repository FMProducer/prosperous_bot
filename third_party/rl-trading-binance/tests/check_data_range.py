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

    print(f"{'File':<40} | {'Start':<12} | {'End':<12} | {'Status'}")
    print("-" * 100)
    
    # Sort and check ALL files
    files.sort()
    
    for f in files:
        try:
            with open(f, 'r') as d:
                data = json.load(d)
                if not data:
                    print(f"{f.name[:40]:<40} | {'EMPTY':<12} | {'EMPTY':<12} | ❌ 0 candles")
                    continue
                
                # Freqtrade JSON: [timestamp, open, high, low, close, volume]
                start_ts = data[0][0]
                end_ts = data[-1][0]
                
                start_dt = datetime.fromtimestamp(start_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')
                end_dt = datetime.fromtimestamp(end_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')
                
                # Check for aux files (funding/mark) required for futures
                base_name = f.name.replace("-1m-futures.json", "").replace("-1m.json", "")
                funding_name = f"{base_name}-1h-funding_rate.json"
                mark_name = f"{base_name}-1h-mark.json"
                
                missing = []
                if not (data_dir / funding_name).exists():
                     missing.append("Funding")
                if not (data_dir / mark_name).exists():
                     missing.append("Mark")
                
                if missing:
                    status = f"⚠️ Missing: {', '.join(missing)}"
                else:
                    status = f"✅ OK ({len(data)})"
                
                print(f"{f.name[:40]:<40} | {start_dt:<12} | {end_dt:<12} | {status}")
        except Exception as e:
            print(f"{f.name[:40]:<40} | ERROR: {e}")

if __name__ == "__main__":
    main()
