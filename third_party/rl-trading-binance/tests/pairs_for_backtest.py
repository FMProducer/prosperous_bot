import os
from pathlib import Path

def main():
    # Determine project root (assuming script is in tests/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # Path to data
    data_dir = project_root / "user_data" / "data" / "binance" / "futures"
    output_file = project_root / "user_data" / "pairs.txt"
    
    print(f"Scanning directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    pairs = set()
    # Get all json files first, then filter strictly
    files = list(data_dir.glob("*.json"))
    
    for file_path in files:
        filename = file_path.name
        
        # 1. Filter out non-candle data types (mark prices, funding rates, indexes)
        if any(x in filename for x in ["mark", "index", "funding", "premiumIndex", "funding_rate"]):
            continue
            
        # 2. Strict suffix check for 1m candles
        if filename.endswith("-1m-futures.json"):
            raw_name = filename[:-len("-1m-futures.json")]
            # Handle Freqtrade futures naming: Base_Quote_Stake (e.g. BTC_USDT_USDT)
            parts = raw_name.split('_')
            if len(parts) >= 3:
                stake = parts[-1]
                quote = parts[-2]
                base = "_".join(parts[:-2])
                
                # FIX: Detect and strip double quote in base (e.g. 1000BONK_USDT -> 1000BONK)
                if base.endswith(f"_{quote}"):
                    # print(f"  Fixing malformed base: {base} -> {base[:-len(quote)-1]}")
                    base = base[:-len(quote)-1]
                
                pairs.add(f"{base}/{quote}:{stake}")
                continue
            pair_name = raw_name
        elif filename.endswith("-1m.json"):
            pair_name = filename[:-len("-1m.json")]
        else:
            continue

        pair_name = pair_name.replace("_", "/")
        pairs.add(f"{pair_name}:USDT")

    sorted_pairs = sorted(list(pairs))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_pairs))
        
    print(f"Готово! Сохранено пар: {len(sorted_pairs)}")
    print(f"Файл сохранен в: {output_file}")
    
    # Print first 5 for verification
    if len(sorted_pairs) > 0:
        print(f"Пример первых 5 пар (из {len(sorted_pairs)}):")
        for p in sorted_pairs[:5]:
            print(f" - {p}")

if __name__ == "__main__":
    main()
