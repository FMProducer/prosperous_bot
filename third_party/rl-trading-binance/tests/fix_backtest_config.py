import json
from pathlib import Path

def main():
    # Setup paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    config_src = project_root / "user_data" / "config_rl4z.json"
    config_dst = project_root / "user_data" / "config_rl4z_backtest.json"
    pairs_file = project_root / "user_data" / "pairs.txt"
    
    if not config_src.exists():
        print(f"Error: Source config not found at {config_src}")
        return

    # 1. Load pairs from file
    pairs_list = []
    if pairs_file.exists():
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(pairs_list)} pairs from {pairs_file}")
    else:
        print(f"Warning: {pairs_file} not found. Using empty pairlist.")

    print(f"Reading config from {config_src}...")
    with open(config_src, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. Patch pairlists to use StaticPairList
    print("Patching pairlists to StaticPairList...")
    config['pairlists'] = [
        {
            "method": "StaticPairList",
            "allow_inactive": True  # Allow backtesting even if pairs are currently inactive
        }
    ]
    
    # 3. Inject pairs into exchange.pair_whitelist
    if 'exchange' not in config:
        config['exchange'] = {}
    
    # Overwrite whitelist with our valid pairs
    config['exchange']['pair_whitelist'] = pairs_list
    
    # 4. Force Data Configuration (Crucial for "No data found" error)
    config['data_format_ohlcv'] = 'json'
    config['trading_mode'] = 'futures'
    config['candle_type_pairs'] = 'futures'
    config['margin_mode'] = 'isolated'
    config['timeframes'] = ['1m']
    
    # Save new config
    print(f"Saving backtest config to {config_dst}...")
    with open(config_dst, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
        
    print("Done! Config patched with pairs and data settings.")

if __name__ == "__main__":
    main()
