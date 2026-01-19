import sys
from pathlib import Path

# Dynamically find freqtrade in parents[3] (third_party) or parents[4] (root)
curr_path = Path(__file__).resolve()
for p in [curr_path.parents[3], curr_path.parents[4]]:
    ft_path = p / "freqtrade"
    if ft_path.exists():
        sys.path.insert(0, str(ft_path))
        break

import pandas as pd
from freqtrade.data.history import load_pair_history  # type: ignore
from freqtrade.enums import CandleType  # type: ignore
from freqtrade.configuration import Configuration  # type: ignore

# Настройка путей (как в команде загрузки)
user_data_dir = Path(r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\ft_userdata")
args = {"config": [str(user_data_dir / "config_rl.json")], "user_data_dir": str(user_data_dir), "strategy": None}
config = Configuration(args).get_config()
data_dir = user_data_dir / "data" / "binance"

# Загрузка скачанного файла
pair = "BTC/USDT:USDT"
timeframe = "1m"
candle_type = CandleType.FUTURES

try:
    df = load_pair_history(
        datadir=data_dir,
        timeframe=timeframe,
        pair=pair,
        candle_type=candle_type
    )

    print(f"\n✅ Data loaded for {pair}")
    print(f"Columns found: {list(df.columns)}")
    
    required_cols = ['quote_volume', 'num_trades', 'taker_base', 'taker_quote']
    missing = [c for c in required_cols if c not in df.columns]
    
    if not missing:
        print("🎉 SUCCESS: All extended columns are present!")
    else:
        print(f"❌ FAILURE: Missing columns: {missing}")
        
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
