import numpy as np
import pandas as pd
for split, fname in [('train', 'train_data_fair_8m.npz'), ('val', 'val_data_fair_2m.npz'), ('backtest', 'backtest_data_fair_2m.npz')]:
    d = np.load(f'data/{fname}', allow_pickle=True)
    sessions = len([k for k in d.files if not k.startswith('_')])
    print(f'{split.capitalize()}: {sessions} sessions')
    if sessions > 0:
        sample_k = [k for k in d.files if not k.startswith('_')][0]
        print(f'  Shape: {d[sample_k].shape}')
        print(f'  Dtype: {d[sample_k].dtype}')
        if '_keys_map_' in d.files:
            keys_map = d['_keys_map_']
            print(f'  Keys_map raw type: {type(keys_map)}, shape: {getattr(keys_map, "shape", "no shape")}')
            if isinstance(keys_map, np.ndarray) and keys_map.ndim == 0:
                keys_map = keys_map.item()
                print('  Unwrapped to:', type(keys_map))
            if isinstance(keys_map, dict):
                symbols = set(k[0] for k in keys_map.values()) if keys_map.values() else set()
                print(f'  Symbols: {len(symbols)} unique')
                print('  Sample keys:', list(keys_map.items())[:2])
                print('  Sample symbols:', sorted(list(symbols))[:3])
            elif isinstance(keys_map, (list, np.ndarray)):
                symbols = set(row[1][0] for row in keys_map if len(row) > 1) if keys_map else set()
                print(f'  Symbols: {len(symbols)} unique')
                print('  Sample keys:', [(keys_map[0][0], keys_map[0][1])] if len(keys_map) > 0 else 'empty')
                print('  Sample symbols:', sorted(list(symbols))[:3])
            else:
                print('  Unknown keys_map type; sample:', str(keys_map)[:100])
        else:
            print('  No _keys_map_')
    print()