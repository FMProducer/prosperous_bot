import numpy as np
import logging
from trading_environment_z import TradingEnvironment

def test_mirror_geometry():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Признаки (колонки данных)
    feature_names = ['open', 'high', 'low', 'close', 'volume', 'indicator']
    idx = {k: i for i, k in enumerate(feature_names)}
    
    # Синтетические данные: 2 шага, 6 признаков
    sample_seq = np.array([
        [100.0, 110.0, 95.0, 105.0, 1000.0, 102.0],
        [105.0, 115.0, 100.0, 112.0, 1100.0, 108.0]
    ])
    
    # Названия последовательностей (должно быть столько же, сколько массивов в sequences)
    sequence_ids = ['test_asset_1']
    
    stats = {
        'test_asset_1': {
            'mean': [100.0, 110.0, 95.0, 105.0, 1000.0, 102.0],
            'std': [1.0, 1.0, 1.0, 1.0, 100.0, 2.0]
        }
    }

    # 2. Инициализируем окружение
    env = TradingEnvironment(
        sequences=[sample_seq],
        stats=stats,
        keys=sequence_ids,  # Теперь длина совпадает: 1 массив == 1 ключ
        render_mode=None,
        full_seq_len=2,
        num_features=len(feature_names),
        num_actions=3,
        flat_state_size=100,
        initial_balance=1000.0,
        pre_signal_len=0,
        datachannels=feature_names, # Здесь передаем названия колонок
        slippage=0.0,
        transaction_fee=0.0,
        agent_session_len=2,
        agent_history_len=1,
        input_history_len=1,
        pricechannels=['open', 'high', 'low', 'close', 'indicator'],
        volumechannels=['volume'],
        otherchannels=[],
        action_history_len=1,
        inaction_penalty_ratio=0.0,
        filter_direction='SHORT'
    )

    mirrored_seq = env.sequences[0]
    
    print("--- Mirror Geometry Verification ---")
    for i in range(len(mirrored_seq)):
        m = mirrored_seq[i]
        o, h, l, c = m[idx['open']], m[idx['high']], m[idx['low']], m[idx['close']]
        ind = m[idx['indicator']]
        vol = m[idx['volume']]

        print(f"Step {i}: O:{o:.2f}, H:{h:.2f}, L:{l:.2f}, C:{c:.2f}, Ind:{ind:.2f}, Vol:{vol:.2f}")

        # Проверка инвариантов
        assert h >= o, f"Invariant High >= Open failed at step {i}"
        assert h >= c, f"Invariant High >= Close failed at step {i}"
        assert l <= o, f"Invariant Low <= Open failed at step {i}"
        assert l <= c, f"Invariant Low <= Close failed at step {i}"
        assert h >= l, f"Invariant High >= Low failed at step {i}"
        
        # Проверка знаков (в зеркале цены отрицательные)
        assert o < 0 and c < 0 and h <= 0 and l <= 0, "Prices must be negative in mirror mode"
        assert ind < 0, f"Indicator should be negative, got {ind}"
        assert vol == sample_seq[i][idx['volume']], "Volume should NOT be inverted"

    print("\n[SUCCESS] All geometric invariants and stats verified.")

if __name__ == "__main__":
    test_mirror_geometry()