Процесс обучения тормозит из-за **критического CPU bottleneck** в препроцессинге данных, вызванного увеличением `agent_history_len` с 30 до 90 минут.[1][2][3][4][5]

## Основные проблемы

**CPU препроцессинг (КРИТИЧЕСКИЙ):** На каждом шаге `_get_observation()` вызывает `apply_normalization` для окна 90×7 элементов, что в 3 раза больше исходного. С 4 параллельными средами это 40 нормализаций на эпизод, каждая включая price_diff, volume scaling и другие операции.[6][4][7][8][1]

**Увеличенное рецептивное поле:** Каждое наблюдение теперь 630 элементов вместо 210, требуя больше копирований памяти и CPU→GPU transfers.[2][4]

**DummyVecEnv + DataLoader workers:** Конфигурация `DummyVecEnv` (1 процесс) + `num_workers=4` создает GIL contention, так как все workers конкурируют за Python GIL в одном процессе.[7][2][6]

## Немедленное решение (3-5x ускорение)

### 1. Pre-normalize датасет (utils.py)

```python
def preprocess_sequences(
    sequences: List[np.ndarray],
    stats: Dict[str, Dict[str, float]],
    data_channels: List[str],
    price_channels: List[str],
    volume_channels: List[str],
    other_channels: List[str]
) -> List[np.ndarray]:
    """Pre-normalize all sequences to avoid runtime overhead."""
    normalized = []
    for seq in tqdm(sequences, desc="Normalizing sequences"):
        norm_seq = apply_normalization(
            seq, stats, data_channels,
            price_channels, volume_channels, other_channels,
            agent_history_len=seq.shape[0],
            input_history_len=seq.shape[0]
        )
        normalized.append(norm_seq)
    return normalized
```

### 2. Применить в train.py (после process_data)

```python
train_seqs = process_data(train_data, "Train", cfg)
val_seqs = process_data(val_data, "Val", cfg)

# PRE-NORMALIZE
from utils import preprocess_sequences
train_seqs = preprocess_sequences(
    train_seqs, stats,
    cfg.data.data_channels,
    cfg.data.price_channels,
    cfg.data.volume_channels,
    cfg.data.other_channels
)
val_seqs = preprocess_sequences(val_seqs, stats, ...)
```

### 3. Упростить trading_environment.py (_get_observation)

```python
def _get_observation(self):
    end = self.pre_signal_len + self.step_idx
    start = end - self.agent_history_len
    normalized = self.current_seq[start:end]  # УЖЕ НОРМАЛИЗОВАНО!
    
    # ... остальной код без изменений
```

### 4. Отключить DataLoader workers (alpha_convolutions_seed_404.py)

```python
cfg.perf.dataloader_num_workers = 0  # Отключить
cfg.perf.persistent_workers = False
```

## Ожидаемый результат

**До:** 2.43 сек/эпизод → 40 часов до конца[4][5]
**После:** 0.5-1.0 сек/эпизод → 8-16 часов до конца

**Итоговое ускорение:** 3-5x (с дополнительными оптимизациями до 10x)[9][10][4]

Примените эти изменения немедленно для критического улучшения производительности.[8][6][4]