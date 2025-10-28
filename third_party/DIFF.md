Ниже — параметры для **pullback/контр-тренд** модели. Примените их в своей конфигурации.

### Общие

* `random_seed = 303`

### Обучение / валидация

* `trainlog.episodes = 320_000`
* `trainlog.num_val_ep = 2_500`
* `trainlog.val_freq = 1_000`

### RL

* `rl.batch_size = 64`
* `rl.learning_rate = 3e-4`
* `rl.gamma = 0.993`
* `rl.n_step = 3`
* `rl.train_start = 12_000`
* `rl.target_update_steps = 2_500`
* `rl.grad_clip_norm = 1.0`

### Replay / PER

* `per.buffer_size = 180_000`

### Параллелизм

* `vec.num_envs = 2`

### Детектор (короткий взгляд)

* `detector.context_minutes = 45`
* `detector.window_minutes = 9`
* `detector.cooldown_minutes = 18`
* `detector.use_lookahead = False`  *(для paper/онлайна; True — только для оффлайн-бэктеста/генерации датасетов)*

### Пороги сигналов (шорт-смещение, быстрые выходы)

* `signals.long_action_threshold = 0.0072`
* `signals.short_action_threshold = 0.0070`
* `signals.close_action_threshold = 0.011`

### Риск-менеджмент

* `risk.take_profit = None`
* `risk.trailing_stop = 0.019`
* `risk.trailing_stop_min = 0.0047`
* `risk.delta_p_hysteresis = 0.0019`

### Размер позиции

* `backtest.position_fraction = 0.38`

### Ансамблевый фильтр

* `selection_strategy = "ensemble_q_filter"`
* `ensemble_n_samples = 5`
* `ensemble_max_sigma = 0.01`

### Архитектура (короткий контекст)

* `agent_history_len = 20`
* `agent_session_len = 8`
* `ACTION_HISTORY_LEN = 2`
* `cnn_maps = [64, 64, 96]`
* `kernels = [5, 3, 3]`
* `strides = [1, 1, 1]`
* `dropout_p = 0.05`
* `dense_val = [96, 48]`
* `dense_adv = [96, 48]`

**Опционально (если поддерживается в коде):**

* `risk.break_even_on_profit = True`
* `risk.break_even_trigger_R = 0.5`
