# 🛠️ Troubleshooting Guide for RL-Trading-Binance

Этот документ предназначен для AI-агента и разработчиков. Он описывает типовые проблемы, возникающие в архитектуре `CustomD3QNStrategy4z`, и алгоритмы их решения.

## 🚨 1. Initialization & Model Loading

### 🔴 Symptom: `FileNotFoundError` or `Config not found`
- **Context:** Strategy initialization (`__init__`).
- **Likely Cause:**
  - Пути к моделям (`long_1_model_dir` и т.д.) в `CustomD3QNStrategy4z.py` указывают на несуществующие директории.
  - В папке модели отсутствует файл конфигурации (например, `alpha_seed_...py`).
- **Agent Action:**
  1. Проверь переменные `self.long_1_model_dir` в коде стратегии.
  2. Используй `ls -R` (или аналог) для проверки содержимого папки `output/`.
  3. Убедись, что `project_root` определяется корректно (зависит от того, откуда запущен `freqtrade`).

### 🔴 Symptom: `ValidationError` / `ImportError` inside `_load_py_config`
- **Context:** Loading model configuration files using `importlib`.
- **Likely Cause:**
  - Структура класса `Config` (Pydantic) в коде изменилась, а сохраненный конфиг модели использует старую структуру.
  - Отсутствуют импорты внутри конфига модели.
- **Agent Action:**
  1. Проверь лог на наличие `Attempting to bypass Pydantic validation...`.
  2. Если критично, добавь `try-except` блок в `_load_py_config` или временно пропатчи определение `Config` в `config.py` для совместимости.

---

## 📉 2. Signal Generation (Inference)

### ⚠️ Symptom: No Trades (Logs show "Votes L:0/1 S:0/1")
- **Context:** `populate_entry_trend`.
- **Likely Cause:**
  1. **Normalization:** `q_min` / `q_max` в `config_rl4z.json` слишком жесткие или не инициализированы (равны 0).
  2. **Dynamic Epsilon:** `epsilon_threshold_eff` вырос до 1.0 из-за просадки (Defensive Mode).
  3. **Volume Filter:** `min_quote_volume_usd` отсекает все пары.
- **Agent Action:**
  1. Проверь лог на наличие записей `⚖️ Auto-tuned ...`. Если их нет, возможно, история `adv_history` не заполняется.
  2. Проверь текущий статус Epsilon в логах: `🛡️ DEFENSIVE MODE`.
  3. Проверь `config_rl4z.json` -> `rl_ensemble` -> `q_normalization`. Если там нули, запусти стратегию в `dry_run` на пару часов для авто-калибровки.

### ⚠️ Symptom: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
- **Context:** `_parallel_inference` -> `policy_net.forward`.
- **Likely Cause:**
  - Несовпадение размерности входного тензора и весов модели.
  - Часто бывает, если `window=90` в стратегии, а модель обучена на `window=100`.
  - Или количество фичей (каналов) изменилось (например, добавили `vwap`, а модель старая).
- **Agent Action:**
  1. Сравни `window` в `populate_indicators` (сейчас 90) и `seq_len` в конфиге модели.
  2. Проверь `get_model_input`: правильно ли формируется shape `(Batch, Channels, Length, 1)`.

---

## 🛡️ 3. Logic & Filters

### ⛔ Symptom: "Filtered by ST (Bullish/Bearish)" constantly
- **Context:** `populate_entry_trend`.
- **Likely Cause:**
  - Неверно рассчитывается `st_regime_15m`.
  - Данные для `informative_pair` (15m) не подгружаются.
- **Agent Action:**
  1. Проверь метод `informative_pairs`: возвращает ли он пары с таймфреймом `15m`.
  2. Проверь `populate_indicators`: корректно ли работает `merge_informative_pair`.
  3. Временно отключи `use_regime_filter = False` в конфиге, чтобы изолировать проблему.

### 🎰 Symptom: Slots Allocation is 0 for one side
- **Context:** `_update_slot_allocation`.
- **Likely Cause:**
  - Сильный перекос PnL (например, Long +1000, Short -1000).
  - `aggression_factor` слишком высокий.
- **Agent Action:**
  1. Проверь лог `🎰 SLOTS`.
  2. Убедись, что `min_slots_per_side` (default 10) соблюдается.

---

## ⚙️ 4. Performance & System

### 🐌 Symptom: "Strategy took XX seconds"
- **Context:** Freqtrade warning.
- **Likely Cause:**
  - `ThreadPoolExecutor` не используется эффективно.
  - Слишком много пар в `pair_whitelist`.
  - PyTorch использует слишком много потоков на одно ядро (`torch.set_num_threads`).
- **Agent Action:**
  1. Проверь `cpu_threads` в конфиге. Для 4 моделей оптимально 4-8 потоков.
  2. Убедись, что `_parallel_inference` запускается.
  3. Проверь кэширование: `get_model_input_cached` должно работать.

### 💾 Symptom: High Memory Usage (OOM)
- **Context:** System crash.
- **Likely Cause:**
  - Утечка памяти в `adv_history` или `feature_cache`.
  - PyTorch тензоры не освобождаются (накапливаются градиенты, хотя нужен `no_grad`).
- **Agent Action:**
  1. Проверь `self.cache_max_size` (должен быть небольшим, ~100).
  2. Убедись, что везде используется `with torch.no_grad():`.
  3. Проверь, что `agent.policy_net.eval()` вызван и `requires_grad=False` установлен.

---

## 📝 Checklist for Agent before PR

1. **Paths:** Все пути к моделям относительны `project_root` или абсолютны и корректны.
2. **Config:** Параметры `q_min`/`q_max` вынесены в JSON, а не захардкожены.
3. **Safety:** `can_short` проверен (для Futures).
4. **Tests:** Запущен `pytest` (если есть тесты стратегии).