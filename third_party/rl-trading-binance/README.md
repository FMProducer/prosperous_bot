# RL Trading System for Binance Futures

> **❗ Critical Notes**  
> 1. **Стратегия использует ЕДИНСТВЕННЫЙ конфиг**: `user_data/config_rl4z.json` (не путать с `configs/*.json`!)  
> 2. **Файлы в `third_party/rl-trading-binance/`**:  
>    - `agent_router.py`/`config.yaml` → интеграция с **Continue VS Code** (внешний агент-маршрутизатор)  
>    - **НЕ влияют** на торговую логику стратегии  
> 3. **Ключевые компоненты стратегии**:  
>    - `user_data/strategies/CustomD3QNStrategy4z.py`  
>    - `user_data/config_rl4z.json`  

## 🗂️ File System Map (Обновлено)

| Путь | Назначение | Критично для стратегии | Как проверить |
|------|------------|-------------------------|--------------|
| `user_data/strategies/CustomD3QNStrategy4z.py` | Основная стратегия Freqtrade | ✅ | `ls user_data/strategies/` |
| `user_data/config_rl4z.json` | Единственный конфиг стратегии | ✅ | `cat user_data/config_rl4z.json` |
| `third_party/rl-trading-binance/agent_router.py` | Маршрутизация запросов в Continue | ❌ | Не редактируйте для торговли |
| `third_party/rl-trading-binance/config.yaml` | Конфиг Continue VS Code | ❌ | Не влияет на бота |
| `configs/*.json` | Тренировочные конфиги RL (не используются в live) | ❌ | `ls configs/` → **игнорируйте** |
| `output/alpha_seed_*/` | Сохраненные модели (пример: `output/alpha_seed_001/saved_models/`) | ✅ | `ls output/` |

## 🧭 Где искать ключевые параметры?

### Всегда проверяйте эти файлы:
```bash
# 1. Конфиг стратегии (единственный источник истины)
user_data/config_rl4z.json

# 2. Код стратегии
user_data/strategies/CustomD3QNStrategy4z.py

# 3. Логи бэктеста
output/<config_name>/backtest_results.csv
```

### ❌ Что НЕ нужно редактировать для торговли:
- Все файлы в `third_party/rl-trading-binance/` (кроме документации)
- Файлы в `configs/` (используются только для тренировки моделей)
- `agent_router.py` и `config.yaml` (только для работы с Continue VS Code)

## 🛠️ Как диагностировать проблемы

### Если стратегия не запускается:
1. **Проверьте конфиг**:
   ```bash
   freqtrade validate-config --config user_data/config_rl4z.json
   ```
   - Ошибка? Значит, конфиг поврежден или находится в неправильном месте.

2. **Проверьте пути к моделям** в `user_data/config_rl4z.json`:
   ```json
   "rl_ensemble": {
     "long_1_model_dir": "output/alpha_seed_001/saved_models/",
     "long_2_model_dir": "output/alpha_seed_002/saved_models/",
     ...
   }
   ```
   - Все пути должны существовать: `ls output/alpha_seed_001/saved_models/`

### Если не генерируются сигналы:
1. **Проверьте логи** на наличие:
   - `⚠️ No trades due to dynamic epsilon threshold` → слишком высокий `epsilon_threshold_eff`
   - `⚠️ Volume filter blocked all pairs` → увеличьте `min_quote_volume_usd`

2. **Временно отключите фильтры** в `user_data/config_rl4z.json`:
   ```json
   "use_regime_filter": false,
   "enable_dynamic_epsilon": false
   ```

## 📌 Проверка целостности (Шаги перед деплоем)
```bash
# 1. Проверка расположения конфига
ls user_data/config_rl4z.json || echo "❌ Конфиг отсутствует!"

# 2. Валидация стратегии
freqtrade backtest --config user_data/config_rl4z.json --strategy CustomD3QNStrategy4z

# 3. Проверка загрузки моделей (ищите в логах)
grep "Loaded model" logs/freqtrade.log
```

## 🔍 Ссылки на документацию
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → Решение типовых проблем  
- [RL_ARCHITECT.md](RL_ARCHITECT.md) → Архитектура стратегии  
- [analysis.md](analysis.md) → Как избежать lookahead bias  

## ❗ Important Reminder
- **Для торговли используется только `user_data/`**  
- Все другие конфиги — для тренировки или интеграции с IDE  
- При сомнениях — выполняйте `ls` и сверяйтесь с этой таблицей