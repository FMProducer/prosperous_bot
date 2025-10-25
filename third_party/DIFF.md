TL;DR: Ошибка из-за того, что в `study.set_user_attr("base_cfg", ...)` передавался не-серриализуемый объект (`torch.device`). Фикс: сохраняем базовую конфигурацию в `user_attrs` как JSON-строку (`model_dump_json()`), а в `objective()` парсим обратно в dict. Ниже — минимальный patch и команды запуска.

**Repo-State Header (требуется Ultra-strict):**
Ветка: `prosperous_bot` (дефолт) — подтверждено через GitHub API. SHA последнего коммита не удалось получить автоматически из-за ограничений доступа к API в этом окружении. Ссылка на репозиторий: [https://github.com/FMProducer/prosperous_bot](https://github.com/FMProducer/prosperous_bot)   

---

| Шаг | Действие                                                             | KPI/риск                                                                                      |
| --- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | Сделать `base_cfg` JSON-совместимой при записи в `study.user_attrs`  | Убираем падение на сериализации (`TypeError: Object of type device is not JSON serializable`) |
| 2   | В `objective()` читать `base_cfg` из `user_attrs` через `json.loads` | Гарантируем совместимость при `n_jobs>1` и с RDB storage (SQLite)                             |
| 3   | Перезапустить оптимизацию                                            | Получаем корректные логи и `trials.parquet` без падений                                       |

---

# Пояснение проблемы

* В `optimize_cfg.py` вы записываете в Optuna user_attrs Python-словарь с полями конфигурации: `study.set_user_attr("base_cfg", base_cfg.model_dump())`. Внутри конфигурации находится объект типа `torch.device`, который стандартный `json.dumps` не умеет сериализовать — отсюда `TypeError` (ваш стек-трейс). Источник в коде: блок записи user_attrs в `main()` и чтение в `objective()`  .
* В файл `orig_master_cfg.json` вы уже писали так: `json.dump(..., default=str)`, и там проблем нет — но внутри Optuna нет возможности передать `default=str`, поэтому нужно заранее преобразовать в JSON-строку (или рекурсивно приводить к строкам).

# Unified diff (≤ 300 строк)

**Файл:** `third_party/rl-trading-binance/optimize_cfg.py`  

```diff
--- a/third_party/rl-trading-binance/optimize_cfg.py
+++ b/third_party/rl-trading-binance/optimize_cfg.py
@@ -12,6 +12,7 @@
 import optuna
 
 from backtest_engine import run_backtest
 from config import MasterConfig
 from utils import load_config, setup_logging
+import json
 
 def objective(trial: optuna.Trial):
-    # Reconstruct the config object from the dictionary stored in user_attrs
-    base_cfg_dict = trial.study.user_attrs["base_cfg"]
+    # Reconstruct the config object from the JSON stored in user_attrs
+    base_cfg_raw = trial.study.user_attrs["base_cfg"]
+    base_cfg_dict = json.loads(base_cfg_raw) if isinstance(base_cfg_raw, str) else base_cfg_raw
     cfg = MasterConfig.model_validate(base_cfg_dict)
     cfg.random_seed = 17 + trial.number
 
@@ -79,8 +80,11 @@
         load_if_exists=False,
     )
 
-    # Store base config and paths in study's user attributes to pass to workers
-    study.set_user_attr("base_cfg", base_cfg.model_dump())
+    # Store base config and paths in study's user attributes to pass to workers
+    # IMPORTANT: store as JSON string to avoid non-serializable objects (e.g., torch.device)
+    # Pydantic v2: model_dump_json() returns a JSON string with safe encoders
+    base_cfg_json = base_cfg.model_dump_json()
+    study.set_user_attr("base_cfg", base_cfg_json)
     study.set_user_attr("opt_dir", opt_dir)
 
     logging.info(f"[Optuna] starting optimisation -- trials={args.trials} jobs={args.jobs}")
     start_t = time.time()
```

# Как запустить (Windows, ваш путь)

```powershell
# (из корня репозитория/проекта)
git checkout -b fix/optuna-json-user-attrs
git apply --index changes.patch
git commit -m "fix(optimize_cfg): store base_cfg as JSON in Optuna user_attrs and parse in objective()"
# локальный прогон
python third_party/rl-trading-binance/optimize_cfg.py third_party/rl-trading-binance/configs/alpha.py --trials 100 --jobs 4
```

# Почему это решает проблему

* `study.set_user_attr(...)` внутри Optuna сериализует значение через `json.dumps`. Передавая **строку JSON**, мы гарантируем корректную запись (без попытки сериализовать `torch.device`). Чтение — обратное преобразование `json.loads(...)`.
* Изменения минимальны и не затрагивают вашу логику поиска гиперпараметров, кеши, метрики и выгрузку артефактов (`trials.parquet`, `optuna_history.png`, `pareto.png`)  .

# Быстрая проверка

1. После применения патча ожидаемый вывод в логах:

```
[INFO] [Optuna] Output dir: output\alpha\optuna_cfg_optimization_results
[I ...] A new study created in RDB with name: backtest_opt_...
[INFO] [Optuna] starting optimisation -- trials=100 jobs=4
...
```

2. Файлы/артефакты:

* `output\alpha\optuna_cfg_optimization_results\optuna.db`
* `output\alpha\optuna_cfg_optimization_results\trials.parquet`
* `output\alpha\optuna_cfg_optimization_results\best_backtest_cfg.json` (если нашлись успешные трейалы)
* PNG-графики истории и Парето (если `matplotlib` доступен)  .

# Ссылки на специфику проекта (для аудита)

* Требование следовать структуре и Ultra-strict режиму — `SYSTEM_PROMPT.md`  
* Подтверждение назначения `optimize_cfg.py` и пайплайна — `README.md` (раздел Quickstart, п.5)  

Если нужно, могу дополнительно подготовить вариант с «универсальным» рекурсивным преобразователем словаря (`make_jsonable()`), но текущий фикс через `model_dump_json()` + `json.loads()` — проще и надёжнее для Optuna RDB.
