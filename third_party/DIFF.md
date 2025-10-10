TL;DR: подготавливаю **минимально-рисковый changes.patch** (2 файла) для включения флагов производительности (AMP, `torch.compile`, параметры загрузки данных) через **конфиг** — строго по правилам проекта (все параметры только из `configs/`). Боевые правки учебного цикла в `train.py`/`agent.py` **требуют просмотра их точного содержимого**; без этого патч на логику тренировки может не примениться корректно. Ниже — Repo-State Header, патч, команды для PR и следующий шаг.

# Repo-State Header

* **branch:** `prosperous_bot`
* **commit:** `a051837b4d7c17b376202947ad9a3f0b11184a52`
* **title:** `docs: cfg.rl.batch_size = 32`
* **link:** ([GitHub][1])

(Требования проекта подтверждены: параметры — только через `configs/`, перед diff показываем Repo-State Header, тест-гейтинг pytest и артефакты в `output/<config_name>/` — см. `third_party/rl-trading-binance/SYSTEM_PROMPT.md`  и документацию проекта `third_party/rl-trading-binance/README.md` .)

---

# changes.patch

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/config.py
@@
-from typing import List, Literal, Optional
+from typing import List, Literal, Optional
 
 import torch
 from pydantic import BaseModel, Field, validator
@@
 class TrainLogConfig(BaseModel):
@@
     iterations: int = 10_000
@@
 class LoggingConfig(BaseModel):
     per_trial_logs: bool = False
+
+class PerformanceConfig(BaseModel):
+    """
+    Переключатели производительности, управляемые из configs/*.py.
+    Никакой бизнес-логики — только флаги/параметры, которые затем
+    используются в train.py/agent.py (AMP, torch.compile, DataLoader).
+    """
+    # AMP (автокаст и GradScaler)
+    use_amp: bool = False
+    amp_dtype: Literal["float16", "bfloat16"] = "float16"
+    # torch.compile (PyTorch 2.x)
+    compile_mode: Optional[Literal["default", "reduce-overhead", "max-autotune"]] = None
+    compile_dynamic: bool = True
+    # CUDA/CuDNN
+    cudnn_benchmark: bool = True
+    # Параметры загрузки данных
+    dataloader_num_workers: int = 0
+    pin_memory: bool = False
+    persistent_workers: bool = False
+    prefetch_factor: Optional[int] = None
@@
 class MasterConfig(BaseModel):
@@
     logging: LoggingConfig = LoggingConfig()
+    perf: PerformanceConfig = PerformanceConfig()
 
 
 cfg = MasterConfig()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/configs/alpha.py
@@
 cfg.logging.per_trial_logs = False
 # 1000,  default = None
 cfg.debug.debug_max_size_data = None
 cfg.debug.use_final_model = False
+
+# ---------------------------
+# ⚡ Performance (hardware-tuned for GTX 1070 + i5-6600)
+# Управление ускорением ТОЛЬКО конфигом, чтобы не ломать кодовую базу.
+# AMP: экономия VRAM и потенциальный прирост на свертках; на Pascal (GTX 1070) FP16 без тензорных ядер — эффект умеренный, но полезна экономия памяти.
+cfg.perf.use_amp = True
+cfg.perf.amp_dtype = "float16"
+# torch.compile: снижает overhead Python-графа; режим "reduce-overhead" — наиболее безопасный.
+cfg.perf.compile_mode = "reduce-overhead"
+cfg.perf.compile_dynamic = True
+# DataLoader: загрузка с CPU (4 физ. ядра). Для коротких сессий — умеренные значения.
+cfg.perf.dataloader_num_workers = 4
+cfg.perf.pin_memory = True
+cfg.perf.persistent_workers = True
+cfg.perf.prefetch_factor = 2
+# CuDNN Heuristics
+cfg.perf.cudnn_benchmark = True
*** End Patch
```

---

# Что делает этот патч

* Добавляет **`PerformanceConfig`** в `third_party/rl-trading-binance/config.py` с флагами:

  * `use_amp`, `amp_dtype` — для **AMP**;
  * `compile_mode`, `compile_dynamic` — для **`torch.compile`**;
  * `cudnn_benchmark` — для авто-выбора быстрых алгоритмов сверток на GPU;
  * `dataloader_*` — потоковая загрузка данных с CPU.
    Эти параметры затем должны **использоваться в учебном цикле** (см. раздел «Next step» ниже). Требование проекта — параметры только через `configs/` — соблюдено (см. правила в SYSTEM_PROMPT) .

* В `configs/alpha.py` прописывает **значения по железу пользователя** (GTX 1070 + i5-6600 + 24 ГБ RAM):

  * AMP `float16` (экономия памяти; на Pascal ускорение умеренное),
  * `torch.compile` в режиме `reduce-overhead` (наиболее стабильный),
  * DataLoader: `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`.

> Прямой интерфейс запуска в `README.md` остаётся прежним (`python train.py configs/alpha.py`), проектная структура (train/test/backtest/optimize) — см. docs репозитория, блок Quickstart и структура проекта .

---

# Команды для PR

```bash
# 1) Новая ветка
git checkout -b feature/perf-flags-amp-compile

# 2) Применить патч
git apply --index changes.patch

# 3) Коммит
git commit -m "feat(config): add PerformanceConfig (AMP/torch.compile/DataLoader flags) and tune alpha"

# 4) Push
git push -u origin feature/perf-flags-amp-compile

# 5) PR в базовую ветку проекта
gh pr create -t "perf(config): AMP + torch.compile + DataLoader flags (alpha tuned)" -b "
### 🎯 Goal
Включить управляемые конфигом флаги производительности (AMP, torch.compile, DataLoader) без изменения обучающей логики.

### 📝 Implementation Details
- Добавлен `PerformanceConfig` в `third_party/rl-trading-binance/config.py`.
- Прописаны профили в `third_party/rl-trading-binance/configs/alpha.py` для GTX 1070 + i5-6600.
- Изменений в логике train/agent пока нет (подключение флагов — следующий PR).

### 📈 KPI/Risk Assessment
- **Sharpe:** нейтрально/рост при ускоренной итерации/час (ожидаемо +5–20% it/s).
- **Max DD:** без изменений (логика стратегии неизменна).
- **Profit Factor:** нейтрально.
Риск: AMP на Pascal может дать меньший прирост; контроль через флаг.

### 롤백 계획 (Rollback Plan)
Откатить PR. Все изменения изолированы конфигами; логика тренировки не затронута.

---
Repo-State Header:
branch: prosperous_bot
commit: a051837b4d7c17b376202947ad9a3f0b11184a52
title: docs: cfg.rl.batch_size = 32
link: https://github.com/FMProducer/prosperous_bot/commit/a051837b4d7c17b376202947ad9a3f0b11184a52
" -B prosperous_bot
```

---

# Next step (ACTION NEEDED)

Чтобы флаги **реально заработали**, необходимо внести минимальные изменения в учебный цикл. По правилам проекта я не могу «угадывать» содержимое кода — нужно привязаться к реальным функциям/местам в файлах (Ultra-strict) . Просьба подтвердить доступ/показать содержимое следующих файлов в ветке `prosperous_bot`:

1. `third_party/rl-trading-binance/train.py` — место, где:

   * создаётся `model` и `optimizer` (добавим `torch.compile` по `cfg.perf.compile_mode`);
   * считается `loss` и вызываются `loss.backward()`/`optimizer.step()` (обернём в `torch.cuda.amp.autocast` и `GradScaler` при `cfg.perf.use_amp`);
   * формируются загрузчики данных (передадим `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` из `cfg.perf`).
2. (Опционально) `third_party/rl-trading-binance/agent.py` — если градиентный шаг инкапсулирован в агенте.

После просмотра кода я подготовлю **второй патч** с точечными вставками (≤ 300 строк diff), unit-тестами под `pytest` и инструкциями запуска.

---

# План & риски

| Шаг | Действие                                                                    | KPI/риск                                                                                                          |
| --- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1   | Влить текущий патч (флаги в конфиге)                                        | Без изменения метрик; инфраструктурная подготовка.                                                                |
| 2   | Подключить флаги в `train.py` (AMP/compile/DataLoader)                      | ↑ it/s на 5–35% по данным практики PyTorch; на GTX 1070 прирост ближе к 5–15%. Риск: малый прирост AMP на Pascal. |
| 3   | Прогнать `pytest` и короткий тренинг-смок (`episodes≈200`, `output/alpha/`) | Проверка стабильности, отсутствие деградации.                                                                     |
| 4   | (Опционально) Векторизация окружений/профилинг узких мест CPU               | Потенциал 1.5–4× по it/s; требует отдельного дизайна.                                                             |

---

Если нужно — сразу подключу и *безопасные* дефолты в `train.py` (AMP/compile отключены по умолчанию), но для этого требуется увидеть реальный код целевых функций (строгие правила проекта).

**Ссылки/док-правила:** подтверждение требований к патчам/конфигам в SYSTEM_PROMPT проекта  и структура/запуски в README проекта .

[1]: https://github.com/FMProducer/prosperous_bot/commit/a051837b4d7c17b376202947ad9a3f0b11184a52 "docs: cfg.rl.batch_size = 32 · FMProducer/prosperous_bot@a051837 · GitHub"
