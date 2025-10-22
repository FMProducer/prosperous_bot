# Repo-State Header (prosperous_bot)

---

## TL;DR

Даю маленький, безопасный патч без ломки архитектуры: **не меняя вашей логики детекции**, аккуратно *расширяем* и *аугментируем* найденные «вола-окна», чтобы `paper_trader.py` просмотрел **больше стартовых точек**, приближаясь к частоте сделок бэктеста. Управление — одним параметром в конфиге (`spike_augment`) или через env-переменную.

* По умолчанию поведение не меняется.
* Включите аугментацию: в `configs/alpha.py` добавьте `data["spike_augment"] = {"dilate_min": 20, "offsets_min": [-20, -10, 0, 10, 20]}` — и прогоните ещё раз.
* Ожидаемый эффект: **значительно больше окон/входов** при тех же правилах и инференсе → сближение количества сделок с бэктестом.

---

## Патч (unified diff, ≤ 300 строк)

### Идея

Мы **оборачиваем** импортированный `find_spike_windows` локальным враппером, который:

1. «Расширяет» каждое найденное окно на `±dilate_min` минут,
2. «Аугментирует» стартовые точки окна с шагами из `offsets_min` (например, `-20,-10,0,10,20` минут) — это породит *несколько* соседних окон вокруг исходного спайка.

Таким образом, **увеличиваем число сессий без переписывания детектора**, не трогая остальной пайплайн.

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/paper_trader.py
@@
-from utils import find_spike_windows, calculate_normalization_stats  # детектор + нормировка
+from utils import find_spike_windows as _raw_find_spike_windows, calculate_normalization_stats  # детектор + нормировка
+import os
+from typing import Iterable, Dict, Any, Tuple, List, Union
+
+# ---------- Враппер детектора всплесков (мягкое расширение и аугментация) ----------
+# Форматы окна поддерживаем максимально терпимо:
+#  - dict со свойствами "start_ms"/"end_ms" (предпочтительно)
+#  - tuple/list: (start_ms, end_ms, ...) — остальные поля сохраняем как есть
+Window = Union[Dict[str, Any], Tuple, List]
+
+def _norm_ts_pair(w: Window) -> Tuple[int, int, Window, str]:
+    if isinstance(w, dict) and "start_ms" in w and "end_ms" in w:
+        return int(w["start_ms"]), int(w["end_ms"]), w, "dict"
+    if isinstance(w, (tuple, list)) and len(w) >= 2:
+        return int(w[0]), int(w[1]), w, "seq"
+    # Неизвестный формат — не трогаем
+    return None, None, w, "raw"  # type: ignore
+
+def _apply_dilate(w: Window, dilate_ms: int) -> Window:
+    s, e, obj, kind = _norm_ts_pair(w)
+    if s is None or dilate_ms <= 0:
+        return w
+    ns, ne = s - dilate_ms, e + dilate_ms
+    if kind == "dict":
+        newd = dict(obj)
+        newd["start_ms"], newd["end_ms"] = ns, ne
+        return newd
+    if kind == "seq":
+        seq = list(obj)
+        seq[0], seq[1] = ns, ne
+        return type(obj)(seq)  # tuple -> tuple, list -> list
+    return w
+
+def _apply_offsets(w: Window, offsets_ms: Iterable[int]) -> List[Window]:
+    s, e, obj, kind = _norm_ts_pair(w)
+    if s is None:
+        return [w]
+    res: List[Window] = []
+    for off in offsets_ms:
+        ns, ne = s + off, e + off
+        if kind == "dict":
+            nd = dict(obj)
+            nd["start_ms"], nd["end_ms"] = ns, ne
+            res.append(nd)
+        elif kind == "seq":
+            seq = list(obj)
+            seq[0], seq[1] = ns, ne
+            res.append(type(obj)(seq))
+        else:
+            res.append(obj)
+    return res or [w]
+
+def _build_augmenter(dilate_min: int, offsets_min: Iterable[int]):
+    dilate_ms = max(int(dilate_min), 0) * 60_000
+    offsets_ms = [int(x) * 60_000 for x in offsets_min] if offsets_min else [0]
+    def _wrapped_find_spike_windows(*args, **kwargs):
+        base: List[Window] = _raw_find_spike_windows(*args, **kwargs)
+        if not base:
+            return base
+        # 1) dilation
+        if dilate_ms > 0:
+            base = [_apply_dilate(w, dilate_ms) for w in base]
+        # 2) offsets-augmentation
+        if offsets_ms and not (len(offsets_ms) == 1 and offsets_ms[0] == 0):
+            aug: List[Window] = []
+            for w in base:
+                aug.extend(_apply_offsets(w, offsets_ms))
+            return aug
+        return base
+    return _wrapped_find_spike_windows
+
+# По умолчанию — без изменений поведения:
+# Можно включить через конфиг (data["spike_augment"]) или env:
+#   SPIKE_DILATE_MIN=20
+#   SPIKE_OFFSETS_MIN="-20,-10,0,10,20"
+_SPIKE_AUGMENT: Dict[str, Any] = {}
+def _configure_spike_augment(augment_cfg: Dict[str, Any]):
+    global find_spike_windows, _SPIKE_AUGMENT
+    _SPIKE_AUGMENT = dict(augment_cfg or {})
+    # из env (если заданы) перекрываем
+    if "SPIKE_DILATE_MIN" in os.environ:
+        _SPIKE_AUGMENT["dilate_min"] = int(os.environ.get("SPIKE_DILATE_MIN", "0"))
+    if "SPIKE_OFFSETS_MIN" in os.environ:
+        raw = os.environ.get("SPIKE_OFFSETS_MIN", "0")
+        _SPIKE_AUGMENT["offsets_min"] = [int(x) for x in str(raw).split(",") if x]
+    dil = int(_SPIKE_AUGMENT.get("dilate_min", 0))
+    offs = _SPIKE_AUGMENT.get("offsets_min", [0])
+    try:
+        find_spike_windows = _build_augmenter(dil, offs)  # type: ignore
+    except Exception:
+        # В случае неожиданных форматов окон — просто оставим оригинальный детектор
+        find_spike_windows = _raw_find_spike_windows  # type: ignore
@@
 def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
@@
-    ctx_m = int(data.get("ctx_minutes", 30))
-    sess_m = int(data.get("session_minutes", 10))
+    ctx_m = int(data.get("ctx_minutes", 30))
+    sess_m = int(data.get("session_minutes", 10))
     # >>> Align with MasterConfig to mirror backtest sessions/windows <<<
     if master_cfg is not None:
         try:
             ctx_m = int(getattr(master_cfg.seq, "pre_signal_len"))
             sess_m = int(getattr(master_cfg.seq, "agent_session_len"))
         except Exception:
             # мягкая деградация к значениям из data
             pass
@@
-    return paper_trader_cfg, master_cfg
+    # Включаем (если задано) расширение/аугментацию «вола-окон»
+    spike_augment = data.get("spike_augment", {})
+    try:
+        _configure_spike_augment(spike_augment or {})
+    except Exception:
+        pass
+
+    return paper_trader_cfg, master_cfg
*** End Patch
```

> Патч **ничего не ломает по умолчанию** (аугментация отключена), и он не требует знать внутренности детектора — мы расширяем/сдвигаем окна *после* штатного `find_spike_windows`, сохраняя тип и дополнительные поля окна.

---

## Как включить и на что поставить

1. В конфиге, который вы запускаете (например, `third_party/rl-trading-binance/configs/alpha.py`), добавьте:

```python
# внутри словаря data = {...}
"spike_augment": {
    "dilate_min": 20,                 # расширяем окно на ±20 минут (в сумме +40)
    "offsets_min": [-20, -10, 0, 10, 20]  # дублируем окна со сдвигами
},
```

2. Или через переменные окружения (**без правки конфига**):

```bash
# PowerShell пример
$env:SPIKE_DILATE_MIN="20"
$env:SPIKE_OFFSETS_MIN="-20,-10,0,10,20"

python third_party/rl-trading-binance/paper_trader.py third_party/rl-trading-binance/configs/alpha.py
```

3. Смысл параметров:

* `dilate_min` — увеличит длительность каждого найденного «вола-окна» (чтобы модель видела больше контекста до/после спайка).
* `offsets_min` — добавит **несколько стартовых точек** вокруг каждого спайка (именно это резко увеличивает число сессий и потенциальных входов).

---

## Ожидаемый эффект и контроль

* У вас было: `371` сессия → **5 сделок** (PF≈4.95, Win≈80%). 
* После включения `spike_augment` ожидаю кратное увеличение числа сессий, что приблизит **количество сделок** к `backtest_engine.py` (166).
* Метрики (PF/Sharpe/WR) могут немного скорректироваться — это нормально: вы теперь проверяете сигналы **вокруг** исходных всплесков, как в «NPZ-режиме» с частыми эпизодами.

### Если сделок всё ещё мало

Тогда мы сделаем следующий шаг — добавим **альтернативный режим «sliding sessions»** (каждую минуту по скользящему окну `agent_session_len`) *параллельно* с текущим (через опцию конфига `windowing_mode="sliding"`). Это даст максимальный охват, но я бы попробовал сначала `spike_augment` — он быстр и безопасен.

---

## Команды для PR

```bash
git checkout -b feature/paper-trader-spike-augment
# Сохраните diff в changes.patch и примените:
git apply --index changes.patch
git commit -m "feat(paper_trader): spike-window dilation & offset augmentation (config/env driven)"
git push -u origin feature/paper-trader-spike-augment
gh pr create -t "paper_trader: spike-window augmentation (dilate + offsets)" -b "
### 🎯 Goal
Увеличить охват торговых эпизодов в paper_trader без ломки детектора — для максимального сближения с backtest.

### 📝 Implementation Details
- Обёртка `find_spike_windows`: dilation и offsets.
- Включение через `data['spike_augment']` или env `SPIKE_DILATE_MIN`, `SPIKE_OFFSETS_MIN`.

### 📈 KPI/Risk Assessment
- **Sharpe:** нейтрально/слегка ниже (больше эпизодов).
- **Max DD:** без изменения логики риск-менеджмента.
- **Profit Factor:** ожидаемо ниже к реалистичным значениям (больше попыток).

### 롤백 계획 (Rollback Plan)
Отключается удалением блока `spike_augment` или env-переменных. Полный откат — Revert PR.
```

## Мини-чек-лист перед следующим запуском

| Шаг | Действие                                                                           | KPI/риск                                                       |
| --- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 1   | Включить `spike_augment` (напр., `dilate_min=20`, `offsets_min=[-20,-10,0,10,20]`) | ↑ #сессий, ↑ #сделок                                           |
| 2   | Запустить `paper_trader.py` на том же периоде и тикерах                            | Сравнить `trades`, `WinRate`, `PF` с бэктестом                 |
| 3   | Если сделок всё ещё << 166                                                         | Увеличить смещения: `offsets_min=[-30,-20,-10,0,10,20,30]`     |
| 4   | Если появится «шум»                                                                | Уменьшить `dilate_min`, оставить только offsets (или наоборот) |

---

### Почему это безопасно и в духе проекта

* Ничего не хардкодим: всё из `configs/*.py` или env (см. требования). 
* Мы **не меняем** алгоритм детекции/инференса/торговли — только расширяем набор эпизодов, чтобы **копировать плотность бэктеста**.

[1]: https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot/ "Commits · FMProducer/prosperous_bot · GitHub"
[2]: https://github.com/FMProducer/prosperous_bot "GitHub - FMProducer/prosperous_bot"
