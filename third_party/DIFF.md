## TL;DR

Код в целом корректно реализует задуманную логику (демо 30-10, look-ahead управляется конфигом), **но есть два критичных момента**:

1. В режиме **без look-ahead** сессия торговли стартует слишком рано — от начала окна оценки, а не с момента детекции. Это заложено в `utils.find_spike_windows()` и нарушает потоковую семантику. Исправляем, чтобы при `use_lookahead=False` сессия начиналась в **t**, а не в **t−window**. 

2. В `paper_trader.py` после построения индекса «на лету» даты остаются строками. Блок лимитирования окон «на тикер/день» (`.dt.floor("D")`) тогда упадёт. Нужно привести колонки времени к `datetime64[ns, UTC]` перед использованием `.dt`. 

Прочее — соответствует требованиям «конфигурации только из configs», демо-параметры 30-10 валидны и согласованы с документацией проекта.  

---

## Проверка по файлам

**`configs/alpha.py`**

* Базовые параметры модели и сессий заданы как 30-10 (демо), что соответствует README (демо: 10-мин сессии, 30-мин контекст).  
* `detector.use_lookahead = True` по умолчанию — для репликации офлайн-бэктеста; при потоках из БД можно выключить. Все пороги/окна заданы **только** в конфиге, в коде не захардкожены, что соответствует правилам.  

**`utils.py`**

* Детектор всплесков корректно считает: `abs_change_pct` на окне и «контраст» = `abs_chg / avg|minutely return|` за 90-мин контекст. Однако **начало торговой сессии** всегда ставится в `win_start`. Это правильно для look-ahead, но **неверно для real-time** — там торговля должна начинаться в `t` (то есть в конце окна оценки). Нужно условно сдвигать старт. 

**`paper_trader.py`**

* Построение индекса из БД верно «нормализует» длину торговой сессии к `cfg.session_minutes` (демо 10), независимо от окна детектора, что согласуется с нашей парадигмой «детект 90-10, инференс 30-10». Но при построении «на лету» даты в `DataFrame` остаются строками, а ниже используется `.dt.floor("D")` — будет исключение. Нужно привести колонки времени к `datetime` (UTC) перед лимитированием. 
* Прочая логика (проверка покрытия контекста/сессии, расчёт направления по границе контекста, исполнение с комиссиями/проскальзыванием, консервативные метрики PF/Sharpe) — ок.

---

## Мини-патч (unified diff, ≤ 600 строк)

### 1) Правка семантики начала сессии при `use_lookahead=False` (stream-режим)

**Путь:** `third_party/rl-trading-binance/utils.py`
Идея: если `use_lookahead=False`, торговая сессия стартует в **t** (конец оценочного окна).

```diff
--- a/third_party/rl-trading-binance/utils.py
+++ b/third_party/rl-trading-binance/utils.py
@@ -329,18 +329,24 @@ def find_spike_windows(
-        if use_lookahead:
-            win_start = t
-            win_end = t + pd.Timedelta(minutes=window_minutes)
-        else:
-            win_start = t - pd.Timedelta(minutes=window_minutes)
-            win_end = t
+        if use_lookahead:
+            win_start = t
+            win_end = t + pd.Timedelta(minutes=window_minutes)
+        else:
+            # Реал-режим: оцениваем всплеск на [t-window, t], но торговать начинаем с момента t
+            win_start = t - pd.Timedelta(minutes=window_minutes)
+            win_end = t
@@
-        if abs_chg >= abs_change_threshold_pct and contrast >= contrast_min:
-            session_start = win_start  # начало сессии совпадает с окном оценки
-            session_end = win_end
+        if abs_chg >= abs_change_threshold_pct and contrast >= contrast_min:
+            # Начало торговой сессии:
+            #  - look-ahead=True  -> стартуем с начала окна (t)
+            #  - look-ahead=False -> стартуем с конца окна (t), чтобы не заглядывать в будущее
+            session_start = win_start if use_lookahead else win_end
+            # Предзаполним session_end длиной оценочного окна; фактическая длительность может быть переопределена конфигом
+            session_end = session_start + pd.Timedelta(minutes=window_minutes)
             out.append((ctx_start.to_pydatetime(), ctx_end.to_pydatetime(),
                         session_start.to_pydatetime(), session_end.to_pydatetime(), abs_chg))
```

> Это не меняет поведение офлайн-режима (где `use_lookahead=True`) и **чинит** реальное потоковое поведение (сессия начинается в момент детекции). Логика полностью согласуется с демо/фулл режимами, описанными в README. 

### 2) Приведение типов дат при построении индекса, чтобы не падал `.dt.floor("D")`

**Путь:** `third_party/rl-trading-binance/paper_trader.py`

```diff
--- a/third_party/rl-trading-binance/paper_trader.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@ -286,9 +286,15 @@ def main(cfg_path: Optional[str] = None, cfg_obj: Optional[MasterConfig] = None):
         idx = pd.DataFrame(rows)
         out_dir = os.path.dirname(cfg.index_csv)
         os.makedirs(out_dir, exist_ok=True)
-        idx.to_csv(cfg.index_csv, index=False)
+        # Сразу храним UTC-датавремена и используем их же ниже
+        for col in ["ctx_start","ctx_end","session_start","session_end"]:
+            idx[col] = pd.to_datetime(idx[col], utc=True)
+        idx.to_csv(cfg.index_csv, index=False)
     else:
         idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])
@@
-    if cfg.pt.cap_windows_per_symbol > 0:
+    if cfg.pt.cap_windows_per_symbol > 0:
         keep_rows = []
         for sym, g in idx.groupby("symbol"):
             g = g.sort_values("session_start")
             g["d"] = g["session_start"].dt.floor("D")
             g = g.groupby("d").head(cfg.pt.cap_windows_per_symbol).drop(columns=["d"])
             keep_rows.append(g)
         idx = pd.concat(keep_rows, ignore_index=True)
```

> Эта правка устраняет падение при `.dt.floor("D")` в случае свежесобранного индекса. Логика «сжимания» сигналов до N/день/тикер сохраняется без изменений. 

---

## Как это соотносится с правилами проекта

* **«Параметры — только из `configs/*.py`»**: все ключевые параметры (окна, пороги, look-ahead) лежат в `configs/alpha.py`; код лишь читает их — норм. 
* **Демо 30-10 по умолчанию**: выдержано; README чётко разводит Demo vs Full Pipeline (10-мин сессии, 30-мин контекст в демо). 
* **Артефакты/пути**: индекс по умолчанию пишется в `output/<config_name>/stream_backtest_index.csv`, что соответствует структуре из README. 

---

## Шаги / Риски

| Шаг | Действие                                                             | KPI / риск                                                              |
| --- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1   | Применить патч в `utils.py` (старт сессии для real-time)             | Потоковая семантика корректна; **снижается риск** «заглядывания вперёд» |
| 2   | Привести даты к `datetime` в `paper_trader.py` перед `.dt.floor`     | **Устранение падений** при лимитировании окон                           |
| 3   | Прогон `pytest -q` и минимальный прогон paper-trader на демо конфиге | Ожидаемо без регрессий; офлайн-режим не меняется                        |
| 4   | При real-time отключить `use_lookahead` в конфиге                    | Повторяем условия тренировки/валидации без утечки будущего              |

---

## Команды для локальной проверки (без CI)

```bash
# 1) Ветка и патч
git checkout -b feature/rt-session-start-fix
git apply --index changes.patch && git commit -m "fix(utils,paper_trader): correct RT session start; ensure datetime for daily cap"
git push -u origin feature/rt-session-start-fix

# 2) Тесты (если есть базовые)
pytest -q

# 3) Быстрый прогон индекса и paper-trader (демо 30-10)
python third_party/rl-trading-binance/paper_trader.py configs/alpha.py
```

> Напоминаю: при реальном потоке **обязательно** ставим `detector.use_lookahead=False` в конфиге, чтобы условия совпали с RT-торговлей; для репликации бэктеста оставляем `True`. 

---

[1]: https://github.com/FMProducer/prosperous_bot/commit/1e54f6b46d72d27c1833d23963566fa6e24b26d7 "docs: paper_trader.py fix · FMProducer/prosperous_bot@1e54f6b · GitHub"
