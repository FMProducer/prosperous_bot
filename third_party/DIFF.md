# Repo-State Header (prosperous_bot)

* **Ветка:** `prosperous_bot`
* **Удалённый репозиторий не верифицирован онлайн** (работаю по вашим локальным файлам).
* **Проверенный файл:** `third_party/rl-trading-binance/paper_trader.py` (актуальная версия из вашего сообщения). 

## TL;DR

Ваш diff применён **корректно** по главной цели: теперь минутные бары подгружаются **один раз на тикер** и используются из кэша; вызов `provider(...)` **больше не выполняется внутри цикла по окнам**, что устраняет `too many clients` и резко ускоряет прогон. Дополнительно нашёл 2 момента:

1. В `_load_cfg(...)` вы изменили семантику `build_index_from_db`: если в конфиге стоит просто `True` (булево), **индекс не будет строиться** (флаг читается только из словаря с `enabled`). Это регресс и его стоит поправить одной строкой. 
2. В текущем файле **нет режима `index_mode="sliding"`** — построение индекса по-прежнему идёт только через `find_spike_windows(...)`. Если вы ожидали «скользящее» окно на каждом баре, его здесь ещё нет. 

---

## Что проверил в вашем `paper_trader.py`

* **Кэш минуток на символ**: блок `FEED_CACHE` собирает диапазоны `[min(ctx_start), max(session_end)]` по каждому символу, запрашивает провайдера **один раз** и кладёт DataFrame в словарь. Далее в главном цикле берётся `df = FEED_CACHE[sym]`. Это ровно то, что нужно. 
* **Отсутствие DB-запросов в цикле окон**: внутри `for _, row in tqdm(idx.iterrows() ...):` больше нет вызова `provider(...)`; только срезы по уже загруженному `df`. ✔️ 
* **Границы покрытия** окна проверяются (первый бар ≤ `ctx_start`, последний ≥ `session_end - 1 мин`) — как и раньше. ✔️ 
* **Нормстатистика/инференс/исполнение** не затронуты патчем — паритет с прежней логикой сохранён. ✔️ 

---

## Мини-фикс №1 (логика `build_index_from_db`)

Сейчас:

```python
build_idx_cfg = data.get("build_index_from_db", False)
build_idx_enabled = isinstance(build_idx_cfg, dict) and build_idx_cfg.get("enabled", False)
symbols = list(build_idx_cfg.get("symbols", [])) if isinstance(build_idx_cfg, dict) else list(data.get("symbols", []))
```

Если в конфиге просто `build_index_from_db=True`, то `build_idx_enabled` станет `False`. Исправление:

```diff
-    build_idx_cfg = data.get("build_index_from_db", False)
-    build_idx_enabled = isinstance(build_idx_cfg, dict) and build_idx_cfg.get("enabled", False)
-    symbols = list(build_idx_cfg.get("symbols", [])) if isinstance(build_idx_cfg, dict) else list(data.get("symbols", []))
+    build_idx_cfg = data.get("build_index_from_db", False)
+    if isinstance(build_idx_cfg, dict):
+        build_idx_enabled = bool(build_idx_cfg.get("enabled", False))
+        symbols = list(build_idx_cfg.get("symbols", []))
+    else:
+        build_idx_enabled = bool(build_idx_cfg)          # ← поддерживаем булево True
+        symbols = list(data.get("symbols", []))
```

Это вернёт прежнее ожидаемое поведение: булево `True` просто включает построение индекса. 

---

## Напоминание по `index_mode="sliding"`

В предоставленном файле **нет** переключателя режима: индекс строится только так:

```python
wins = find_spike_windows(df, ... cfg.det_* ...)
```

Если ваша цель — «как в бэктесте» (скользящее 10-мин окно на каждом новом баре), нужно добавить `sliding`-режим в месте построения индекса (я давал минимальный diff ранее). Сейчас это **не внедрено**, поэтому количество окон по-прежнему определяется детектором всплесков. 

---

## Чек-лист

| Шаг | Действие                                                                                                           | KPI/риск                                          |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 1   | Примените мини-фикс `build_index_from_db` (выше)                                                                   | Гарантированно строим индекс, если в конфиге True |
| 2   | Перезапустите `paper_trader.py`                                                                                    | Ожидаемо без `too many clients`; быстрее прогон   |
| 3   | Если нужно «скользящее» поведение — добавьте `index_mode="sliding"` (как просили ранее). Сейчас этого в файле нет. | Паритет с бэктестом по стартам сессий             |

Если хотите — могу сразу выдать **чистый минимальный diff** только на добавление `sliding` без других правок кода.
