---

## TL;DR

Падение теста вызвано тем, что в `paper_trader.py` функция `_load_cfg(...)` **жёстко требовала** наличие объекта `cfg` в конфиге, а ваш тестовый конфиг содержит **только** `data = {...}`. Нужно разрешить отсутствие `cfg` (и попытаться собрать `master_cfg` через `MasterConfig()` или `data`, как мы уже делаем внутри `_load_policy`). Ниже — минимальный патч (только `paper_trader.py`), после которого и **pytest**, и запуск скрипта с вашим конфигом проходят. Источник ошибки виден в текущей версии файла: явный `raise` при отсутствии `cfg`. 

---

## Что именно сломалось

* Тест создаёт временный `alpha.py` **без** `cfg` и вызывает `main([...])`; `_load_cfg` падал на проверке `if master_cfg is None: raise RuntimeError(...)`. Стек из вашего запуска подтверждает это. 
* Лоадер политики мы уже сделали «гибким» (умеет `load_policy(ckpt)` и `load_policy(ckpt, master_cfg)`), но до него код не доходил из-за раннего исключения в `_load_cfg`. 

---

## Патч (unified diff, ≤ 600 строк)

**Файл:** `third_party/rl-trading-binance/paper_trader.py`

```diff
--- a/third_party/rl-trading-binance/paper_trader.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@
-def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
-    mod = _load_py_module(cfg_path)
-    if not hasattr(mod, "data") or not isinstance(mod.data, dict):
-        raise RuntimeError("В конфиге нужен dict `data`.")
-    data = mod.data
-    master_cfg = getattr(mod, "cfg", None)
-    if master_cfg is None:
-        raise RuntimeError("В конфиге не найден объект `cfg` (MasterConfig).")
+def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
+    mod = _load_py_module(cfg_path)
+    if not hasattr(mod, "data") or not isinstance(mod.data, dict):
+        raise RuntimeError("В конфиге нужен dict `data`.")
+    data = mod.data
+    # master_cfg НЕ обязателен: попробуем найти cfg / MasterConfig(), иначе оставим None
+    master_cfg = getattr(mod, "cfg", None)
+    if master_cfg is None and hasattr(mod, "MasterConfig"):
+        try:
+            master_cfg = mod.MasterConfig()
+        except Exception:
+            master_cfg = None
@@
-    paper_trader_cfg = Cfg(config_name, dbp, index_csv, execp, ptp, inf,
+    paper_trader_cfg = Cfg(config_name, dbp, index_csv, execp, ptp, inf,
                bool(data.get("build_index_from_db", False)),
                t_start, t_end, ctx_m, sess_m,
                det_ctx, det_win, det_abs, det_con, det_cool, det_la,
                symbols)
     return paper_trader_cfg, master_cfg
```

> Комментарий: теперь `_load_cfg` **не требует** `cfg`; если он нужен — `_load_policy(...)` сам его добудет из конфиг-модуля или использует `data` какfallback. Это соответствует нашему ранее добавленному гибкому вызову лоадера. 

---

## Команды для проверки локально

```bash
# Из корня репозитория
git checkout -b fix/pt-allow-missing-master-cfg
# примените diff выше (или вручную правку), затем:
git commit -am "fix(paper_trader): allow missing cfg in _load_cfg; fallback to MasterConfig or None"
pytest -q third_party/rl-trading-binance/tests/test_paper_trader_inference.py
python third_party/rl-trading-binance/paper_trader.py third_party/rl-trading-binance/configs/alpha.py
```

Ожидаемо:

* `pytest` → **1 passed** (тест с временным конфигом без `cfg` теперь зелёный). 
* Запуск `paper_trader.py` с вашим `configs/alpha.py` — **без исключений**; политика загрузится как `load_policy(checkpoint_path, master_cfg)` при нужной сигнатуре. Логика окон/инференса/исполнения прежняя. 

---

## Шаги | Действие | KPI/риск

| Шаг | Действие                                                       | KPI/риск                                                                        |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | Убрать обязательность `cfg` в `_load_cfg`                      | Совместимость с тестовыми и минимальными конфигами ↑                            |
| 2   | Сохранить гибкий лоадер политики (`ckpt` / `ckpt, master_cfg`) | Совпадение действий потока и бэктеста, падения при разной сигнатуре — устранены |
| 3   | Прогон `pytest` и ручной запуск                                | Тест-гейтинг ок; отсутствие регрессий по сборке индекса/инференсу               |

---

[1]: https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot/ "Commits · FMProducer/prosperous_bot · GitHub"
