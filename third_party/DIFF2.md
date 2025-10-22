# Repo-State Header (prosperous_bot)

* **Branch:** `prosperous_bot`
* **Latest commit:** `c2f965d` — *docs: paper_trader inference_adapter rebuild* (Oct 21, 2025). ([GitHub][1])
* **Commits page:** [https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot](https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot) ([GitHub][1])

## TL;DR

Тесты падают при динамической загрузке `paper_trader.py`: модуль ещё не зарегистрирован в `sys.modules` к моменту декоратора `@dataclass`, и `dataclasses` не может найти модуль по `cls.__module__` → `AttributeError: 'NoneType' object has no attribute '__dict__'`. Это видно в вашем логе `pytest` (см. трассировку к `dataclasses.py:757`). 
Исправление: **перед** `spec.loader.exec_module(mod)` положить модуль в `sys.modules[...]` (и то же для `config.py`). Ниже — минимальный `unified diff` для `third_party/rl-trading-binance/tests/test_exec_parity.py`.

---

## Патч (фикс импорта через `importlib`)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/tests/test_exec_parity.py
@@
-# -*- coding: utf-8 -*-
-"""
-Проверка паритета расчётов paper_trader c бэктест-логикой:
- - размер позиции: cfg.backtest.position_fraction
- - комиссии: cfg.market.transaction_fee
- - проскальзывание: cfg.market.slippage
-Тесты лёгкие, без БД/модели.
-"""
-import importlib.util
-import pathlib
-import math
-
-def _load_master_cfg():
-    """Загружаем config.py из той же директории проекта через importlib (дефисы в именах пакетов не мешают)."""
-    tests_dir = pathlib.Path(__file__).parent
-    cfg_path = tests_dir.parent / "config.py"
-    spec = importlib.util.spec_from_file_location("rtb_config", cfg_path.as_posix())
-    mod = importlib.util.module_from_spec(spec)
-    assert spec and spec.loader
-    spec.loader.exec_module(mod)  # type: ignore
-    # Пытаемся получить cfg; если его нет — пробуем MasterConfig()
-    if hasattr(mod, "cfg"):
-        return mod.cfg
-    if hasattr(mod, "MasterConfig"):
-        return mod.MasterConfig()
-    raise RuntimeError("Не найден ни `cfg`, ни `MasterConfig` в config.py")
+# -*- coding: utf-8 -*-
+"""
+Проверка паритета расчётов paper_trader c бэктест-логикой:
+ - размер позиции: cfg.backtest.position_fraction
+ - комиссии: cfg.market.transaction_fee
+ - проскальзывание: cfg.market.slippage
+Тесты лёгкие, без БД/модели.
+"""
+import importlib.util
+import pathlib
+import math
+import sys
+
+def _load_master_cfg():
+    """Загружаем config.py через importlib и РЕГИСТРИРУЕМ его в sys.modules до exec_module."""
+    tests_dir = pathlib.Path(__file__).parent
+    cfg_path = tests_dir.parent / "config.py"
+    name = "rtb_config_for_tests"
+    spec = importlib.util.spec_from_file_location(name, cfg_path.as_posix())
+    mod = importlib.util.module_from_spec(spec)
+    assert spec and spec.loader
+    sys.modules[name] = mod  # <-- важно для dataclasses и строковых аннотаций
+    spec.loader.exec_module(mod)  # type: ignore
+    # Пытаемся получить cfg; если его нет — пробуем MasterConfig()
+    if hasattr(mod, "cfg"):
+        return mod.cfg
+    if hasattr(mod, "MasterConfig"):
+        return mod.MasterConfig()
+    raise RuntimeError("Не найден ни `cfg`, ни `MasterConfig` в config.py")
@@
-def _load_paper_trader():
-    # Используем __file__, чтобы построить абсолютный путь к paper_trader.py
-    # Это делает тест независимым от текущей рабочей директории
-    tests_dir = pathlib.Path(__file__).parent
-    p = tests_dir.parent / "paper_trader.py"
-    spec = importlib.util.spec_from_file_location("paper_trader", p.as_posix())
-    mod = importlib.util.module_from_spec(spec)
-    assert spec and spec.loader
-    spec.loader.exec_module(mod)  # type: ignore
-    # пробрасываем MasterConfig
-    mod._MASTER_CFG = _load_master_cfg()
-    return mod
+def _load_paper_trader():
+    """Динамическая загрузка paper_trader.py с РЕГИСТРАЦИЕЙ в sys.modules до exec_module."""
+    tests_dir = pathlib.Path(__file__).parent
+    p = tests_dir.parent / "paper_trader.py"
+    name = "rtb_paper_trader_for_tests"
+    spec = importlib.util.spec_from_file_location(name, p.as_posix())
+    mod = importlib.util.module_from_spec(spec)
+    assert spec and spec.loader
+    sys.modules[name] = mod  # <-- критично: иначе dataclasses не найдёт модуль по cls.__module__
+    spec.loader.exec_module(mod)  # type: ignore
+    # пробрасываем MasterConfig
+    mod._MASTER_CFG = _load_master_cfg()
+    return mod
*** End Patch
```

### Почему это решает проблему

* В вашем логе видно, что падение происходит в `dataclasses._is_type`, где выполняется `sys.modules.get(cls.__module__)` и возвращается `None`, после чего обращение к `.__dict__` валится с `AttributeError` (стр. `dataclasses.py:757`). 
* Причина — модуль, из которого объявлен `@dataclass`, ещё не зарегистрирован в `sys.modules`. Регистрация **до** `exec_module` делает модуль доступным в момент декорирования, и `dataclasses` корректно резолвит строковые аннотации/метки `KW_ONLY`.

---

## Команды проверки

```bash
# Из корня репозитория
pytest -q third_party/rl-trading-binance/tests/test_exec_parity.py
```

Ожидаемое: тесты проходят этап импорта `paper_trader.py`. Если всплывут новые ошибки (например, связанные с типами/Optional/Any внутри `paper_trader.py`), пришлите свежий лог — подготовлю следующий точечный diff.

[1]: https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot/ "Commits · FMProducer/prosperous_bot · GitHub"
