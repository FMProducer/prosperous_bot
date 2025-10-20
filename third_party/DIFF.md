# Repo-State Header

**Ветка (default):** `prosperous_bot`
**Примечание:** в этой сессии не смог надёжно подтвердить актуальный SHA-1 HEAD по REPO_URL. Действую по предоставленным локальным файлам и артефактам; SHA-1 зафиксируем при подготовке PR.

---

## TL;DR

Ваши логи падают на `db_provider.py` из-за запроса столбцов, которых **нет** в таблице (`num_trades`, иногда `quote_asset_volume`). Это не про модель/трейдер — это чисто **SQL-схема**.
Сделал точечный фикс провайдера: динамически проверяем наличие колонок и собираем `SELECT` только из существующих; расчёт `volume_weighted_average` выполняем **только** при наличии `quote_asset_volume`.
Дополнительно устраняю потенциальную **циклическую зависимость**: `inference_adapter.py` импортирует `paper_trader.Cfg`, что может «закусить» импорт при загрузке политики. Патч убирает этот импорт и использует уже загруженный `MasterConfig` (он у вас есть в `paper_trader.py`).

Ниже — причины и минимальные патчи.

---

## Что именно сломалось и где

1. **Ошибка SQL (UndefinedColumn)**
   В логе видно: запрос включает `num_trades`, а в вашей `klines_1m` этого столбца нет → `psycopg2.errors.UndefinedColumn`. Источник — текущий запрос в `db_provider.py`: он всегда тянет
   `..., base_volume, num_trades, quote_asset_volume ...` 

2. **Риск кругового импорта**
   `inference_adapter.py` делает `from paper_trader import Cfg`. А `paper_trader.py` динамически импортирует адаптер для загрузки политики. Это создаёт цикл *paper_trader → inference_adapter → paper_trader*. Сегодня повезло, но это хрупко. 

---

## Патч 1 — безопасный SQL в провайдере БД

**Файл:** `third_party/rl-trading-binance/db_provider.py` (ваша версия) 

```diff
*** a/third_party/rl-trading-binance/db_provider.py
--- b/third_party/rl-trading-binance/db_provider.py
@@
-    for symbol in symbols:
-        query = "SELECT open_time_ms, open_price, high_price, low_price, close_price, base_volume, num_trades, quote_asset_volume FROM klines_1m WHERE symbol = :symbol AND open_time_ms >= :start_ms AND open_time_ms <= :end_ms ORDER BY open_time_ms"
+    # Определим доступные колонки в таблице (один раз на соединение)
+    try:
+        with engine.connect() as connection:
+            cols_res = connection.execute(
+                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'klines_1m'")
+            )
+            available_cols = {row[0] for row in cols_res}
+    except Exception as e:
+        raise RuntimeError(f"Could not inspect table columns: {e}")
+
+    base_cols = ["open_time_ms", "open_price", "high_price", "low_price", "close_price", "base_volume"]
+    opt_cols = []
+    # Добавим только реально существующие «опциональные» поля
+    if "quote_asset_volume" in available_cols:
+        opt_cols.append("quote_asset_volume")
+    # num_trades не обязателен — используем только если есть
+    if "num_trades" in available_cols:
+        opt_cols.append("num_trades")
+
+    select_cols = base_cols + opt_cols
+    select_clause = ", ".join(select_cols)
+
+    for symbol in symbols:
+        query = (
+            f"SELECT {select_clause} FROM klines_1m "
+            "WHERE symbol = :symbol AND open_time_ms >= :start_ms AND open_time_ms <= :end_ms "
+            "ORDER BY open_time_ms"
+        )
@@
-        df.rename(columns={
+        df.rename(columns={
             'open_price': 'open',
             'high_price': 'high',
             'low_price': 'low',
             'close_price': 'close',
             'base_volume': 'volume'
         }, inplace=True)
-
-        df['volume_weighted_average'] = df['quote_asset_volume'] / (df['volume'] + 1e-9)
-        df.drop(columns=['quote_asset_volume'], inplace=True)
+        # Рассчитываем VWAP-подобную метрику, только если есть quote_asset_volume
+        if 'quote_asset_volume' in df.columns:
+            df['volume_weighted_average'] = df['quote_asset_volume'] / (df['volume'] + 1e-9)
+            df.drop(columns=['quote_asset_volume'], inplace=True)
 
         yield (symbol, df)
```

**Эффект:**

* Больше **нет** запросов к несуществующим колонкам → исчезают `UndefinedColumn` для `num_trades`/`quote_asset_volume`.
* Формат отдачи в трейдер прежний (обязательные `open/high/low/close/volume`), расчёт `volume_weighted_average` — если есть данные.

---

## Патч 2 — убрать потенциальный circular import в адаптере

**Файл:** `third_party/rl-trading-binance/inference_adapter.py` (ваша версия) 
И **согласовать сигнатуры** с тем, как вы вызываете загрузчик в `paper_trader.py` (там вы передаёте `MasterConfig` как `master_cfg`). 

```diff
*** a/third_party/rl-trading-binance/inference_adapter.py
--- b/third_party/rl-trading-binance/inference_adapter.py
@@
-import torch
-import pandas as pd
-import numpy as np
-import datetime as dt
-
-from model import DuelingQNetwork
-from paper_trader import Cfg
-from utils import calculate_normalization_stats, apply_normalization
-from config import MasterConfig
+import torch
+import pandas as pd
+import numpy as np
+import datetime as dt
+
+from model import DuelingQNetwork
+from utils import calculate_normalization_stats, apply_normalization
+from config import MasterConfig
@@
-def load_policy(ckpt_path: str, cfg: Cfg):
-    # This is a simplified way to get the master config.
-    # It assumes that the paper_trader config has enough information.
-    master_cfg = MasterConfig()
-    # We need to populate master_cfg with values from cfg if they exist.
-    # For now, we will use the defaults from config.py and alpha.py,
-    # as they are loaded in train.py.
-    
-    # A better way would be to load the config file that was used for training.
-    # Assuming the config is compatible.
+def load_policy(ckpt_path: str, master_cfg: MasterConfig):
+    # Используем master_cfg, который передаёт paper_trader (без импортов paper_trader → нет цикла).
@@
-    # This is the hardest part: getting the normalization stats.
-    # For now, I will create dummy stats.
-    # In a real scenario, these should be loaded from a file saved during training.
+    # TODO: подставить реальные нормировочные статистики, сохранённые при обучении
     stats = {
         "means": {ch: 0.0 for ch in master_cfg.data.data_channels},
         "stds": {ch: 1.0 for ch in master_cfg.data.data_channels},
     }
 
-    return DuelingQPolicy(model, stats, master_cfg)
+    return DuelingQPolicy(model, stats, master_cfg)
```

**Эффект:**

* `inference_adapter.py` больше **не импортирует** `paper_trader.Cfg` → исключаем круговой импорт.
* Подпись `load_policy()` совпадает с используемой в `paper_trader.py` (вы туда уже передаёте `master_cfg`). 

---

## Что делать сейчас (быстрые шаги)

| Шаг | Действие                                                                                  | KPI/риск                                                         |
| --- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1   | Применить патч к `db_provider.py`                                                         | Ошибка `UndefinedColumn` исчезает                                |
| 2   | Прогнать `paper_trader.py` ещё раз                                                        | Прогон завершается; скорость вернётся к прежней                  |
| 3   | Применить патч к `inference_adapter.py`                                                   | Исключён риск кругового импорта; поведение инференса не меняется |
| 4   | (Опционально) Снять метрику «сколько окон было пропущено из-за неполного покрытия данных» | Контроль качества БД/окон                                        |

**Команды (локально):**

```bash
git checkout -b fix/db-provider-safe-select
# сохраните diff(ы) как changes.patch
git apply --index changes.patch
git commit -m "fix(db_provider): safe SELECT only existing columns; VWAP only if quote_asset_volume present"

# если делали и правку inference_adapter:
git commit -a -m "chore(inference): remove circular import; align loader signature with paper_trader"

# прогон
$env:PGPASSWORD="9691"
python third_party/rl-trading-binance/paper_trader.py third_party/rl-trading-binance/configs/alpha.py
```

---

## Пояснения и соответствие регламенту

* Все параметры остаются **в конфиге**; мы правили только код провайдера/адаптера.
* Формат артефактов `paper_trader` не менялся (`output/<config>/trades.csv`, `metrics.json`).
* Изменения минимальны и безопасны: не затрагивают расчёт PnL/комиссий/времени, только читают корректные колонки и исключают потенциальную цикличность импортов.

Если хотите, дополнительно добавлю в `paper_trader.py` счётчик и лог строк вида:
`[paper_trader] skipped windows due to missing bars: N` — это поможет быстро видеть, если БД иногда «рвётся» на отдельных окнах/символах.
