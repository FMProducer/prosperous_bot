TL;DR: В вашей версии `backtest_engine.py` флаг `cfg.detector.use_lookahead` реально подставляется в SQL через параметр `:use_lookahead`, но именно такая форма CASE в SQL (булевый плейсхолдер внутри CASE) иногда ведёт себя неочевидно и «залипает» на одну ветку. Чтобы гарантированно исключить «подглядку в будущее», предложил безусловно разветвить SQL на уровне Python (две разные строки запроса для `True/False`) и залогировать выбранный режим. Это убирает риск. Ниже — минимальный патч.

**Repo-State Header (Ultra-strict):**
Ветка: `prosperous_bot` (default). Полный SHA последнего коммита получить из текущего окружения не удалось (ограничение доступа к API). Репозиторий: [https://github.com/FMProducer/prosperous_bot](https://github.com/FMProducer/prosperous_bot). Требования проекта и режим Ultra-strict — см. SYSTEM_PROMPT/README    

| Шаг | Что сделал                                                                                                     | KPI/риск                                                                                                                        |
| --- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Проверил логику `use_lookahead` в `load_from_db_and_prepare_signals`                                           | Обнаружено: CASE WHEN :use_lookahead THEN LEAD(...) ELSE LAG(...) — теоретически верно, но может «залипать» (DB/driver нюанс)   |
| 2   | Подтвердил, что флаг в `alpha.py` выставлен `False` и используется `data_source="find_spikes"`                 | Конфиг корректный; параметр действительно должен выключать lookahead                                                            |
| 3   | Предложил патч: формировать ДВА варианта SQL без параметризации ветки CASE; добавить явный лог `use_lookahead` | Исключаем неоднозначность, получаем воспроизводимое поведение                                                                   |

# Почему сейчас «как будто не выключается»

* CASE с параметром `:use_lookahead` корректен синтаксически, но на практике встречается поведение, когда драйвер/планировщик запроса компилирует план с одной веткой и переиспользует его (особенно при повторных вызовах одинакового текста запроса). Вы это видите как «переключатель не влияет». Перенос выбора ветки из SQL в Python гарантированно устраняет эффект. Кодовая база вокруг (формирование сессий, вызовы Env) при этом не меняется  .

---

# Unified diff (минимально необходимый)

**Файл:** `third_party/rl-trading-binance/backtest_engine.py`  

```diff
--- a/third_party/rl-trading-binance/backtest_engine.py
+++ b/third_party/rl-trading-binance/backtest_engine.py
@@ -133,6 +133,8 @@ def load_from_db_and_prepare_signals(cfg: MasterConfig) -> List[Tuple[Tuple[str,
     engine = create_engine(cfg.db.dsn)
 
     if isinstance(cfg.paper.symbols, list) and cfg.paper.symbols:
         symbols = cfg.paper.symbols
@@ -163,35 +165,63 @@ def load_from_db_and_prepare_signals(cfg: MasterConfig) -> List[Tuple[Tuple[str,
     logging.info(f"Scanning for spike signals from {start_utc} to {end_utc} for {len(symbols)} symbols...")
     try:
         with engine.connect() as conn:
             from sqlalchemy import text
-            # This SQL query uses window functions to find spikes directly in the database.
-            # It's much faster than loading all data into Python.
             detector_cfg = cfg.detector
-            query = text(f"""
-            WITH minute_returns AS (
-                SELECT
-                    ts,
-                    symbol,
-                    close,
-                    (close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS ret
-                FROM v_klines_1m_npz
-                WHERE symbol = ANY(:symbols) AND ts >= :start_ts AND ts < :end_ts
-            ),
-            rolling_stats AS (
-                SELECT
-                    ts,
-                    symbol,
-                    -- NEW: Conditional logic for lookahead
-                    CASE
-                        WHEN :use_lookahead THEN
-                            (LEAD(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts) / close) - 1 -- Заглядываем вперед
-                        ELSE
-                            (close / LAG(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts)) - 1 -- Смотрим только в прошлое
-                    END AS abs_change,
-                    CASE
-                        WHEN :use_lookahead THEN
-                            AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN {detector_cfg.context_minutes} PRECEDING AND 1 PRECEDING) -- Контекст для lookahead
-                        ELSE
-                            AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN {detector_cfg.context_minutes + detector_cfg.window_minutes} PRECEDING AND {detector_cfg.window_minutes} PRECEDING) -- Контекст для "честного" режима
-                    END AS avg_abs_ret_pre
-                FROM minute_returns
-            )
-            SELECT ts, symbol
-            FROM rolling_stats
-            WHERE
-                ABS(abs_change) * 100.0 >= :abs_change_pct AND
-                (ABS(abs_change) / (avg_abs_ret_pre + 1e-9)) >= :contrast_min
-            ORDER BY ts, symbol;
-            """)
+            logging.info(f"[Detector] use_lookahead={detector_cfg.use_lookahead} "
+                         f"window_minutes={detector_cfg.window_minutes} "
+                         f"context_minutes={detector_cfg.context_minutes}")
+
+            # ДВА ВАРИАНТА SQL: без параметризации булевой ветки
+            if detector_cfg.use_lookahead:
+                query = text(f"""
+                WITH minute_returns AS (
+                    SELECT
+                        ts,
+                        symbol,
+                        close,
+                        (close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS ret
+                    FROM v_klines_1m_npz
+                    WHERE symbol = ANY(:symbols) AND ts >= :start_ts AND ts < :end_ts
+                ),
+                rolling_stats AS (
+                    SELECT
+                        ts,
+                        symbol,
+                        (LEAD(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts) / close) - 1 AS abs_change,
+                        AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts
+                            ROWS BETWEEN {detector_cfg.context_minutes} PRECEDING AND 1 PRECEDING) AS avg_abs_ret_pre
+                    FROM minute_returns
+                )
+                SELECT ts, symbol
+                FROM rolling_stats
+                WHERE
+                    ABS(abs_change) * 100.0 >= :abs_change_pct AND
+                    (ABS(abs_change) / (avg_abs_ret_pre + 1e-9)) >= :contrast_min
+                ORDER BY ts, symbol;
+                """)
+            else:
+                query = text(f"""
+                WITH minute_returns AS (
+                    SELECT
+                        ts,
+                        symbol,
+                        close,
+                        (close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS ret
+                    FROM v_klines_1m_npz
+                    WHERE symbol = ANY(:symbols) AND ts >= :start_ts AND ts < :end_ts
+                ),
+                rolling_stats AS (
+                    SELECT
+                        ts,
+                        symbol,
+                        (close / LAG(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS abs_change,
+                        AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts
+                            ROWS BETWEEN {detector_cfg.context_minutes + detector_cfg.window_minutes} PRECEDING
+                            AND {detector_cfg.window_minutes} PRECEDING) AS avg_abs_ret_pre
+                    FROM minute_returns
+                )
+                SELECT ts, symbol
+                FROM rolling_stats
+                WHERE
+                    ABS(abs_change) * 100.0 >= :abs_change_pct AND
+                    (ABS(abs_change) / (avg_abs_ret_pre + 1e-9)) >= :contrast_min
+                ORDER BY ts, symbol;
+                """)
 
             found_spikes_df = pd.read_sql(query, conn, params={
                 "symbols": symbols,
                 "start_ts": int(pd.to_datetime(start_utc).timestamp() * 1000),
                 "end_ts": int(pd.to_datetime(end_utc).timestamp() * 1000),
-                "abs_change_pct": detector_cfg.abs_change_pct,
-                "contrast_min": detector_cfg.contrast_min,
-                "use_lookahead": detector_cfg.use_lookahead,
+                "abs_change_pct": detector_cfg.abs_change_pct,
+                "contrast_min": detector_cfg.contrast_min,
             })
             found_spikes_df['ts'] = pd.to_datetime(found_spikes_df['ts'], unit='ms', utc=True)
```

---

# Что и как проверить локально

1. Запустите два раза, меняя только `cfg.detector.use_lookahead` в `configs/alpha.py` на `False`/`True`. Логи теперь явно покажут выбранный режим:

```
[Detector] use_lookahead=False window_minutes=10 context_minutes=30
```

И количество найденных сигналов должно **заметно отличаться** между режимами (обычно с lookahead их больше/«ярче»).

2. Убедитесь, что в `alpha.py` стоит:

```python
cfg.backtest.data_source = "find_spikes"
cfg.detector.use_lookahead = False
```

и правильно задан `time_range`  .

3. Чтобы исключить прочие источники «заглядывания вперёд»:

* Нормализация берётся из заранее сохранённых stats (`norm_stats.json`) — в этом месте нет локальной подгонки по текущей сессии (нет «per-sequence» нормализации)  .
* Сессия загружается как `[-pre_signal_len .. +post_signal_len]`, но наблюдение в среде должно идти пошагово; сам выбор сигналов мы уже сделали «честным». Если подозрения сохранятся, включим отладочный лог в `TradingEnvironment`, чтобы печатать индекс последней доступной свечи в наблюдении на каждом шаге — это быстро добавляется, но в представленных файлах среды нет  .

Если после этого обновления разницы не будет — значит проблема **не** в детекторе, а в самой среде/агенте (например, если наблюдение формируется из «полной сессии» сразу). Тогда предложу точечный патч в `trading_environment.py` (логика нарезки окна на каждом шаге) и дам юнит-тест, который детектирует «утечку будущего» на синтетической последовательности.
