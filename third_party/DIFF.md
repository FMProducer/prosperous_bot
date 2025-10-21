---

## TL;DR

1. Синтаксис и базовая логика во всех трёх файлах — **OK** (parse/импорт проходит).
2. **Но:** текущая версия `paper_trader.py` всё ещё торгует по эвристике *Follow-Context* и **не использует инференс модели** — в точности против вашего пункта «убрать эвристику и подключить `data.inference.policy_loader / checkpoint_path`». Это видно по коду определения `side` через знак изменения цены в контексте и отсутствию вызовов policy (`predict_side`) в основном цикле. 
3. Конфиг `alpha.py` — корректно описывает секцию `data.inference` (loader, checkpoint, `strict=True`), и совместим с демо-режимом 30-10, look-ahead для детектора 90-10 включён (для паритета с бэктестом). 
4. Тест `test_paper_trader_inference.py` подменяет `inference_adapter:load_policy`, но **не проверяет**, что инференс действительно вызван — поэтому тест пройдёт даже при эвристике. 
5. Соответствие правилам проекта («конфиги только из `configs/*.py`», отчёты в `output/<config>/`) выдержано; см. системные требования проекта.

Итого: чтобы потоковая логика совпадала с бэктестом **и по окнам, и по действиям**, нужно внести небольшой патч в `paper_trader.py` (интегрировать инференс) и усилить тест.

---

## Проверка по файлам (факты и риски)

**`third_party/rl-trading-binance/paper_trader.py`**

* Сейчас «направление» сделки берётся из контекста (знак доходности), а не из модели; в коде это прямая установка `side = "BUY" if ... else "SELL"` и выбор цен через `_get_close_near(...)`. 
* Вспомогательный `_get_close_near` использует `method="nearest"` — это может *редко* выбирать будущий бар при дырках в минутках (микро-front-run). Для честности на границах сессии лучше брать `≤ ts` (asof/pad). 
* Докстринг говорит «Стратегия по умолчанию: Follow-Context», что расходится с вашим требованием «эвристику убрать». (Должно быть: «стратегия — инференс модели; без политики — пропуск».) 

**`third_party/rl-trading-binance/utils.py`**

* Детектор всплесков (`find_spike_windows`) соответствует описанию: окно контраста до скачка, окно оценки скачка (по абсолютному изменению), кулдаун и опция `use_lookahead` для паритета с бэктестом. (Окна 90/10 из конфига — поддержаны.) *(Контент не показан здесь, сверялось с загруженным файлом)*.

**`third_party/rl-trading-binance/configs/alpha.py`**

* Блок `data.inference` присутствует: `policy_loader: "inference_adapter:load_policy"`, `checkpoint_path: ...`, `strict: True`. Базовый режим 30-10; детектор 90-10 с look-ahead — как в демо и README. 
* Политика проекта «конфиги в `configs/` и артефакты в `output/<config>`» соблюдена. 

**`third_party/rl-trading-binance/tests/test_paper_trader_inference.py`**

* Тест создает стаб провайдера БД и стаб `inference_adapter:load_policy`, но не утверждает факт вызова инференса, лишь проверяет, что артефакты создались — этого недостаточно для «тест-гейтинга» интеграции. 

---

## Что именно не так и как исправить (минимальный патч)

Ниже — целевые правки: (а) подключить политику, (б) убрать эвристику из основного потока, (в) поправить честность получения цены на границах.

### ✅ Unified diff (≤ 600 строк)

```diff
--- a/third_party/rl-trading-binance/paper_trader.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@
- только на период сессии. Стратегия по умолчанию: Follow-Context.
+ только на период сессии. Стратегия: инференс модели (policy.predict_side),
+ эвристика Follow-Context удалена; при отсутствии/ошибке политики — пропуск окна.
@@
-from typing import Dict, Iterable, Iterator, List, Optional, Tuple
+from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Protocol
@@
-def _get_close_near(df: pd.DataFrame, ts: pd.Timestamp) -> float:
-    """Безопасно получить цену close вблизи ts (UTC, минутные бары)."""
-    if ts in df.index:
-        return float(df.loc[ts, "close"])
-    # ближайший бар
-    i = df.index.get_indexer([ts], method="nearest")[0]
-    return float(df.iloc[i]["close"])
+def _get_close_near(df: pd.DataFrame, ts: pd.Timestamp, how: str = "pad") -> float:
+    """Честно получить close на/до ts (UTC, минутные бары).
+    how: "pad" → берём последний бар ≤ ts; "nearest" оставлен для совместимости."""
+    if ts in df.index:
+        return float(df.loc[ts, "close"])
+    method = how if how in ("pad", "nearest", "backfill") else "pad"
+    i = df.index.get_indexer([ts], method=method)[0]
+    return float(df.iloc[i]["close"])
@@
-# ------------------------------ Strategy ---------------------------
-# Baseline: Follow-Context — направление = sign(close(ctx_end)-close(ctx_start))
-
-def _direction_from_ctx(row: pd.Series) -> int:
-    # Предполагаем, что в index CSV нет direction. Направление восстановим из ret_ctx, если доступно,
-    # иначе по sign(abs_change + эвристика через session цены).
-    # В базовой реализации — вычислим позже по реальным баррам session_start-ctx_end (см. ниже).
-    return 0  # placeholder; определим после загрузки цен
+# ------------------------------ Inference --------------------------
+class _Policy(Protocol):
+    def predict_side(self, df_ctx: pd.DataFrame) -> str: ...
+
+def _load_policy(loader_path: Optional[str], checkpoint_path: Optional[str]) -> Optional[_Policy]:
+    if not loader_path:
+        return None
+    if ":" not in loader_path:
+        raise RuntimeError("`data.inference.policy_loader` должен быть 'module:function'.")
+    mod_path, fn_name = loader_path.split(":", 1)
+    mod = importlib.import_module(mod_path)
+    if not hasattr(mod, fn_name):
+        raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
+    loader = getattr(mod, fn_name)
+    return loader(checkpoint_path)
@@
 @dataclass
 class Cfg:
@@
-    # детектор
+    # детектор
     det_context: int
     det_window: int
@@
     det_use_lookahead: bool
     # опционально: список тикеров для сканирования
     symbols: List[str]
+    # inference
+    inf_loader: Optional[str]
+    inf_checkpoint: Optional[str]
+    inf_strict: bool
@@
 def _load_cfg(cfg_path: str) -> Cfg:
@@
-    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
+    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
     ex_d = data.get("exec", {})
+    inf_d = data.get("inference", {})
@@
     execp = ExecParams(
@@
-    return Cfg(
+    return Cfg(
         config_name=config_name,
         db_provider_path=dbp,
         index_csv=index_csv,
         exec=execp,
         pt=PTParams(mode=str(pt_d.get("mode", "asap")), cap_windows_per_symbol=int(pt_d.get("cap_windows_per_symbol", 0))),
@@
         det_cooldown=int(det.get("cooldown_minutes", 60)),
         det_use_lookahead=bool(det.get("use_lookahead", True)),
         symbols=list(data.get("symbols", [])),
+        inf_loader=inf_d.get("policy_loader"),
+        inf_checkpoint=inf_d.get("checkpoint_path"),
+        inf_strict=bool(inf_d.get("strict", True)),
     )
@@
 def main(argv: List[str]) -> int:
@@
-    provider = _load_db_provider(cfg.db_provider_path)
+    provider = _load_db_provider(cfg.db_provider_path)
+    policy: Optional[_Policy] = _load_policy(cfg.inf_loader, cfg.inf_checkpoint)
@@
-        # Направление из контекста: sign(close(ctx_end-1m) - close(ctx_start))
-        ctx_end_minus = ctx_end - pd.Timedelta(minutes=1)
-        px_ctx_start = _get_close_near(df, pd.Timestamp(ctx_start))
-        px_ctx_endm1 = _get_close_near(df, pd.Timestamp(ctx_end_minus))
-        side = "BUY" if (px_ctx_endm1 - px_ctx_start) >= 0 else "SELL"
+        # Направление определяет модель по контексту
+        if policy is None:
+            if cfg.inf_strict:
+                continue  # строгий режим: без политики окно пропускаем
+            else:
+                raise RuntimeError("Policy не загружена, а inf_strict=False запрещает эвристику.")
+        df_ctx = df.loc[pd.Timestamp(ctx_start):pd.Timestamp(ctx_end - pd.Timedelta(minutes=1))]
+        side = getattr(policy, "predict_side")(df_ctx)
+        if side not in ("BUY", "SELL"):
+            if cfg.inf_strict:
+                continue
+            else:
+                raise RuntimeError("predict_side вернул некорректное значение.")
@@
-        first_px = _get_close_near(df, pd.Timestamp(ses_start))
+        first_px = _get_close_near(df, pd.Timestamp(ses_start), how="pad")
@@
-        last_px = _get_close_near(df, pd.Timestamp(ses_end))
+        last_px = _get_close_near(df, pd.Timestamp(ses_end), how="pad")
```

### ✅ Усиление теста (проверяем, что инференс действительно вызван)

```diff
--- a/third_party/rl-trading-binance/tests/test_paper_trader_inference.py
+++ b/third_party/rl-trading-binance/tests/test_paper_trader_inference.py
@@
-class _DummyPolicy:
+class _DummyPolicy:
+    calls = 0
     def predict_side(self, df_ctx):
-        # Всегда BUY
+        _DummyPolicy.calls += 1
         return "BUY"
 def load_policy(ckpt_path):
     return _DummyPolicy()
@@
     main = paper_trader.main
@@
     main(["paper_trader.py", cfg_file.as_posix()])
-    assert out_dir.exists() and (out_dir/"trades.csv").exists() and (out_dir/"metrics.json").exists()
+    assert out_dir.exists() and (out_dir/"trades.csv").exists() and (out_dir/"metrics.json").exists()
+    # Критично: модель должна быть вызвана хотя бы раз
+    assert _DummyPolicy.calls > 0
```

> Почему именно так:
>
> * Политика загружается через `module:function` из `data.inference` (строго по вашим требованиям), а эвристика *полностью убрана*.
> * `how="pad"` на границах сессии исключает случайный выбор будущего бара (честность исполнения). 
> * Тест теперь фейлится, если инференс не используется — тем самым «тест-гейтинг» действительно покрывает интеграцию, как требует SYSTEM_PROMPT. 

---

## Команды для локального запуска (pytest + артефакты)

```bash
# из корня репо
pytest -q third_party/rl-trading-binance/tests/test_paper_trader_inference.py

# пример запуска трейдера (демо 30-10)
python third_party/rl-trading-binance/paper_trader.py third_party/rl-trading-binance/configs/alpha.py
```

Артефакты должны появиться в `third_party/rl-trading-binance/output/alpha/{trades.csv,metrics.json}` — как требует POLICY. 

---

## Шаг | Действие | KPI/риск

| Шаг | Действие                                           | KPI/риск                                                                |
| --- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| 1   | Статическая проверка синтаксиса (AST)              | Ошибок нет; тесты запускаемы                                            |
| 2   | Верификация конфига `data.inference`               | Совместим с демо 30-10; `strict=True` — безопасно (пропуск без модели)  |
| 3   | Интеграция политики в `paper_trader.py`            | Совпадение действий с бэктестом; исключаем дрейф                        |
| 4   | Удаление эвристики Follow-Context                  | Исключаем систематическое смещение; ↑ честность                         |
| 5   | Исправление `_get_close_near` на `pad` на границах | ↓ риск «заглядывания вперёд»                                            |
| 6   | Усиление теста: assert вызова модели               | Тест-гейтинг интеграции; ↓ риск регрессий                               |
| 7   | Отчётность в `output/<config>`                     | Соответствие POLICY; удобный аудит артефактов                           |

---

## Вывод

* **Корректность текущих правок:** базовая логика и формат артефактов — **OK**.
* **Несоответствие требования:** инференс **не подключён** в `paper_trader.py`, эвристика всё ещё активна. Предложенный патч решает это и усиливает тест-гейтинг.
* После применения патча потоковый трейдер будет работать **в тех же условиях**, что и бэктест: те же окна (90-10 детекция → 30-10 сессия), та же политика действий модели — как и задумывалось в README/Policy.

[1]: https://github.com/FMProducer/prosperous_bot/commit/1e54f6b46d72d27c1833d23963566fa6e24b26d7 "docs: paper_trader.py fix · FMProducer/prosperous_bot@1e54f6b · GitHub"
