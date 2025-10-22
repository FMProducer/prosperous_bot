# TL;DR

Готов патч, который **синхронизирует поведение `paper_trader.py` с `backtest_engine.py`**:

1. унифицирует **размер позиции** (как в бэктесте, через `backtest.position_fraction`),
2. использует **те же комиссии/проскальзывание** (`market.transaction_fee` и `market.slippage`),
3. выравнивает **длины окон** (контекст/сессия) c `cfg.seq.pre_signal_len` и `cfg.seq.agent_session_len` для построения окон из БД,
4. добавляет минимальные **pytest-тесты** на паритет расчётов, чтобы не допустить регрессии.

Это должно резко приблизить число сделок и метрики в `paper_trader.py` к тем, что даёт `backtest_engine.py` на тех же данных/модели.

---

## Repo-State Header (prosperous_bot)

* **Branch:** `prosperous_bot`
* **Latest commit:** `4f8ad89f74ffc56661386dc4b058165681a8e591` — *"docs: test_paper_trader_inference.py"*, 2025-10-21. ([GitHub][1])
* **Репозиторий:** `FMProducer/prosperous_bot` (структура ветки `third_party/rl-trading-binance` подтверждена). ([GitHub][2])
* **Релевантные файлы исходного кода:**

  * `third_party/rl-trading-binance/backtest_engine.py` — использует `market.slippage`, `market.transaction_fee`, `backtest.position_fraction` и пороги **advantage**. ([GitHub][3])
  * `third_party/rl-trading-binance/trading_environment.py` — точная механика исполнения/комиссий/лимитов. ([GitHub][4])
  * `third_party/rl-trading-binance/inference_adapter.py` — инференс с **advantage** и порогами из конфига. ([GitHub][5])
  * `third_party/rl-trading-binance/utils.py` — нормализация и детектор всплесков (динамическое окно). ([GitHub][6])
  * `third_party/rl-trading-binance/config.py` — источник истинных параметров: `seq.*`, `market.*`, `backtest.*`. ([GitHub][7])

> Примечание: я сверил логику по исходникам выше, чтобы изменения были **repo-first**, без допущений.

---

## Патч (unified diff)

> Лимиты соблюдены: 2 файла, ~<300 строк diff.

### 1) Выравнивание `paper_trader.py` под параметры бэктеста

* глобально фиксируем `MasterConfig` в модуле (`_MASTER_CFG`);
* используем `market.slippage`, `market.transaction_fee`, `backtest.position_fraction`;
* в `_load_cfg` подменяем `ctx_minutes` и `session_minutes` значениями из `cfg.seq.*`, чтобы **генерация окон** на БД совпала с бэктест-сессиями;
* в `main()` присваиваем `_MASTER_CFG = master_cfg`.

### 2) Тест на паритет расчётов

* лёгкий `pytest`, проверяющий, что функции размера/комиссий/проскальзывания в `paper_trader.py` выдают те же формулы, что в `TradingEnvironment`/бэктест-коде.

---

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/paper_trader.py
@@
-from utils import find_spike_windows, calculate_normalization_stats  # детектор + нормировка
+from utils import find_spike_windows, calculate_normalization_stats  # детектор + нормировка
+
+# --- Global master config for execution parity (set in main) ---
+_MASTER_CFG: Optional[Any] = None
 
@@
-def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
+def _load_cfg(cfg_path: str) -> Tuple[Cfg, Any]:
     mod = _load_py_module(cfg_path)
     if not hasattr(mod, "data") or not isinstance(mod.data, dict):
         raise RuntimeError("В конфиге нужен dict `data`.")
     data = mod.data
     # master_cfg НЕ обязателен: попробуем найти cfg / MasterConfig(), иначе оставим None
     master_cfg = getattr(mod, "cfg", None)
     if master_cfg is None and hasattr(mod, "MasterConfig"):
         try:
             master_cfg = mod.MasterConfig()
         except Exception:
             master_cfg = None
 
     # обязательные части (см. SYSTEM_PROMPT.md / README)
     dbp = data["db_provider"]
     config_name = os.path.splitext(os.path.basename(cfg_path))[0]
     index_csv = os.path.join("third_party", "rl-trading-binance", "output", config_name, "stream_backtest_index.csv")
@@
-    ctx_m = int(data.get("ctx_minutes", 30))
-    sess_m = int(data.get("session_minutes", 10))
+    ctx_m = int(data.get("ctx_minutes", 30))
+    sess_m = int(data.get("session_minutes", 10))
+    # >>> Align with MasterConfig to mirror backtest sessions/windows <<<
+    if master_cfg is not None:
+        try:
+            ctx_m = int(getattr(master_cfg.seq, "pre_signal_len"))
+            sess_m = int(getattr(master_cfg.seq, "agent_session_len"))
+        except Exception:
+            # мягкая деградация к значениям из data
+            pass
@@
-    return paper_trader_cfg, master_cfg
+    return paper_trader_cfg, master_cfg
 
@@
-def _apply_slippage(price: float, bps: float, side: str) -> float:
-    # bps = basis points (0.01% = 1 bps). Для покупки повышаем цену, для продажи понижаем.
-    delta = price * (bps / 10000.0)
-    return price + delta if side == "BUY" else price - delta
+def _apply_slippage(price: float, bps: float, side: str) -> float:
+    """
+    Prefer backtest slippage from MasterConfig (fraction), fallback to bps if absent.
+    """
+    global _MASTER_CFG
+    slip = (
+        float(getattr(getattr(_MASTER_CFG, "market", None), "slippage", None))
+        if _MASTER_CFG is not None
+        else None
+    )
+    slippage = slip if isinstance(slip, (float, int)) else (bps / 10000.0)
+    delta = price * slippage
+    return price + delta if side == "BUY" else price - delta
 
-def _fees_cost(notional: float, fee_bps: float) -> float:
-    return notional * (fee_bps / 10000.0)
+def _fees_cost(notional: float, fee_bps: float) -> float:
+    """
+    Prefer backtest fee from MasterConfig (fraction), fallback to bps if absent.
+    """
+    global _MASTER_CFG
+    fee = (
+        float(getattr(getattr(_MASTER_CFG, "market", None), "transaction_fee", None))
+        if _MASTER_CFG is not None
+        else None
+    )
+    fee_rate = fee if isinstance(fee, (float, int)) else (fee_bps / 10000.0)
+    return notional * fee_rate
 
-def _position_size(capital: float, risk_pct: float, entry: float) -> float:
-    risk_usdt = capital * (risk_pct / 100.0)
-    qty = max(risk_usdt / max(entry, 1e-12), 0.0)
-    return qty
+def _position_size(capital: float, risk_pct: float, entry: float) -> float:
+    """
+    If MasterConfig is available, mirror backtest sizing:
+      notional = capital * cfg.backtest.position_fraction
+      qty = notional / entry
+    Else, use legacy risk% sizing.
+    """
+    global _MASTER_CFG
+    if _MASTER_CFG is not None:
+        pf = float(getattr(getattr(_MASTER_CFG, "backtest", None), "position_fraction", 0.5))
+        notional = max(capital * pf, 0.0)
+        return max(notional / max(entry, 1e-12), 0.0)
+    risk_usdt = capital * (risk_pct / 100.0)
+    return max(risk_usdt / max(entry, 1e-12), 0.0)
 
@@
 def main(argv: List[str]) -> int:
     if len(argv) < 2:
         print("Использование: python paper_trader.py configs/alpha.py")
         return 2
     cfg_path = argv[1]
-    cfg, master_cfg = _load_cfg(cfg_path)
+    cfg, master_cfg = _load_cfg(cfg_path)
+    # expose MasterConfig globally for execution helpers
+    global _MASTER_CFG
+    _MASTER_CFG = master_cfg
+
+    # Harmonize minutes with backtest session lengths if MasterConfig present
+    if master_cfg is not None:
+        try:
+            if cfg.session_minutes != master_cfg.seq.agent_session_len:
+                print(f"[WARN] Override session_minutes: {cfg.session_minutes} -> {master_cfg.seq.agent_session_len}")
+                cfg.session_minutes = master_cfg.seq.agent_session_len
+            if cfg.ctx_minutes != master_cfg.seq.pre_signal_len:
+                print(f"[WARN] Override ctx_minutes: {cfg.ctx_minutes} -> {master_cfg.seq.pre_signal_len}")
+                cfg.ctx_minutes = master_cfg.seq.pre_signal_len
+        except Exception:
+            pass
@@
     # Загружаем модель (если указана)
     policy: Optional[_Policy] = _load_policy(
         cfg.inference.policy_loader,
         cfg.inference.checkpoint_path,
         cfg_path,
         stats
     )
*** End Patch
```

---

```diff
*** Begin Patch
*** Add File: third_party/rl-trading-binance/tests/test_exec_parity.py
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
+from third_party.rl-trading-binance.config import cfg as master_cfg
+
+def _load_paper_trader():
+    p = pathlib.Path("third_party/rl-trading-binance/paper_trader.py")
+    spec = importlib.util.spec_from_file_location("paper_trader", p.as_posix())
+    mod = importlib.util.module_from_spec(spec)
+    assert spec and spec.loader
+    spec.loader.exec_module(mod)  # type: ignore
+    # пробрасываем MasterConfig
+    mod._MASTER_CFG = master_cfg
+    return mod
+
+def test_position_size_parity():
+    pt = _load_paper_trader()
+    capital = 10_000.0
+    entry = 100.0
+    pf = master_cfg.backtest.position_fraction
+    expected_qty = (capital * pf) / entry
+    qty = pt._position_size(capital, risk_pct=1.0, entry=entry)
+    assert math.isclose(qty, expected_qty, rel_tol=1e-7)
+
+def test_fees_parity():
+    pt = _load_paper_trader()
+    notional = 1234.56
+    expected_fee = notional * master_cfg.market.transaction_fee
+    fee = pt._fees_cost(notional, fee_bps=40.0)  # игнорируется при наличии MasterConfig
+    assert math.isclose(fee, expected_fee, rel_tol=1e-12)
+
+def test_slippage_parity_buy_sell():
+    pt = _load_paper_trader()
+    price = 200.0
+    slip = master_cfg.market.slippage
+    assert math.isclose(pt._apply_slippage(price, 5.0, "BUY"),  price * (1 + slip), rel_tol=1e-12)
+    assert math.isclose(pt._apply_slippage(price, 5.0, "SELL"), price * (1 - slip), rel_tol=1e-12)
+
*** End Patch
```

---

## Как это минимизирует расхождения

* **Размер позиции**: теперь расчёт ровно как в бэктесте — `notional = balance * backtest.position_fraction`, а не через `% риска`. Это ключ к совпадению **количества и PnL** сделок. См. реализацию в `backtest_engine.py` (по месту создания `TradingEnvironment`) и в `TradingEnvironment.backtest_step` (объём = `balance / exec_price`). ([GitHub][3])
* **Комиссии/проскальзывание**: берём `market.transaction_fee` и `market.slippage` из `config.py`, как делает окружение бэктеста. Раньше в `paper_trader` это было в bps и могло отличаться. ([GitHub][7])
* **Длины окон**: детектор и индекс теперь жёстко синхронизированы c `cfg.seq.pre_signal_len`/`agent_session_len`, чтобы контекст/сессия из БД соответствовали сессиям в NPZ. ([GitHub][6])
* **Пороги действий**: уже унифицированы через `inference_adapter.py` (advantage-thresholds из `cfg.backtest.*`). ([GitHub][5])

---

## Команды для PR

> Базовая ветка PR — `prosperous_bot` (как требует процесс).

```bash
# 1) Новая ветка
git checkout -b feature/paper-trader-exec-parity

# 2) Применить патч (сохраните patch в файл changes.patch)
git apply --index changes.patch
git commit -m "feat(paper_trader): align sizing/fees/slippage & session/ctx minutes with backtest; add exec parity tests"

# 3) Пуш
git push -u origin feature/paper-trader-exec-parity

# 4) PR (base = prosperous_bot)
gh pr create -t "paper_trader ↔ backtest: execution parity (sizing/fees/slippage + window lengths)" -b "
### 🎯 Goal
Сделать поведение paper_trader максимально идентичным бэктесту: размер позиции, комиссии/проскальзывание, длительности окон.

### 📝 Implementation Details
- paper_trader.py: глобальный доступ к MasterConfig; выравнивание ctx/session минут; комиссии/проскальзывание/размер позиции из cfg.
- tests/test_exec_parity.py: pytest на паритет формул.

### 📈 KPI/Risk Assessment
- **Sharpe:** ≈ без изменения (ожидается сближение с бэктестом).
- **Max DD:** ≈ без изменения (схожее исполнение).
- **Profit Factor:** ожидается сближение к бэктестовым значениям.

### 롤백 계획 (Rollback Plan)
Простой revert PR. Фича изолирована в paper_trader + тесты.

---

## Чек-лист «1:1» поведения на потоке/БД

| Шаг | Действие                                                                                                                                                                     | KPI/риск                                  |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1   | **Единые окна**: `ctx_minutes = cfg.seq.pre_signal_len`, `session_minutes = cfg.seq.agent_session_len` в `paper_trader` (патч делает это автоматически)                      | Согласованность выборки контекстов/сессий |
| 2   | **Нормализация**: считать статистики теми же каналами/формулой, как в бэктесте (`utils.calculate_normalization_stats`), проверяя, что **все expected_channels присутствуют** | Идентичный вход модели                    |
| 3   | **Размер позиции**: использовать `backtest.position_fraction` (а не risk%)                                                                                                   | Кол-во/масштаб PnL совпадает              |
| 4   | **Комиссии/проскальзывание**: `market.transaction_fee`, `market.slippage`                                                                                                    | Сходство метрик (PF, Win-rate)            |
| 5   | **Пороги действий**: advantage-thresholds из `cfg.backtest.*` (через `inference_adapter`)                                                                                    | Идентичная фильтрация сигналов            |
| 6   | **Часовой пояс/индекс минуток**: UTC-индекс без разрывов, метод поиска close = `"pad"`                                                                                       | Исключить артефакты off-by-one            |
| 7   | **Кэш/детерминизм**: зафиксировать seed и отключить сторонние стохастики в инференсе                                                                                         | Повторяемость результатов                 |
| 8   | **pytest**: прогнать `tests/test_exec_parity.py`                                                                                                                             | Гейтинг регрессий до запуска              |

---

## Важные отличия «Бэктест vs БД/реалтайм» и как их минимизировать

* **Look-ahead**: на реальном потоке будущего нет. В детекторе используйте `use_lookahead=False` для «честных» окон, а для «копирования» бэктеста **оставьте `True`** (как по умолчанию) — это объясняет часть расхождений в числе сделок. ([GitHub][6])
* **Исполнение по минутным барам**: бэктест исполняет на «следующей минуте» со слippage/fee из `config.py`. В `paper_trader` после патча используется та же формула. ([GitHub][4])
* **Сайзинг/капитал**: в бэктесте «подсчёт PnL» идёт от виртуального саб-капитала с фиксированной долей; риск-процентная модель в реальном времени даёт другую динамику — мы её отключили по умолчанию (через приоритет MasterConfig). ([GitHub][3])

---

## Что запускать после применения патча

```bash
# прогнать тесты
pytest -q third_party/rl-trading-binance/tests/test_exec_parity.py

# пример запуска paper_trader (как раньше)
python third_party/rl-trading-binance/paper_trader.py configs/alpha.py
```

---

### Ссылки на используемые места кода (для ревью)

* `backtest_engine.py`: формирование размера позиции и конфигурация окружения, где используются `market.slippage` и `market.transaction_fee`. ([GitHub][3])
* `trading_environment.py`: фактическое применение проскальзывания/комиссий при `BUY/SELL/CLOSE`. ([GitHub][4])
* `inference_adapter.py`: advantage-пороговая логика, совпадающая с бэктестом. ([GitHub][5])
* `utils.py`: детектор и нормализация (одна и та же формула, каналы и длины историй). ([GitHub][6])
* `config.py`: **источник истины** по `seq.*`, `market.*`, `backtest.*`. ([GitHub][7])

---

[1]: https://github.com/FMProducer/prosperous_bot/commit/4f8ad89f74ffc56661386dc4b058165681a8e591 "docs: test_paper_trader_inference.py · FMProducer/prosperous_bot@4f8ad89 · GitHub"
[2]: https://github.com/FMProducer/prosperous_bot "GitHub - FMProducer/prosperous_bot"
[3]: https://github.com/FMProducer/prosperous_bot/raw/prosperous_bot/third_party/rl-trading-binance/backtest_engine.py "raw.githubusercontent.com"
[4]: https://github.com/FMProducer/prosperous_bot/raw/prosperous_bot/third_party/rl-trading-binance/trading_environment.py "raw.githubusercontent.com"
[5]: https://github.com/FMProducer/prosperous_bot/raw/prosperous_bot/third_party/rl-trading-binance/inference_adapter.py "raw.githubusercontent.com"
[6]: https://github.com/FMProducer/prosperous_bot/raw/prosperous_bot/third_party/rl-trading-binance/utils.py "raw.githubusercontent.com"
[7]: https://github.com/FMProducer/prosperous_bot/raw/prosperous_bot/third_party/rl-trading-binance/config.py "raw.githubusercontent.com"
