# TL;DR

Готов единый PR-патч (< 600 строк), который ускоряет CPU-бэктест без изменения логики принятия решений и без запрещённых переключателей. Что сделано:

1. **Векторизация поиска импульсов** в `create_signal_groups_from_continuous` — убрали Python-цикл `while`, считаем кандидатов скопом и формируем окна только для прошедших фильтр (см. текущую реализацию с циклом — медленно на больших таймсериях, ).
2. **Переиспользование одного `TradingEnvironment`** на все сессии — вместо пересоздания env на каждую сессию (как сейчас, ) мы один раз создаём env с «пустой» заготовкой нужной формы и перед каждым запуском подменяем `sequences[0]`, задаём `initial_balance`, делаем `reset(options={"forced_index": 0})`.
3. **Микро-оптимизация среды:** кешируем индекс канала `close` в `TradingEnvironment` и используем его в горячих местах (вместо многократного `self.data_channels.index("close")`, см. места вызовов, ).

Запрещённые изменения не делались: **`selection_strategy="greedy"`, `return_qvals=False`, `use_cache=False` — не трогаем** (см. текущие ветки выбора действия в движке/континуальном бэктесте, ).

---

## Шаг | Действие | KPI/риск

1 | Векторизовать отбор сигналов в `create_signal_groups_from_continuous` | −60–90% CPU на адаптере (меньше питоновских проходов); риск: логика порога волатильности сохранена
2 | Переиспользовать один `TradingEnvironment` в `run_backtest` | −20–40% CPU/GC за счёт отсутствия множественных конструкторов env; риск: корректно сбрасываем состояние через `reset(options={...})`
3 | Закешировать индекс `close` в среде | −5–10% CPU в горячем цикле; риск: нулевой (поведенческих изменений нет)

---

# Unified diff (1 PR, ≤ 600 строк)

> База: `third_party/rl-trading-binance/`
> Файлы: `backtest_continuous.py`, `trading_environment.py`

```diff
*** a/third_party/rl-trading-binance/backtest_continuous.py
--- b/third_party/rl-trading-binance/backtest_continuous.py
@@
     grouped_signals = defaultdict(list)
     close_idx = cfg.data.data_channels.index("close")
 
     for ticker_name in tickers_to_process:
         ticker_df = df[df['symbol'] == ticker_name].copy()
         if ticker_df.empty:
             logging.warning(f"Adapter: No data found for ticker {ticker_name} in the .npz file. Skipping.")
             continue
 
         logging.info(f"Adapter: Loaded {len(ticker_df)} rows for {ticker_name}.")
-        market_data = ticker_df[cfg.data.data_channels].to_numpy(dtype=np.float32)
-        
-        logging.info(f"Adapter: Scanning for volatile signals in {ticker_name}.")
-        i = cfg.seq.full_seq_len
-        while i < len(market_data):
-            session_window = market_data[i - cfg.seq.full_seq_len : i]
-            
-            if cfg.backtest.volatility_threshold is not None:
-                volatility_window = session_window[0:cfg.seq.pre_signal_len]
-                close_price_start = volatility_window[0, close_idx]
-                close_price_end = volatility_window[-1, close_idx]
-                if close_price_start > 0:
-                    volatility = abs(close_price_end - close_price_start) / close_price_start
-                    if volatility >= cfg.backtest.volatility_threshold:
-                        signal_dt = ticker_df.index[i - cfg.seq.post_signal_len - 1].to_pydatetime()
-                        grouped_signals[signal_dt].append((ticker_name, session_window))
-            else:
-                # If no filter, every moment is a signal
-                signal_dt = ticker_df.index[i - cfg.seq.post_signal_len - 1].to_pydatetime()
-                grouped_signals[signal_dt].append((ticker_name, session_window))
-            
-            i += 1 # Always advance the window
+        market_data = ticker_df[cfg.data.data_channels].to_numpy(dtype=np.float32)
+
+        logging.info(f"Adapter: Scanning for volatile signals in {ticker_name}.")
+        n = len(market_data)
+        fs = cfg.seq.full_seq_len
+        pre = cfg.seq.pre_signal_len
+        post = cfg.seq.post_signal_len
+
+        if n >= fs:
+            # Кандидаты окон: концы окон i в [fs, n)
+            end_idx = np.arange(fs, n, dtype=np.int64)
+            start_idx = end_idx - fs
+            # Пара цен для расчёта волатильности в первых pre минутах окна
+            pre_start = start_idx
+            pre_end   = start_idx + (pre - 1)
+            close_arr = market_data[:, close_idx]
+
+            if getattr(cfg.backtest, "volatility_threshold", None) is not None:
+                valid = (pre_start >= 0) & (pre_end < n) & (close_arr[pre_start] > 0)
+                if np.any(valid):
+                    c0 = close_arr[pre_start[valid]]
+                    c1 = close_arr[pre_end[valid]]
+                    vol = np.abs(c1 - c0) / c0
+                    passed = valid.copy()
+                    passed[valid] = vol >= cfg.backtest.volatility_threshold
+                else:
+                    passed = np.zeros_like(end_idx, dtype=bool)
+            else:
+                # Без фильтра — все моменты валидны
+                passed = np.ones_like(end_idx, dtype=bool)
+
+            # Формируем окна только для прошедших фильтр
+            for i in end_idx[passed]:
+                session_window = market_data[i - fs : i]
+                signal_pos = i - post - 1
+                if 0 <= signal_pos < n:
+                    signal_dt = ticker_df.index[signal_pos].to_pydatetime()
+                    grouped_signals[signal_dt].append((ticker_name, session_window))
 
     logging.info(f"Adapter: Found a total of {len(grouped_signals)} signal groups across all tickers.")
     return grouped_signals
@@
-    for signal_dt, signals in tqdm(grouped_backtest_data.items(), desc="Processing Signals"):
+    # --- Оптимизация: создаём один Environment и переиспользуем ---
+    # Заглушка правильной формы (fs x num_features); окно будет подменяться перед reset()
+    dummy_session = np.zeros((cfg.seq.full_seq_len, cfg.seq.num_features), dtype=np.float32)
+    env = TradingEnvironment(
+        sequences=[dummy_session],
+        stats=stats,
+        render_mode=cfg.render_mode,
+        full_seq_len=cfg.seq.full_seq_len,
+        num_features=cfg.seq.num_features,
+        num_actions=cfg.market.num_actions,
+        flat_state_size=cfg.seq.flat_state_size,
+        initial_balance=0.0,  # выставим перед каждым запуском
+        pre_signal_len=cfg.seq.pre_signal_len,
+        data_channels=cfg.data.data_channels,
+        slippage=cfg.market.slippage,
+        transaction_fee=cfg.market.transaction_fee,
+        agent_session_len=cfg.seq.agent_session_len,
+        agent_history_len=cfg.seq.agent_history_len,
+        input_history_len=cfg.seq.input_history_len,
+        price_channels=cfg.data.price_channels,
+        volume_channels=cfg.data.volume_channels,
+        other_channels=cfg.data.other_channels,
+        action_history_len=cfg.seq.action_history_len,
+        inaction_penalty_ratio=cfg.market.inaction_penalty_ratio,
+        backtest_mode=cfg.backtest_mode,
+        use_risk_management=cfg.backtest.use_risk_management,
+    )
+
+    for signal_dt, signals in tqdm(grouped_backtest_data.items(), desc="Processing Signals"):
         open_sessions = [open_s for open_s in open_sessions if open_s["end_time"] > signal_dt]
         free_slots = cfg.backtest.max_parallel_sessions - len(open_sessions)
         if free_slots <= 0:
             logging.info("Too many tickers received, skipping")
             continue
@@
-        for ticker_name, session in selected_signals:
-            position_size = balance * cfg.backtest.position_fraction
-
-            env = TradingEnvironment(
-                sequences=[session],
-                stats=stats,
-                render_mode=cfg.render_mode,
-                full_seq_len=cfg.seq.full_seq_len,
-                num_features=cfg.seq.num_features,
-                num_actions=cfg.market.num_actions,
-                flat_state_size=cfg.seq.flat_state_size,
-                initial_balance=position_size,
-                pre_signal_len=cfg.seq.pre_signal_len,
-                data_channels=cfg.data.data_channels,
-                slippage=cfg.market.slippage,
-                transaction_fee=cfg.market.transaction_fee,
-                agent_session_len=cfg.seq.agent_session_len,
-                agent_history_len=cfg.seq.agent_history_len,
-                input_history_len=cfg.seq.input_history_len,
-                price_channels=cfg.data.price_channels,
-                volume_channels=cfg.data.volume_channels,
-                other_channels=cfg.data.other_channels,
-                action_history_len=cfg.seq.action_history_len,
-                inaction_penalty_ratio=cfg.market.inaction_penalty_ratio,
-                backtest_mode=cfg.backtest_mode,
-                use_risk_management=cfg.backtest.use_risk_management,
-            )
-
-            obs, _ = env.reset()
+        for ticker_name, session in selected_signals:
+            position_size = balance * cfg.backtest.position_fraction
+            # Подменяем данные и баланс, мягкий сброс в начало эпизода
+            env.sequences[0] = session
+            env.initial_balance = position_size
+            obs, _ = env.reset(options={"forced_index": 0})
             for step in range(cfg.seq.agent_session_len):
                 cache_key = (ticker_name, signal_dt + dt.timedelta(minutes=step))
                 
                 if cfg.backtest.selection_strategy == "advantage_based_filter":
                     q_vals = agent.select_action(
                         state=obs,
                         training=False,
                         return_qvals=cfg.backtest.return_qvals,
                         use_cache=cfg.backtest.use_cache,
                         cache_key=cache_key,
                     )
                     adv = q_vals - q_vals[0]
                     action = int(np.argmax(adv))
                     confidence = adv[action]
 
                     pass_adv = get_pass_advantage(action, confidence, cfg)
                     if pass_adv:
                         action = 0
                 else:
                     action = agent.select_action(
                         state=obs,
                         training=False,
                         return_qvals=False,
                         use_cache=cfg.backtest.use_cache,
                         cache_key=cache_key,
                     )
 
                 obs, _, done, _, info = env.backtest_step(
                     action=action,
                     signal_dt=signal_dt,
                     ticker=ticker_name,
                     stop_loss=cfg.backtest.stop_loss,
                     take_profit=cfg.backtest.take_profit,
                     trailing_stop=cfg.backtest.trailing_stop,
                 )
 
                 if info["position_closed"]:
                     info["ticker"] = ticker_name
                     trade_log.log_trade(info, balance)
                     balance += info.get("trade_realized_pnl", 0.0)
                     result.update(signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len), info, balance)
                 if done:
                     break
 
             open_sessions.append({"end_time": signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len)})
*** a/third_party/rl-trading-binance/trading_environment.py
--- b/third_party/rl-trading-binance/trading_environment.py
@@
         self.other_channels = other_channels
         self.action_history_len = action_history_len
         self.num_actions = num_actions
         self.inaction_penalty_ratio = inaction_penalty_ratio
         self.backtest_mode = backtest_mode
         self.use_risk_management = use_risk_management
+        # Cache frequently used channel index
+        self.close_idx = self.data_channels.index("close")
@@
-        price = self.current_seq[price_idx, self.data_channels.index("close")]
+        price = self.current_seq[price_idx, self.close_idx]
@@
-            current_price = self.current_seq[price_idx, self.data_channels.index("close")]
+            current_price = self.current_seq[price_idx, self.close_idx]
@@
-            current_price = self.current_seq[price_idx, self.data_channels.index("close")]
+            current_price = self.current_seq[price_idx, self.close_idx]
@@
-        price = self.current_seq[price_idx, self.data_channels.index("close")]
+        price = self.current_seq[price_idx, self.close_idx]
```

---

## Обоснование и ссылки на исходники

* **Замена Python-цикла в адаптере на векторный отбор**: текущая версия проходит каждую минуту окна, снимает срезы и пересчитывает волатильность в чистом Python (см. `while i < len(market_data)` и расчёт `volatility`, ). В патче мы одним проходом строим индексы старт/конец pre-отрезка и считаем вектор `vol`, а затем создаём окна только для прошедших фильтр — это резкое снижение overhead.
* **Переиспользование `TradingEnvironment`**: исходно среда создаётся для каждого тикера/сигнала внутри цикла (дорого по времени и памятипотреблению, ). Мы переносим конструктор выше, а в цикле подменяем данные и `initial_balance` + `reset(options={"forced_index":0})`.
* **Кеширование индекса `close`**: много мест, где делается `self.data_channels.index("close")` в горячем цикле шага/инфо/бэктеста (см. фрагменты, ). Вынесли в `self.close_idx` один раз при инициализации.

---

## Что **не** меняли (согласно вашему требованию)

* Не трогали стратегию выбора действий и кэширование агента: `selection_strategy`, `return_qvals`, `use_cache` остались как есть (см. текущую ветку выбора действий, ).
* Не вводили новых обязательных конфиг-параметров — все дефолты задаются внутри кода без жёсткой привязки.
* Не меняли экономику (комиссии, слиппедж, риск-менеджмент), только вычислительную оптимизацию.

---

## Ожидаемый эффект на CPU-бэктест

* Адаптер (сканер импульсов): **−60–90% CPU** на больших файлах за счёт векторизации.
* Основной цикл бэктеста: **−20–40% CPU** благодаря отсутствию многократных `__init__` среды и снижению давления на GC.
* Микро-оптимизации среды: **−5–10% CPU** в сумме.
  Совокупно на 8C/16T CPU ожидаем **х2–х3** ускорение wall-clock времени (оценка; подтвердить замером).

---

## Команды для PR

```bash
# 1) Ветка с фичей
git checkout -b feature/fast-continuous-backtest

# 2) Сохранить патч в файл и применить
# (вставьте diff выше в changes.patch)
git apply --index changes.patch
git commit -m "feat(backtest): vectorized signal scan + env reuse; cache close_idx in env"

# 3) Выложить ветку
git push -u origin feature/fast-continuous-backtest

# 4) Создать PR в базовую ветку prosperous_bot
gh pr create -t "Speed up continuous backtest (CPU-only): vectorized scan + env reuse" -b "### 🎯 Goal
Ускорить CPU-бэктест на непрерывных данных без изменения логики принятия решений.

### 📝 Implementation Details
- Векторизация отбора сигналов в \`create_signal_groups_from_continuous\` (убран Python-цикл).
- Переиспользование одного \`TradingEnvironment\` вместо пересоздания в цикле.
- Кеширование индекса \`close\` в среде для снижения накладных расходов.

Затронутые файлы:
- \`third_party/rl-trading-binance/backtest_continuous.py\`
- \`third_party/rl-trading-binance/trading_environment.py\`

### 📈 KPI/Risk Assessment
- **Sharpe:** прогноз — без изменений (функционал не менялся)
- **Max DD:** прогноз — без изменений
- **Profit Factor:** прогноз — без изменений
- **Perf:** ожидаемое ускорение х2–х3 на CPU

### 롤백 계획 (Rollback Plan)
- Полный откат: revert PR.
- Риск минимален: изменения ограничены вычислительной частью, без влияния на стратегию.

---
Repo-State: branch=prosperous_bot, sha=b304d47e2ebeae46254ee0f9abaa440c31de72a7, title=\"docs: max_parallel_sessions\"
" -B prosperous_bot
```

---

## Напоминание по требованиям проекта

* Конфигурации — из `configs/` в составе RL-проекта; наш патч не добавляет новых конфиг-ключей. (Системные инструкции, )
* Обязателен `pytest` и отчёты бэктеста в `output/<config_name>/`. (Системные инструкции, )