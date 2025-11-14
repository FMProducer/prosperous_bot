Предлагаемый diff для alpha_convolutions_seed_404.py
Минимальные изменения: добавьте db periods/symbols (train: 2024-01 to 2025-01 для исторических данных; val/test: 2025-02–05 или под backtest Aug–Sep для OOS), set npz paths=None, optional data_channels match DB. Добавьте в конец (после optuna_search_space), перед # Notes.

text
--- alpha_convolutions_seed_404.py (оригинал)
+++ alpha_convolutions_seed_404.py (обновлённый)
@@ -100,6 +100,10 @@ cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025
 cfg.logging.per_trial_logs = True
 
 # 1000, default = None
 cfg.debug.debug_max_size_data = 100  # Для теста DB; set None для full после
+ 
+ # DB overrides: periods для split (UTC, YYYY-MM-DD); symbols optional для ускорения
+ cfg.db.train_period_start = "2024-01-01"
+ cfg.db.train_period_end = "2025-01-01"  # Широкий train для спайков
+ cfg.db.val_period_start = "2025-02-01"
+ cfg.db.val_period_end = "2025-05-01"  # Pre-backtest val
+ cfg.db.test_period_start = "2025-08-01"  # Align с backtest.time_range
+ cfg.db.test_period_end = "2025-09-30"
+ cfg.db.symbols = None  # None = all; or ["BTCUSDT", "ETHUSDT"] для теста (filter в query)
 
 cfg.debug.use_final_model = False
 
@@ -150,10 +155,10 @@ cfg.perf.cudnn_benchmark = True
 # ---- Vectorized Environments ----
 
 # ... vec params unchanged ...
 
 # ... mc_dropout unchanged ...
 
-cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
-cfg.paths.val_data_path = "data/val_data_fair_2m.npz"
-# test_data_path отдельный или тот же что и для backtest
-cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"
+# DB mode: ignore .npz; set None (train.py skips if dsn set)
+cfg.paths.train_data_path = None
+cfg.paths.val_data_path = None
+cfg.paths.test_data_path = None
+cfg.paths.backtest_data_path = None  # Если используется
+
+# Optional: ensure data_channels match DB SELECT (OHLCV + num_trades)
+cfg.data.data_channels = ["open", "high", "low", "close", "volume", "num_trades"]
+cfg.data.price_channels = ["open", "high", "low", "close"]
+cfg.data.volume_channels = ["volume"]
+cfg.data.other_channels = ["num_trades"]  # Для calculate_normalization_stats
 
 # Модель для бэктеста.
 # cfg.paths.model_path = r"C:\\...\\best.pth"
 
 # cfg.paths.norm_stats_path = r"C:\\...\\norm_stats.json"  # Уже закомментировано; dynamic compute ok
 
 cfg.random_seed = 404
 
 cfg.paths.config_name = "alpha_convolutions_seed_404"
 
 # ... detector params unchanged (abs_change_pct=4.0 etc. — adjust if few spikes)
 
 # ... rest unchanged (bundle, per, eps, optuna)
Пояснения к diff
db periods: Train — исторический (2024–2025) для ~8M баров/spikes; val — 2025-02–05 (pre-OOS); test — align с backtest.time_range (Aug–Sep 2025 для fair eval). Измените под ваши данные (проверьте SELECT MIN/MAX(open_time_ms) FROM klines_1m; — ожидается ~2024+ для futures).

db.symbols: None= all из БД (может 100+ символов, ~1–2 мин query); set list для теста (top volatile: BTC/ETH + altcoins из paper.symbols comment).

paths.*_data_path: Set None — явно отключает npz (train.py не использует, но avoids warnings). backtest_data_path тоже, если backtest использует test_seqs.

data_channels: Explicit set для match DB (base_volume=volume, trade_count=num_trades; no quote_volume/taker). price/volume/other — для stats (calculate_normalization_stats splits by type). Если default в config.py другой (напр. +VWAP), скорректируйте или удалите.

debug_max_size_data=100: Для быстрого теста (sequences<=100/symbol); set None для full (60k episodes, +5–10 мин preprocess).

detector params: abs_change_pct=4.0/constrast_min=5.0 — aggressive (few spikes); if 0 sequences, lower to 1.0/2.0 в diff.

norm_stats_path: Уже commented — ok; train.py saves auto в output/norm_stats.json.

Размер: +~20 строк; нет конфликтов с CNN/rl params.

Применение и тест
Примените diff: Добавьте в конец (после cfg.vec.scale_epsilon_by_envs=True), перед # Spike Detector.