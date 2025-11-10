# configs/alpha.py
from config import MasterConfig
cfg = MasterConfig()
ACTION_HISTORY_LEN = 2
cfg.model.cnn_maps = [64, 96, 128]
cfg.model.cnn_kernels = [7, 5, 3]
cfg.model.cnn_strides = [2, 1, 1]
cfg.model.dense_val = [128, 64, 32]
cfg.model.dense_adv = [128, 64, 32]
# 4 + action_history_len * num_actions
cfg.model.additional_feats = 4 + ACTION_HISTORY_LEN * 4 # 4 + 2*4 = 12
# 0 ≤ p < 0.5; typical values are 0.1–0.2
cfg.model.dropout_p = 0.15
# Для устойчивого отбора чекпоинтов на GTX 1070 + 6C/12T
cfg.trainlog.num_val_ep = 3500
cfg.trainlog.val_freq = 1000
# Увеличиваем общий горизонт обучения (качество > скорость)
cfg.trainlog.episodes = 60000
cfg.trainlog.plot_top_n = 10
cfg.per.buffer_size = 200000
cfg.rl.batch_size = 64
# Стабильнее обновления с меньшим шагом
cfg.rl.learning_rate = 2e-5
cfg.rl.train_start = 15000
cfg.rl.gamma = 0.9995
cfg.rl.n_step = 20
# стабильнее целевые обновления
cfg.rl.target_update_freq = 10000
# Чуть мягче клиппинг — меньше «зажимаем» обучение, но защищаемся от выбросов
cfg.rl.max_gradient_norm = 5.0
# альтернативы: "Validation_mean_reward" и "Validation_mean_pnl"
cfg.trainlog.available_metrics=[
    "Validation_mean_reward",
    "Validation_mean_pnl", # Приоритет №6
    "Validation_win_rate", # Приоритет №4
    "Validation_profit_factor", # Приоритет №3
    "Validation_max_drawdown", # Приоритет №5
    "Validation_all_pnls",
    "Validation_sharpe", # Приоритет №1
    "Validation_sortino", # Приоритет №2
]
# Мультикритериальный отбор
cfg.trainlog.val_selection_metrics = [
    "Validation_win_rate",
    "Validation_profit_factor",
    "Validation_sharpe",
    "Validation_sortino",
    "Validation_max_drawdown",
    "Validation_mean_pnl"
]
# ── Валидационный гейт для отбора best.pth.
# ВАЖНО: TrainLogConfig запрещает extra-поля, поэтому кладём гейт на верхний уровень MasterConfig:
# train.py теперь читает fallback из cfg.validation_gate.
cfg.validation_gate = {
    "min_sharpe": 0.20,
    "min_sortino": 0.40,
    "min_profit_factor": 1.30,
    # Максимально допустимая просадка (20%). Меньшая просадка = большее число (e.g., -0.10 > -0.20).
    "max_drawdown_at_most": -0.20,
    "min_win_rate": 0.74,   # 0..1
    "min_trades": 40,      # минимум сделок на валидации (ваше требование)
    "deny_inf_pf": True,    # запрещаем PF=inf
    # НОВОЕ: Запрещаем модели с нулевой просадкой, т.к. это нереалистично.
    "deny_zero_drawdown": True,
}
# Широкий взгляд на рынок для оценки волатильности и риска
cfg.seq.agent_history_len = 30
cfg.seq.agent_session_len = 10
# NB: pre_signal_len используется в utils.compute_metrics; согласуем с agent_history_len при отсутствии явной настройки.
if not hasattr(cfg.seq, "pre_signal_len"):
    cfg.seq.pre_signal_len = cfg.seq.agent_history_len
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = False
cfg.backtest.max_parallel_sessions = 2
cfg.backtest.position_fraction = 0.4
cfg.backtest.order_size_usdt = 4000.0
# ["advantage_based_filter", "ensemble_q_filter"]
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.015
cfg.backtest.short_action_threshold = 0.015
# cfg.backtest.close_action_threshold = 0.0180067279703063
# cfg.backtest.ensemble_n_samples = 1
# maximum allowed variance (uncertainty) (range: 0.001 to 0.015) prev: 0.002582999563187257
# cfg.backtest.ensemble_max_sigma = 0.0064449245324541046
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
# use_risk_management (from Trial #187)
cfg.backtest.use_risk_management = True
# cfg.backtest.stop_loss = 0.0153228603723445
# cfg.backtest.take_profit = 0.0447149987567144
cfg.backtest.trailing_stop = 0.018
# Execution timing: 0 = current behavior (may inflate returns), 1 = honest next-bar execution
cfg.backtest.exec_delay_bars = 1
cfg.backtest.plot_backtest_balance_curve = True
# Linear Trailing Stop Parameters for Optuna ---
# d_min: нижний пол для отступа трейла (напр. 0.2–0.6%)
cfg.backtest.trailing_stop_min = 0.005
# Множитель для fee_buf = fee_buffer_mult * fee (обычно ~2.0)
cfg.backtest.fee_buffer_mult = 2.0
# Hysteresis for TSL updates ---
cfg.backtest.delta_p_hysteresis = 0.0015
# Explicit time range for backtesting ---
# This ensures the backtest runs on the correct, unseen data period.
cfg.backtest.time_range = {"start_utc": "2024-10-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

cfg.logging.per_trial_logs = False
# 1000,  default = None
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
# AMP: экономия VRAM и потенциальный прирост на свертках; на Pascal (GTX 1070) и новее FP16 даёт ускорение и экономию памяти.
cfg.perf.use_amp = False
cfg.perf.amp_dtype = "float16"
# torch.compile: снижает overhead Python-графа; режим "reduce-overhead" — наиболее безопасный.
cfg.perf.compile_mode = None
cfg.perf.compile_dynamic = False
# DataLoader: загрузка с CPU (4 физ. ядра). Для коротких сессий — умеренные значения.
cfg.perf.dataloader_num_workers = 4
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = True
cfg.perf.prefetch_factor = 2
# CuDNN Heuristics
cfg.perf.cudnn_benchmark = False
# ---- Vectorized Environments ----
# Увеличим количество параллельных сред для ускорения сбора данных.
# На Windows/спавн backend "subproc" может оказаться медленнее из-за накладных расходов spawn.
cfg.vec.num_envs = 1 # Количество параллельных сред
# По умолчанию используем DummyVecEnv (часто быстрее для "лёгких" env).
# "subproc" # Использовать мультипроцессинг
cfg.vec.backend = "dummy"
cfg.vec.start_method = "spawn"
# Масштабировать скорость убывания epsilon на количество параллельных сред.
# Это восстанавливает паритет поведения между single-env и vec-env по числу env-шагов до той же ε.
cfg.vec.scale_epsilon_by_envs = True

# ─────────────────────────────────────────────────────────────
# MC-DROPOUT ДЛЯ ОБУЧЕНИЯ — внешний контейнер (не в pydantic RLConfig)
# ─────────────────────────────────────────────────────────────
mc_dropout_cfg = type("obj", (), {})()
mc_dropout_cfg.enable = False                 # ВЫКЛ: без MC в обучении
# Сколько семплов при выборе действия (он-полиси): 1 = без ансамбля
mc_dropout_cfg.n_action_samples = 1
# Агрегатор действий: "mean" | "lcb" | "thompson"
mc_dropout_cfg.action_agg = "mean"
mc_dropout_cfg.lcb_k = 0.5
# MC в таргетах (офф-полиси/bootstrap): ВЫКЛ
mc_dropout_cfg.use_for_target = False
mc_dropout_cfg.n_target_samples = 1
# Для таргетов: "mean_max" (рекомендуется) | "max_mean"
mc_dropout_cfg.target_agg = "mean_max"
# ✅ ВЕРНО: хранить контейнер на верхнем уровне MasterConfig (extra='allow').
# Тренер читает по приоритету: rl.mc_dropout → mc_dropout → cfg_mod.mc_dropout_cfg.
cfg.mc_dropout = mc_dropout_cfg
cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"

cfg.paper.source = "database"
# Установка плеча для бумажной торговли
cfg.paper.leverage = 2.0
# Список тикеров для симуляции в paper_trader.
# Если список пустой или None, будут использованы все тикеры из data/tickers.txt
cfg.paper.symbols = "ALL"
# cfg.paper.symbols = [
#     "OMUSDT", "1000RATSUSDT", "KAVAUSDT", "FILUSDT", "POPCATUSDT", "ZECUSDT",
#     "LUNA2USDT", "BRETTUSDT", "BELUSDT", "LISTAUSDT", "ZKUSDT", "PORTALUSDT",
#     "AUCTIONUSDT", "BIGTIMEUSDT", "TRBUSDT", "ARKMUSDT", "TIAUSDT", "NEOUSDT",
#     "IMXUSDT", "AXLUSDT", "MASKUSDT", "CATIUSDT", "REZUSDT"
# ]
# Опционально, для замедления симуляции (0.1 секунды на каждую минуту данных)
# cfg.paper.db_source_speed = 0.1 
cfg.backtest.data_source = "find_spikes"
cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"
# test_data_path отдельный или тот же что и для backtest
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz" 
# Модель для бэктеста.
cfg.paths.model_path = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\third_party\rl-trading-binance\output\alpha_aggressive_seed_260\saved_models\rl_binance_futures_trading_date_20251110_time_000932\best.pth"
cfg.paths.norm_stats_path = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\third_party\rl-trading-binance\output\alpha_aggressive_seed_260\saved_models\rl_binance_futures_trading_date_20251110_time_000932\norm_stats.json"
cfg.random_seed = 260
cfg.paths.config_name = "alpha_aggressive_seed_260"
# Spike Detector Configuration
cfg.detector.context_minutes = 30
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False # IMPORTANT: This should be True for backtesting/dataset creation
cfg.detector.abs_change_pct = 5.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 30

#   python train.py configs/alpha.py
#   python test_agent.py configs/alpha.py
#   python backtest_engine.py configs/alpha.py
#   python optimize_cfg.py configs/alpha.py
# Mini run with 10 short sessions
#   python optimize_cfg.py configs/alpha.py --trials 100 --jobs 1
# Notes: Default metric is values_0; default direction is max.
#   python get_info_from_optuna.py configs/alpha.py --n-best-trials 10
#   rm -r output/alpha
# Main workflow:
# Step                              Command
# 1. Train the model:               python train.py configs/...
# 2. Update the cache:              python backtest_engine.py configs/...  | When running a backtest, set backtest_mode = True
# 3. Run optimization:              python optimize_cfg.py configs/...
# 4. Show and save top-n trials:    python get_info_from_optuna.py configs/...

# ---------- Output/bundle paths & flags ----------
# Стандартизируем хранение артефактов: third_party/rl-trading-binance/output/<config_name>/
cfg.project_name = "rl_binance_futures_trading"
cfg.paths.base_output_dir = "third_party/rl-trading-binance/output"

# Управляющие флаги упаковки (используются в train.py):
bundle_cfg = type("obj", (), {})()
bundle_cfg.enable = True
# При необходимости можно включить snapshot кода; по умолчанию выключено.
bundle_cfg.include_code_snapshot = False
bundle_cfg.code_snapshot_paths = ["third_party/rl-trading-binance", "configs", "src", "utils.py"]
bundle_cfg.extra_files = []
try:
    cfg.bundle = bundle_cfg
except ValueError:
    pass # Поле 'bundle' не определено в MasterConfig, но train.py будет использовать bundle_cfg

# ---------- Prioritized Experience Replay (устойчивость) ----------
cfg.per.per_alpha = 0.6
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 1000000
cfg.per.per_eps = 1e-6

# ---------- Epsilon schedule (качественная, длинная эксплорация) ----------
# Долго держим исследование; низкий eps_end для аккуратной политики
cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.02
cfg.eps.eps_decay_frames = 3000000

# ─────────────────────────────────────────────────────────────
# Optuna Search Space (для optimize_cfg.py)
# ─────────────────────────────────────────────────────────────
optuna_search_space = {
    # Название параметра в Optuna | Тип | Нижняя граница | Верхняя граница | Лог. шкала | Путь в конфиге
    "long_thr":       ("suggest_float", 0.001,  0.03,   True,  "backtest.long_action_threshold"),
    "short_thr":      ("suggest_float", 0.001,  0.03,   True,  "backtest.short_action_threshold"),
    "pos_frac":       ("suggest_float", 0.10,   0.60,   False, "backtest.position_fraction"),
    "d_min":          ("suggest_float", 0.001,  0.005,  True,  "backtest.trailing_stop_min"),
    # Для d0 нижняя граница зависит от уже выбранного d_min
    "d0":             ("suggest_float", "d_min", 0.02,  True,  "backtest.trailing_stop"),
    "delta_p_hyst":   ("suggest_float", 0.0005, 0.005,  True,  "backtest.delta_p_hysteresis"),
    # "ensemble_max_sigma": ("suggest_float", 0.001, 0.015, True, "backtest.ensemble_max_sigma"),
}

cfg.optuna_search_space = optuna_search_space

# Для досрочной остановки
cfg.trainlog.early_stopping_patience = 10 # Остановить, если нет улучшений в течение 10 валидаций