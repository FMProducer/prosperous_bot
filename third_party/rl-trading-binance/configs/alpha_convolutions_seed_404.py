# configs/alpha.py
import torch
from config import MasterConfig
import logging

logger = logging.getLogger(__name__)
cfg = MasterConfig()
ACTION_HISTORY_LEN = 2 # This is already defined in the base config, but we keep it for clarity
cfg.model.cnn_maps = [64, 96, 96, 96, 64]  # Stable, ~2.4M params
cfg.model.cnn_kernels = [3, 3, 3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4, 8, 16]  # Multi-scale receptive field
cfg.model.cnn_strides = [1, 1, 1, 1, 1]
cfg.model.dense_val = [128, 64, 32]
cfg.model.dense_adv = [128, 64, 32]
# 4 + action_history_len * num_actions
cfg.model.additional_feats = 12
# 0 ≤ p < 0.5; typical values are 0.1–0.2
cfg.model.dropout_p = 0.15
# Для устойчивого отбора чекпоинтов на GTX 1070 + 6C/12T
cfg.trainlog.num_val_ep = 100  # 100 validation episodes per checkpoint (good quality)
cfg.trainlog.val_freq = 500  # Validate every 500 episodes (~1 min per 500 on GTX 1070)
# Период "прогрева" после train_start, в течение которого валидация пропускается.
# Позволяет модели стабилизироваться перед первой оценкой.
cfg.trainlog.validation_warmup_steps = 10000  # Warmup 10000 steps before val (stabilization)
# Увеличиваем общий горизонт обучения (качество > скорость)
cfg.trainlog.episodes = 60000  # Full RL horizon
cfg.trainlog.plot_top_n = 10
cfg.per.buffer_size = 1000000  # 1M capacity for longer training
cfg.rl.batch_size = 256  # Larger batch, stable gradients
# Стабильнее обновления с меньшим шагом
cfg.rl.learning_rate = 1e-5  # Stable LR for DuelingQNetwork
cfg.rl.train_start = 15000
cfg.rl.gamma = 0.9995
cfg.rl.n_step = 20
# стабильнее целевые обновления
cfg.rl.target_update_freq = 1000  # Target net sync every 1000 steps
# Чуть мягче клиппинг — меньше «зажимаем» обучение, но защищаемся от выбросов
cfg.rl.max_gradient_norm = 3.0
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
    "Validation_sortino",      # Приоритет №1: Прибыль, скорректированная на риск плохих исходов.
    "Validation_sharpe",       # Приоритет №2: Стандартная индустриальная метрика.
]
# ── Валидационный гейт для отбора best.pth.
# ВАЖНО: TrainLogConfig запрещает extra-поля, поэтому кладём гейт на верхний уровень MasterConfig:
# train.py теперь читает fallback из cfg.validation_gate.
cfg.validation_gate = {
    "min_sharpe": 0.002, # These are placeholders, adjust based on expected performance
    "min_sortino": 0.006, # These are placeholders, adjust based on expected performance
    "min_profit_factor": 1.00,
    # Максимально допустимая просадка (20%). Меньшая просадка = большее число (e.g., -0.10 > -0.20).
    "max_drawdown_at_most": -0.25,
    "min_win_rate": 0.44,   # 0..1, placeholder
    "min_trades": 600,      # минимум сделок на валидации
    "deny_inf_pf": True,    # запрещаем PF=inf
    # Запрещаем модели с нулевой просадкой
    "deny_zero_drawdown": True,
}
# Широкий взгляд на рынок для оценки волатильности и риска
cfg.seq.agent_history_len = 90
# cfg.seq.input_history_len = 90
cfg.seq.agent_session_len = 60
# NB: pre_signal_len используется в utils.compute_metrics; согласуем с agent_history_len при отсутствии явной настройки.
if not hasattr(cfg.seq, "pre_signal_len"):
    cfg.seq.pre_signal_len = cfg.seq.agent_history_len
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = False
cfg.market.initial_balance = 10000.0  # 10k USD starting balance
cfg.backtest.max_parallel_sessions = 2 # This is for session-based backtester, not envs
cfg.backtest.position_fraction = 0.1  # 10% balance per position
cfg.backtest.order_size_usdt = 0.0 # Use position_fraction instead
# ["advantage_based_filter", "ensemble_q_filter"]
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.015
cfg.backtest.short_action_threshold = 0.015
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
# use_risk_management (from Trial #187)
cfg.backtest.use_risk_management = True
# Backtest params (production)
cfg.backtest.stop_loss = 0.045   # 4.5% stop loss
cfg.backtest.take_profit = 0.072   # 7.2% take profit
cfg.backtest.trailing_stop = 0.018  # 1.8% trailing stop loss

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

cfg.logging.per_trial_logs = True

# 1000, default = None
cfg.debug.debug_max_size_data = None  # Full sequences (~100-500 per split, realistic volume)

# DB overrides: periods для split (UTC, YYYY-MM-DD); symbols optional для ускорения
cfg.db.train_period_start = "2025-01-01"
cfg.db.train_period_end = "2025-06-01"  # 6 месяцев train data
cfg.db.val_period_start = "2025-06-01"
cfg.db.val_period_end = "2025-06-30"    # 1 месяц validation (post train)
cfg.db.test_period_start = "2025-08-01"
cfg.db.test_period_end = "2025-08-31"   # 1 месяц OOS test (Aug 2025 for paper trading)
cfg.db.symbols = None  # Auto-select top 10 symbols by volume (BTC, ETH, SOL, ...)

cfg.debug.use_final_model = False
# AMP: Включаем для ускорения на GPU (Tensor Cores).
cfg.perf.use_amp = True
cfg.perf.amp_dtype = "float16"
# torch.compile: Отключено, т.к. GTX 1070 (CUDA 6.1) не поддерживается компилятором Triton (требуется >= 7.0).
cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None
cfg.perf.compile_dynamic = False
# DataLoader: num_workers > 0 требует multiprocessing. persistent_workers=True сокращает оверхед.
cfg.perf.dataloader_num_workers = 0 # Отключить
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
# CuDNN Heuristics
cfg.perf.cudnn_benchmark = True
# ---- Vectorized Environments ----
# Увеличим количество параллельных сред для ускорения сбора данных.
# На Windows/спавн backend "subproc" может оказаться медленнее из-за накладных расходов spawn.
cfg.vec.num_envs = 4  # Parallel envs: 4x faster training, good for GTX 1070
# По умолчанию используем "dummy"
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
cfg.backtest.data_source = "find_spikes"
# DB mode: ignore .npz; set None (train.py skips if dsn set)
cfg.paths.train_data_path = None
cfg.paths.val_data_path = None
cfg.paths.test_data_path = None
cfg.paths.backtest_data_path = None  # Если используется

# 7 channels, match DB SELECT
cfg.data.data_channels = ["open", "high", "low", "close", "volume", "num_trades", "quote_volume"]
cfg.data.price_channels = ["open", "high", "low", "close"]
cfg.data.volume_channels = ["volume", "quote_volume"]  # Both volumes for log normalization
cfg.data.other_channels = ["num_trades"]  # Trade count as other
cfg.data.expected_channels = ["open", "high", "low", "close", "volume", "num_trades", "quote_volume"]  # 7 channels, match DB SELECT
logger.info(f"Config channels: data={len(cfg.data.data_channels)}, expected={len(cfg.data.expected_channels)}")
logger.info(f"Volume channels: {cfg.data.volume_channels}, other={cfg.data.other_channels}")

# Production logging
cfg.trainlog.plot_freq = 100  # Plot every 100 episodes (less spam)
cfg.trainlog.verbose_validation = False  # Less verbose

cfg.random_seed = 404
cfg.paths.config_name = "alpha_convolutions_seed_404"
# Spike Detector Configuration
cfg.detector.context_minutes = 90
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False # IMPORTANT: This should be True for backtesting/dataset creation
cfg.detector.abs_change_pct = 1.0  # 0.5 слишком loose, 1.0 standard (catch significant moves)
cfg.detector.contrast_min = 2.0    # 1.5 loose for test, 2.0 standard (signal vs noise)
cfg.detector.cooldown_minutes = 5

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
cfg.per.per_alpha = 0.6  # Prioritized experience replay
cfg.per.per_beta_start = 0.4   # Beta for importance sampling
cfg.per.per_beta_frames = 100000
cfg.per.per_eps = 1e-6

# ---------- Epsilon schedule (качественная, длинная эксплорация) ----------
# Долго держим исследование; низкий eps_end для аккуратной политики
cfg.eps.eps_start = 1.0 # Epsilon start
cfg.eps.eps_end = 0.01  # End exploration
cfg.eps.eps_decay_frames = 100000  # Gradual decay over training

# ---------- Env config (production) ----------
cfg.market.slippage = 0.0001  # 0.01% slippage
cfg.market.transaction_fee = 0.0004  # 0.04% maker/taker fee (Binance futures)
cfg.market.inaction_penalty_ratio = 0.1  # Penalty for no action

# ---------- Training validation (production) ----------
cfg.trainlog.validation_gate = True  # Enable quality gate
cfg.trainlog.save_best_only = True   # Save only best validation
cfg.trainlog.checkpoint_freq = 500   # Checkpoints every 500 episodes

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