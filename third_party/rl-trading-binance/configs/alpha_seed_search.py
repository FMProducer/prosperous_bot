# configs/alpha_seed_search.py — Конфиг для быстрого поиска удачного seed
# Основан на alpha_seed_404_v13.py, но с параметрами для "1 тренировка + 1 валидация"

import torch
from config import cfg  # noqa: F401
from pathlib import Path
import json

# --- AGENT MODE SELECTOR ---
# Используем тот же режим, что и в целевом конфиге
AGENT_MODE = "LONG_ONLY" 

if AGENT_MODE == "UNIVERSAL":
    cfg.paths.config_name = "alpha_seed_search"
else:
    cfg.paths.config_name = f"alpha_seed_search_{AGENT_MODE}"

print(f"🚀 CONFIG LOADED: AGENT_MODE = {AGENT_MODE} (SEED SEARCH MODE)")

cfg.paths.model_dir = f"output/{cfg.paths.config_name}/saved_models"
cfg.paths.plot_dir = f"output/{cfg.paths.config_name}/plots"

# --- DATA PARAMS (Копия из v13) ---
cfg.num_channels = 10
cfg.state_shape = (10, 90, 1)
cfg.seq.full_seq_len = 150
cfg.seq.agent_history_len = 90
cfg.seq.agent_session_len = 60
cfg.seq.action_history_len = 2
cfg.seq.pre_signal_len = 90
cfg.seq.post_signal_len = 60
cfg.seq.state_shape = (10, 90, 1)
cfg.seq.input_history_len = 90
cfg.episodes_per_epoch = 10000

cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"
cfg.paths.norm_stats_path = "norm_stats.json"

# --- MODEL (Копия из v13) ---
cfg.model.cnn_maps = [64, 96, 128, 128, 96, 64]
cfg.model.cnn_kernels = [3, 3, 3, 3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4, 8, 16, 28]
cfg.model.cnn_strides = [1, 1, 1, 1, 1, 1]
cfg.model.dense_val = [128, 64, 32]
cfg.model.dense_adv = [128, 64, 32]
cfg.model.additional_feats = 10
cfg.model.dropout_p = 0.10

# --- MARKET (Копия из v13) ---
cfg.market.num_actions = 3
cfg.market.position_fraction = 0.10
cfg.market.transaction_fee = 0.0004
cfg.market.slippage = 0.0002
cfg.market.allow_opposite_trades = False

if AGENT_MODE == "LONG_ONLY":
    cfg.market.allowed_directions = ['LONG']
elif AGENT_MODE == "SHORT_ONLY":
    cfg.market.allowed_directions = ['SHORT']
else:
    cfg.market.allowed_directions = ['LONG', 'SHORT']
    cfg.market.filter_direction = None

# =============================================================================
# ПАРАМЕТРЫ ДЛЯ УСКОРЕНИЯ (SEED SEARCH)
# =============================================================================

# 1. RL Params - Быстрый старт обучения
cfg.rl.lr = 0.0003
cfg.rl.gamma = 0.99
cfg.rl.n_step = 5
cfg.rl.batch_size = 64
# ! ВАЖНО: Начинаем обучение почти сразу (через 1000 шагов), чтобы успеть обновить веса
cfg.rl.train_start = 1000  
cfg.rl.target_update_freq = 1000
cfg.rl.max_gradient_norm = 1.0

# 2. PER - Уменьшенный буфер для экономии памяти и скорости
cfg.per.buffer_size = 100000
cfg.per.per_alpha = 0.6
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 10000
cfg.per.per_eps = 1e-6

# 3. Epsilon - Быстрое затухание (или константа, если decay_frames > total_timesteps)
cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.05
cfg.eps.eps_decay_frames = 5000

# 4. Env - Используем параллелизм
cfg.vec.num_envs = 8
cfg.vec.backend = "subproc"
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True

# 5. Training Loop - КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ
# Всего 10 эпизодов. При 8 средах это 80 траекторий.
cfg.trainlog.episodes = 10 
# Бюджет шагов подгоняем под кол-во эпизодов (10 * 8 * 60 ~ 4800 шагов)
cfg.trainlog.total_timesteps = 10000

# Валидация: 1 раз в самом конце (на 10-м эпизоде)
cfg.trainlog.val_freq = 10 
# Отключаем прогрев, чтобы валидация точно сработала
cfg.trainlog.validation_warmup_steps = 0 

# Быстрая валидация (меньше эпизодов проверки)
cfg.trainlog.num_val_ep = 50 

cfg.trainlog.plot_top_n = 5
cfg.trainlog.available_metrics = [
    "Validation_mean_reward", "Validation_mean_pnl", "Validation_win_rate",
    "Validation_profit_factor", "Validation_max_drawdown", "Validation_net_pnl",
    "Validation_sharpe", "Validation_sortino"
]
# Сортируем сиды по Sortino
cfg.trainlog.val_selection_metrics = ["Validation_sortino"]
cfg.trainlog.early_stopping_patience = 5

# Ослабляем Validation Gate, чтобы скрипт не падал с ошибкой "No best_validation metrics"
# если модель на старте показывает плохие результаты. Нам нужно сравнить сиды, а не отфильтровать их.
cfg.validation_gate = {
    "min_sharpe": -999.0, 
    "min_sortino": -999.0,
    "min_profit_factor": 0.0,
    "max_drawdown_at_most": -1.0, # Допускаем просадку до 100%
    "min_win_rate": 0.0,
    "min_trades": 1,
    "deny_inf_pf": False,
    "deny_zero_drawdown": False
}

cfg.trainlog.save_top_k = 1 # Храним только 1 чекпоинт
cfg.trainlog.checkpoint_metric = "Validation_sortino"
cfg.trainlog.save_mode = "max"

# --- Rewards (Копия из v13) ---
cfg.market.new_equity_peak_reward = 0.0
cfg.market.perfect_entry_reward = 0.0
cfg.market.risk_reward_ratio_reward = 0.0
cfg.market.good_exit_bonus = 0.0
cfg.market.fast_exit_bonus = 0.0
cfg.market.bankruptcy_penalty = 10.0
cfg.market.bankruptcy_threshold = 0.0
cfg.market.bankruptcy_slippage_penalty = 0.05
cfg.market.max_drawdown_threshold = -0.10
cfg.market.max_drawdown_penalty_type = "proportional"
cfg.market.max_drawdown_penalty = 1.0
cfg.market.continuous_pain_penalty_ratio = 0.001
cfg.market.inaction_penalty_ratio = 0.0
cfg.market.low_balance_penalty = 1.0
cfg.market.holding_penalty_multiplier = 0.0
cfg.market.greed_penalty_multiplier = 0.0
cfg.market.premature_profit_exit_penalty = 0.0
cfg.market.holding_loss_penalty = 0.0
cfg.market.premature_exit_penalty = 0.0
cfg.market.profit_holding_bonus = 0.0
cfg.market.risk_reward_ratio_threshold = 0.0
cfg.market.profit_exit_threshold = 0
cfg.market.greed_penalty_threshold = 0.4
cfg.market.exit_quality_threshold = 0.8
cfg.market.loss_exit_threshold = 70
cfg.market.holding_penalty_threshold = 70
cfg.market.fast_exit_threshold = 70
cfg.market.premature_exit_threshold = 70

# Backtest
cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 4
cfg.backtest.position_fraction = 0.10
cfg.backtest.order_size_usdt = 0.0
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.015
cfg.backtest.short_action_threshold = -0.015
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.use_risk_management = False
cfg.backtest.trailing_stop = 0.04
cfg.backtest.exec_delay_bars = 1
cfg.backtest.plot_backtest_balance_curve = True
cfg.backtest.trailing_stop_min = 0.0005
cfg.backtest.fee_buffer_mult = 2.5
cfg.backtest.delta_p_hysteresis = 0.0015
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

# Perf
cfg.perf.use_amp = False
cfg.perf.amp_dtype = "float16"
cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None
cfg.perf.dataloader_num_workers = 0
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
cfg.perf.cudnn_benchmark = True

# MC-Dropout
mc_dropout_cfg = type("obj", (), {})()
mc_dropout_cfg.enable = False
cfg.mc_dropout = mc_dropout_cfg

# Misc
cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paper.source = "database"
cfg.paper.leverage = 1.0
cfg.paper.symbols = "ALL"
cfg.backtest.data_source = "npz"
cfg.random_seed = 404 # Будет перезаписан скриптом find_best_seed
cfg.logging.per_trial_logs = True
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
cfg.deterministic = False

# Bundle
bundle_cfg = type("obj", (), {})()
bundle_cfg.enable = True
bundle_cfg.include_code_snapshot = False
bundle_cfg.code_snapshot_paths = ["train.py", "model.py", "agent.py", "trading_environment.py"]
bundle_cfg.extra_files = ["norm_stats.json", "data_manifest.json"]
try:
    cfg.bundle = bundle_cfg
except ValueError:
    pass

# Spike Detector
cfg.detector.context_minutes = 40
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False
cfg.detector.abs_change_pct = 4.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 60

# Paths
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd()