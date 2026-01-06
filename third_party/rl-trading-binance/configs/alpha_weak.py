# configs/alpha_weak.py — Оптимизированный конфиг для слабого железа
import torch
from config import cfg  # noqa: F401
from pathlib import Path
import json

# Имя конфига для папок вывода
cfg.paths.config_name = "alpha_weak"
cfg.paths.model_dir = f"output/{cfg.paths.config_name}/saved_models"
cfg.paths.plot_dir = f"output/{cfg.paths.config_name}/plots"

# --- DATA (Уменьшаем окно истории для экономии памяти) ---
cfg.num_channels = 10
cfg.state_shape = (10, 60, 1)   # Input: (C, L, 1) — окно истории 60 (было 90)
cfg.seq.full_seq_len = 90       # 60 контекст + 30 сессия
cfg.seq.agent_history_len = 60
cfg.seq.agent_session_len = 30  # Короткая сессия для частых обновлений
cfg.seq.action_history_len = 2
cfg.seq.pre_signal_len = 60
cfg.seq.post_signal_len = 30
cfg.seq.input_history_len = 60

cfg.episodes_per_epoch = 2000   # Меньше семплов за эпоху
cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"
cfg.paths.norm_stats_path = "norm_stats.json"

# --- MODEL (Lightweight Architecture) ---
# Упрощенная CNN: меньше слоев и каналов для экономии VRAM и FLOPs
cfg.model.cnn_maps = [32, 64, 64]
cfg.model.cnn_kernels = [3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4]
cfg.model.cnn_strides = [1, 1, 1]
cfg.model.dense_val = [64, 32]
cfg.model.dense_adv = [64, 32]
cfg.model.additional_feats = 10
cfg.model.dropout_p = 0.1

# --- MARKET ---
cfg.market.num_actions = 3      # 0=hold, 1=buy, 2=sell
cfg.market.position_fraction = 0.10
cfg.market.transaction_fee = 0.0004
cfg.market.slippage = 0.0002
cfg.market.allow_opposite_trades = False
cfg.market.allowed_directions = ['LONG', 'SHORT']

# --- RL/DQN Params ---
cfg.rl.lr = 3e-4
cfg.rl.gamma = 0.99
cfg.rl.n_step = 5
cfg.rl.batch_size = 32          # Маленький батч (можно 16 если совсем мало VRAM)
cfg.rl.train_start = 2000       # Быстрый старт обучения
cfg.rl.target_update_freq = 1000
cfg.rl.max_gradient_norm = 1.0

# --- PER (RAM Optimization) ---
# ! КРИТИЧНО: Уменьшаем буфер с 1M до 50k. Экономит ~8-10 ГБ RAM.
cfg.per.buffer_size = 50000
cfg.per.per_alpha = 0.6
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 100000
cfg.per.per_eps = 1e-6

# --- EPSILON ---
cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.05
cfg.eps.eps_decay_frames = 100000

# --- ENV (CPU Optimization) ---
# ! КРИТИЧНО: 2 среды вместо 8-12. Разгружает CPU.
cfg.vec.num_envs = 2
cfg.vec.backend = "subproc"
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True

# --- TRAINING LOOP ---
cfg.trainlog.episodes = 5000
cfg.trainlog.total_timesteps = 150000 # Меньше шагов
cfg.trainlog.val_freq = 100           # Реже валидация
cfg.trainlog.num_val_ep = 50          # Быстрая валидация (всего 50 эпизодов)
cfg.trainlog.validation_warmup_steps = 1000
cfg.trainlog.plot_top_n = 5
cfg.trainlog.available_metrics = [
    "Validation_mean_reward", "Validation_win_rate",
    "Validation_profit_factor", "Validation_sortino"
]
cfg.trainlog.val_selection_metrics = ["Validation_sortino"]
cfg.trainlog.early_stopping_patience = 10
cfg.trainlog.save_top_k = 3           # Хранить только 3 лучших модели
cfg.trainlog.checkpoint_metric = "Validation_sortino"
cfg.trainlog.save_mode = "max"

# --- VALIDATION GATE (Ослабленные требования) ---
cfg.validation_gate = {
    "min_sharpe": 0.0,
    "min_sortino": 0.0,
    "min_profit_factor": 1.0,
    "min_trades": 50,
    "deny_inf_pf": False,
    "deny_zero_drawdown": False
}

# --- REWARDS (Minimal) ---
cfg.market.new_equity_peak_reward = 0.0
cfg.market.perfect_entry_reward = 0.0
cfg.market.risk_reward_ratio_reward = 0.0
cfg.market.good_exit_bonus = 0.0
cfg.market.fast_exit_bonus = 0.0
cfg.market.bankruptcy_penalty = 1.0
cfg.market.max_drawdown_penalty = 0.0
cfg.market.continuous_pain_penalty_ratio = 0.0

# --- BACKTEST ---
cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 2 # Мало потоков
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
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

# --- PERF ---
cfg.perf.use_amp = True  # Включить Mixed Precision (экономит VRAM)
cfg.perf.amp_dtype = "float16"
cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None
cfg.perf.dataloader_num_workers = 0
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.cudnn_benchmark = True

# --- MC DROPOUT (OFF) ---
mc_dropout_cfg = type("obj", (), {})()
mc_dropout_cfg.enable = False  # Выключено для скорости
mc_dropout_cfg.n_action_samples = 1
mc_dropout_cfg.action_agg = "mean"
cfg.mc_dropout = mc_dropout_cfg

# --- MISC ---
cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paper.source = "database"
cfg.paper.leverage = 1.0
cfg.paper.symbols = "ALL"
cfg.backtest.data_source = "npz"
cfg.random_seed = 42
cfg.logging.per_trial_logs = True
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
cfg.deterministic = False

# --- ENSEMBLE ---
cfg.ensemble = type("obj", (), {})()
cfg.ensemble.enable_long = False
cfg.ensemble.enable_short = False