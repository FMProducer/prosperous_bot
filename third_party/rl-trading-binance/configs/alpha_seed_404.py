# configs/alpha.py — Все параметры RL-тренировки в одном месте
import torch
from config import MasterConfig  # Только для BaseModel (пустая shell)
from pathlib import Path
import json  # Для fallback norm_stats если нужно

cfg = MasterConfig()  # Инициализация пустого Pydantic

# Core Data Params (10 channels: OHLCV + vol/taker_buy/trades)
cfg.num_channels = 10
cfg.state_shape = (10, 90, 1)   # Input для CNN: (C, L, 1) — окно истории 90
cfg.seq.full_seq_len = 150      # 90 контекст + 60 сессия
cfg.seq.agent_history_len = 90  # Context window (история)
cfg.seq.agent_session_len = 30  # Trading session length (60 шагов)
cfg.seq.action_history_len = 2  # Recent actions feat
cfg.seq.pre_signal_len = 90     # Старт эпизода после 90 баров истории
cfg.seq.post_signal_len = 60

# Явно фиксируем длину входного окна истории для env/model
cfg.seq.input_history_len = 90
cfg.episodes_per_epoch = 10000  # Sampling для memory (full 24k fallback) # This line was not in the diff but seems to belong with this block.
cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"  # Или proxy
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"
cfg.paths.norm_stats_path = "norm_stats.json"  # Auto-generated

# Model: ActorCritic CNN (dilated 1D Conv для ~60-min receptive)
cfg.model.cnn_maps = [64, 96, 128, 128, 96]  # Reduced для GTX1070 (vs [64,96,...])
cfg.model.cnn_kernels = [3, 3, 3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4, 8, 16]  # Receptive ~150+ bars
cfg.model.cnn_strides = [1, 1, 1, 1, 1]
cfg.model.dense_val = [128, 64, 32]  # Value head
cfg.model.dense_adv = [128, 64, 32]  # Advantage/policy head
cfg.model.additional_feats = 12  # Pos + actions + time
cfg.model.dropout_p = 0.15

# Market Config - ДОБАВЬТЕ ЭТУ СТРОКУ
cfg.market.num_actions = 4  # Discrete: 0=hold, 1=buy, 2=sell, 3=close

# RL/DQN Params (custom agent)
cfg.rl.lr = 2e-5  # AdamW
cfg.rl.gamma = 0.9995         # Discount
cfg.rl.n_step = 10   # Steps per rollout == длина торговой сессии
cfg.rl.batch_size = 32  # Mini-batch (GTX fit)
cfg.rl.train_start = 15000  # Warmup steps
cfg.rl.target_update_freq = 5000   # Soft target? (DQN-style if needed)
cfg.rl.max_gradient_norm = 	3.0  # Clip grads

# DQN-specific (PER/epsilon)
cfg.per.buffer_size = 500000
cfg.per.per_alpha = 0.7
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 30000       # 50000 * (600000 / 1000000) ≈ 30000
cfg.per.per_eps = 1e-6
cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.05
cfg.eps.eps_decay_frames = 600000     # чтобы ε-декей шёл на том же интервале реальных шагов

# Env/Vectorized
cfg.vec.num_envs = 4             # 4 параллельные среды
cfg.vec.backend = "subproc"        # сначала DummyVecEnv, потом можно subproc
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True  # Adjust eps decay

# Training Log/Validation
cfg.trainlog.num_val_ep = 1500      # Val episodes (10% train)

# При 4 env один эпизод даёт ~4× больше шагов.
# Чтобы общий бюджет шагов остался ≈600k, эпизодов можно делать ~в 4 раза меньше.
cfg.trainlog.episodes = 15000       # 4× меньше эпизодов при 4 env -> ~тот же total_steps
cfg.trainlog.total_timesteps = 600000  # Бюджет шагов оставляем прежним

# Валидация: масштабируем по эпизодам, чтобы частота и прогрев соответствовали новому числу эпизодов.
cfg.trainlog.val_freq = 125              # было 500; 500 * 15000 / 60000 ≈ 125
cfg.trainlog.validation_warmup_steps = 150000  # было 585000; прогрев ≈ 1/4 от полного бюджета
cfg.trainlog.plot_top_n = 10
cfg.trainlog.available_metrics = [
    "Validation_mean_reward", "Validation_mean_pnl", "Validation_win_rate",
    "Validation_profit_factor", "Validation_max_drawdown", "Validation_all_pnls",
    "Validation_sharpe", "Validation_sortino"
]
cfg.trainlog.val_selection_metrics = ["Validation_sharpe", "Validation_sortino", "Validation_profit_factor"]
cfg.trainlog.early_stopping_patience = 10

# Validation Gate (multi-crit; deny bad models) "max_drawdown_at_most": -0.30, 
cfg.validation_gate = {
    "min_sharpe": 0.001, "min_sortino": 0.001, "min_profit_factor": 1.00,
    "min_win_rate": 0.41, "min_trades": 600,
    "deny_inf_pf": True, "deny_zero_drawdown": True
}

# Штраф за банкротство
cfg.market.bankruptcy_threshold = 0.0  # Порог, ниже которого эквити считается банкротом
cfg.market.bankruptcy_penalty = 1.0    # Размер штрафа (очень большая отрицательная награда)

# Штраф за превышение максимальной просадки (MaxDD)
cfg.market.max_drawdown_threshold = -0.20  # Порог просадки (-20%). Штраф применяется, если MaxDD < этого значения.
cfg.market.max_drawdown_penalty_type = "proportional"  # 'proportional' или 'constant'.
cfg.market.max_drawdown_penalty = 1.0      # Коэффициент для штрафа. Начните с 0.1-0.5.


# Backtest/Paper Trader
cfg.backtest_mode = False
cfg.backtest.max_parallel_sessions = 4
cfg.backtest.position_fraction = 0.1
cfg.backtest.order_size_usdt = 0.0
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.015
cfg.backtest.short_action_threshold = -0.015  # Negative for short
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.use_risk_management = True
cfg.backtest.trailing_stop = 0.018
cfg.backtest.exec_delay_bars = 1
cfg.backtest.plot_backtest_balance_curve = True
cfg.backtest.trailing_stop_min = 0.005
cfg.backtest.fee_buffer_mult = 2.0
cfg.backtest.delta_p_hysteresis = 0.0015
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

# Perf/Perf (GTX1070 opt)
cfg.perf.use_amp = True  # Mixed precision
cfg.perf.amp_dtype = "float16"
cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None  # No torch.compile (old CUDA)
cfg.perf.dataloader_num_workers = 0  # Windows safe
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
cfg.perf.cudnn_benchmark = True

# MC-Dropout (ensemble; off by default)
mc_dropout_cfg = type("obj", (), {})()
mc_dropout_cfg.enable = False
mc_dropout_cfg.n_action_samples = 1
mc_dropout_cfg.action_agg = "mean"
mc_dropout_cfg.lcb_k = 0.5
mc_dropout_cfg.use_for_target = False
mc_dropout_cfg.n_target_samples = 1
mc_dropout_cfg.target_agg = "mean_max"
cfg.mc_dropout = mc_dropout_cfg

# DB/Paper (if needed)
cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paper.source = "database"
cfg.paper.leverage = 2.0
cfg.paper.symbols = "ALL"  # Or list from tickers.txt
cfg.backtest.data_source = "npz"  # For test/backtest

# Random/Logging
cfg.random_seed = 404
cfg.paths.config_name = "alpha_seed_404"
cfg.logging.per_trial_logs = True
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
cfg.deterministic = False

# Bundle (for saving artifacts)
bundle_cfg = type("obj", (), {})()
bundle_cfg.enable = True
bundle_cfg.include_code_snapshot = False
bundle_cfg.code_snapshot_paths = ["train.py", "model.py", "agent.py", "trading_environment.py"]
bundle_cfg.extra_files = ["norm_stats.json", "data_manifest.json"]
try:
    cfg.bundle = bundle_cfg
except ValueError:
    pass  # Fallback in train.py

# Optuna Search Space (for hyperopt if needed; backtest thresholds)
cfg.optuna_search_space = {
    # Название параметра в Optuna | Тип | Нижняя граница | Верхняя граница | Лог. шкала | Путь в конфиге
    "long_thr":       ("suggest_float", 0.001,  0.03,   True,  "backtest.long_action_threshold"),
    "short_thr":      ("suggest_float", -0.03,  -0.001, True,  "backtest.short_action_threshold"),
    "pos_frac":       ("suggest_float", 0.10,   0.60,   False, "backtest.position_fraction"),
    "d_min":          ("suggest_float", 0.001,  0.005,  True,  "backtest.trailing_stop_min"),
    # Для d0 нижняя граница зависит от уже выбранного d_min
    "d0":             ("suggest_float", "d_min", 0.02,  True,  "backtest.trailing_stop"),
    "delta_p_hyst":   ("suggest_float", 0.0005, 0.005,  True,  "backtest.delta_p_hysteresis"),
    # "ensemble_max_sigma": ("suggest_float", 0.001, 0.015, True, "backtest.ensemble_max_sigma"),
}

# Spike Detector (data prep; if regenerating)
cfg.detector.context_minutes = 40
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False
cfg.detector.abs_change_pct = 4.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 60

# Workflow notes (run from root):
# python train.py --config alpha.py --total_timesteps 10000  # Test
# python train.py --config alpha.py  # Full
# python paper_trader_q.py --model rl_model.pth --config alpha.py  # Backtest
