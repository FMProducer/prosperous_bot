# configs/alpha_seed_404_v12.py — FULL GRADIENT v12 (все параметры сохранены из v8)

import torch
from config import cfg # noqa: F401
from pathlib import Path
import json # Для fallback norm_stats если нужно

# ============= CORE DATA PARAMS =============
# (10 channels: OHLCV + vol/taker_buy/trades)

cfg.num_channels = 10
cfg.state_shape = (10, 90, 1)  # Input для CNN: (C, L, 1) — окно истории 90
cfg.seq.full_seq_len = 150  # 90 контекст + 60 сессия
cfg.seq.agent_history_len = 90  # Context window (история)
cfg.seq.agent_session_len = 60  # Trading session length (60 шагов)
cfg.seq.action_history_len = 2  # Recent actions feat
cfg.seq.pre_signal_len = 90  # Старт эпизода после 90 баров истории
cfg.seq.post_signal_len = 60
cfg.seq.state_shape = (10, 90, 1)

# Явно фиксируем длину входного окна истории для env/model
cfg.seq.input_history_len = 90
cfg.episodes_per_epoch = 10000  # Sampling для memory (full 24k fallback)

cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"  # Или proxy
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"
cfg.paths.norm_stats_path = "norm_stats.json"  # Auto-generated

# ============= MODEL: ActorCritic CNN =============
# (dilated 1D Conv для ~60-min receptive)

cfg.model.cnn_maps = [64, 96, 128, 128, 96]  # Reduced для GTX1070
cfg.model.cnn_kernels = [3, 3, 3, 3, 3]
cfg.model.cnn_dilations = [1, 2, 4, 6, 9]  # RF=45 баров
cfg.model.cnn_strides = [1, 1, 1, 1, 1]
cfg.model.dense_val = [128, 64, 32]  # Value head
cfg.model.dense_adv = [128, 64, 32]  # Advantage/policy head
cfg.model.additional_feats = 12  # Pos + actions + time
cfg.model.dropout_p = 0.20

# ============= MARKET CONFIG =============

cfg.market.num_actions = 4  # Discrete: 0=hold, 1=buy, 2=sell, 3=close

# Market/Position Sizing (для обучения, НЕ только backtest!)
cfg.market.position_fraction = 0.10  # 10% баланса на сделку
cfg.market.transaction_fee = 0.0004
cfg.market.slippage = 0.0002

# ============= RL/DQN PARAMS =============
# (custom agent)

cfg.rl.lr = 3e-4  # AdamW
cfg.rl.gamma = 0.95  # Discount
cfg.rl.n_step = 60  # Steps per rollout == длина торговой сессии
cfg.rl.batch_size = 32  # Mini-batch (GTX fit)
cfg.rl.train_start = 15000  # Warmup steps
cfg.rl.target_update_freq = 2000  # Soft target? (DQN-style if needed)
cfg.rl.max_gradient_norm = 1.0  # Clip grads

# ============= DQN-SPECIFIC (PER/epsilon) =============

cfg.per.buffer_size = 1000000
cfg.per.per_alpha = 0.6
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 400000
cfg.per.per_eps = 1e-6

cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.05
cfg.eps.eps_decay_frames = 400000

# ============= ENV/VECTORIZED =============

cfg.vec.num_envs = 8  # параллельные среды
cfg.vec.backend = "subproc"  # сначала DummyVecEnv, потом можно subproc
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True  # Adjust eps decay

# ============= TRAINING LOG/VALIDATION =============

cfg.trainlog.num_val_ep = 750  # Val episodes (20% train)

# При 8 env один эпизод даёт больше шагов.
# Чтобы общий бюджет шагов остался ≈300k, эпизодов можно делать меньше.
cfg.trainlog.episodes = 7500  # норма 15000
cfg.trainlog.total_timesteps = 300000  # Бюджет шагов, норма 600000

# Валидация: масштабируем по эпизодам
cfg.trainlog.val_freq = 62  # норма 125
cfg.trainlog.validation_warmup_steps = 225000  # норма 450000
cfg.trainlog.plot_top_n = 10

cfg.trainlog.available_metrics = [
    "Validation_mean_reward", "Validation_mean_pnl", "Validation_win_rate",
    "Validation_profit_factor", "Validation_max_drawdown", "Validation_all_pnls",
    "Validation_sharpe", "Validation_sortino"
]

cfg.trainlog.val_selection_metrics = ["Validation_sortino", "Validation_sharpe", "Validation_profit_factor"]
cfg.trainlog.early_stopping_patience = 10

# ============= VALIDATION GATE =============
# (multi-crit; deny bad models)

cfg.validation_gate = {
    "min_sharpe": -0.15,
    "min_sortino": -0.18,
    "min_profit_factor": 0.63,
    "max_drawdown_at_most": -1.17,
    "min_win_rate": 0.42,
    "min_trades": 200,  # Ослабленный порог для промежуточных чекпоинтов
    "deny_inf_pf": True,
    "deny_zero_drawdown": True,
    "profit_factor_atleast": 0.63,
    "sortino_atleast": -0.18
}

# ============= TOP-K CHECKPOINT SAVING =============

cfg.trainlog.save_top_k = 10  # Сохранять топ-10 моделей
cfg.trainlog.checkpoint_metric = "Validation_sortino"  # Основная метрика для ранжирования
cfg.trainlog.save_mode = "max"  # Максимизировать метрику

# ============= REWARD SHAPING (FULL GRADIENT v12) =============

# --- Базовые параметры ---
cfg.market.base_reward_scale = 0.01  # Масштаб базовой награды
cfg.market.holding_cost = 0.0001  # Линейный cost за удержание

# --- GRADIENT EXIT BONUS (НОВОЕ v12) ---
cfg.market.exit_bonus = 0.05

# --- GRADIENT GREED PENALTY (НОВОЕ v12) ---
cfg.market.greed_penalty = 0.10

# --- GRADIENT PREMATURE PROFIT EXIT PENALTY (v11) ---
cfg.market.premature_profit_exit_penalty = 0.12
cfg.market.profit_exit_threshold = 5  # баров

# --- GRADIENT HOLDING LOSS PENALTY (v11) ---
cfg.market.holding_loss_penalty = 0.08
cfg.market.loss_exit_threshold = 3  # баров

# --- Отключить старые фиксированные параметры ---
cfg.market.premature_exit_penalty = 0.0  # старый фиксированный (ОТКЛЮЧЕН)
cfg.market.profit_holding_bonus = 0.0  # старый фиксированный (ОТКЛЮЧЕН)
cfg.market.inaction_penalty_ratio = 0.0  # пока не используем

# --- BONUSES (из v8, сохранены) ---

# Награда за достижение нового максимума эквити
cfg.market.new_equity_peak_reward = 0.005

# Награда за прибыльную сделку, которая не уходила в минус
cfg.market.perfect_entry_reward = 0.05

# Порог для соотношения риск/прибыль (3:1)
cfg.market.risk_reward_ratio_threshold = 3.0

# Награда за сделку с высоким соотношением риск/прибыль
cfg.market.risk_reward_ratio_reward = 0.075

# Бонус за хороший выход (закрытие сделки с >=80% от пиковой прибыли)
cfg.market.good_exit_bonus = 0.30

# Дополнительный бонус за быстрый выход (< 20 шагов)
cfg.market.fast_exit_bonus = 0.10

# --- PENALTIES (из v8, сохранены) ---

# Штраф за банкротство
cfg.market.bankruptcy_threshold = 0.0
cfg.market.bankruptcy_penalty = 1.0

# Штрафное проскальзывание при принудительной ликвидации
cfg.market.bankruptcy_slippage_penalty = 0.05

# Штраф за превышение максимальной просадки (MaxDD)
cfg.market.max_drawdown_threshold = -0.20
cfg.market.max_drawdown_penalty_type = "proportional"
cfg.market.max_drawdown_penalty = 1.0

# Штраф за удержание убыточной позиции (каждый шаг)
cfg.market.continuous_pain_penalty_ratio = 0.12

# Штраф за бездействие (когда нет открытых позиций)
cfg.market.inaction_penalty_ratio = 0.0

# Штраф за попытку торговли с низким балансом
cfg.market.low_balance_penalty = 0.01

# Множитель для прогрессивного штрафа за удержание убыточной позиции
cfg.market.holding_penalty_multiplier = 0.2

# "Штраф за жадность" (незафиксированная прибыль) - СТАРЫЙ, сохранен
cfg.market.greed_penalty_multiplier = 0.1

# --- THRESHOLDS FOR SHAPED REWARDS (из v8, сохранены) ---

# Порог времени удержания для начала прогрессивного штрафа (шагов)
cfg.market.holding_penalty_threshold = 15

# Порог отката прибыли для штрафа за жадность (0.50 = 50%)
cfg.market.greed_penalty_threshold = 0.50

# Порог качества выхода для бонуса (0.80 = 80% от максимальной прибыли)
cfg.market.exit_quality_threshold = 0.80

# Порог для бонуса за быстрый выход (шагов)
cfg.market.fast_exit_threshold = 20

# Порог для штрафа за преждевременный выход (шагов) - СТАРЫЙ
cfg.market.premature_exit_threshold = 8

# ============= BACKTEST/PAPER TRADER =============

cfg.backtest_mode = False
cfg.backtest.max_parallel_sessions = 4
cfg.backtest.position_fraction = 0.10
cfg.backtest.order_size_usdt = 0.0
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.015
cfg.backtest.short_action_threshold = -0.015  # Negative for short
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.use_risk_management = False
cfg.backtest.trailing_stop = 0.018
cfg.backtest.exec_delay_bars = 1
cfg.backtest.plot_backtest_balance_curve = True
cfg.backtest.trailing_stop_min = 0.005
cfg.backtest.fee_buffer_mult = 2.5
cfg.backtest.delta_p_hysteresis = 0.0015
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

# ============= PERF/OPTIMIZATION (GTX1070 opt) =============

cfg.perf.use_amp = True  # Mixed precision
cfg.perf.amp_dtype = "float16"
cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None  # None - no torch.compile
cfg.perf.dataloader_num_workers = 0  # Windows safe
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
cfg.perf.cudnn_benchmark = True

# ============= MC-DROPOUT (ensemble; off by default) =============

mc_dropout_cfg = type("obj", (), {})()
mc_dropout_cfg.enable = False
mc_dropout_cfg.n_action_samples = 1
mc_dropout_cfg.action_agg = "mean"
mc_dropout_cfg.lcb_k = 0.5
mc_dropout_cfg.use_for_target = False
mc_dropout_cfg.n_target_samples = 1
mc_dropout_cfg.target_agg = "mean_max"
cfg.mc_dropout = mc_dropout_cfg

# ============= DB/PAPER =============

cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paper.source = "database"
cfg.paper.leverage = 1.0
cfg.paper.symbols = "ALL"  # Or "ALL" or list from tickers.txt
cfg.backtest.data_source = "npz"  # For test/backtest

# ============= RANDOM/LOGGING =============

cfg.random_seed = 404
cfg.paths.config_name = "alpha_seed_404_v12"  # UPDATED для v12
cfg.logging.per_trial_logs = True
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
cfg.deterministic = False

# ============= BUNDLE (for saving artifacts) =============

bundle_cfg = type("obj", (), {})
bundle_cfg.enable = True
bundle_cfg.include_code_snapshot = False
bundle_cfg.code_snapshot_paths = ["train.py", "model.py", "agent.py", "trading_environment.py"]
bundle_cfg.extra_files = ["norm_stats.json", "data_manifest.json"]
try:
    cfg.bundle = bundle_cfg
except ValueError:
    pass  # Fallback in train.py

# ============= OPTUNA SEARCH SPACE =============
# (for hyperopt if needed; backtest thresholds)

cfg.optuna_search_space = {
    # Название параметра в Optuna | Тип | Нижняя граница | Верхняя граница | Лог. шкала | Путь в конфиге
    "long_thr": ("suggest_float", 0.001, 0.03, True, "backtest.long_action_threshold"),
    "short_thr": ("suggest_float", -0.03, -0.001, True, "backtest.short_action_threshold"),
    "pos_frac": ("suggest_float", 0.10, 0.60, False, "backtest.position_fraction"),
    "d_min": ("suggest_float", 0.001, 0.005, True, "backtest.trailing_stop_min"),
    # Для d0 нижняя граница зависит от уже выбранного d_min
    "d0": ("suggest_float", "d_min", 0.02, True, "backtest.trailing_stop"),
    "delta_p_hyst": ("suggest_float", 0.0005, 0.005, True, "backtest.delta_p_hysteresis"),
}

# ============= SPIKE DETECTOR =============
# (data prep; if regenerating)

cfg.detector.context_minutes = 40
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False
cfg.detector.abs_change_pct = 4.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 60

# ============= WORKFLOW NOTES =============
# Запуск из корня:
# python train.py --config alpha_seed_404_v12 --total_timesteps 10000  # Test
# python train.py --config alpha_seed_404_v12  # Full (300k timesteps)
# python paper_trader_q.py --model rl_model.pth --config alpha_seed_404_v12  # Backtest

# ============= CHANGELOG v12 =============
# 1. Добавлен cfg.reward.base_reward_scale = 0.01
# 2. Добавлен cfg.reward.holding_cost = 0.0001
# 3. Добавлен cfg.reward.exit_bonus = 0.05 (gradient v12)
# 4. Добавлен cfg.reward.greed_penalty = 0.10 (gradient v12)
# 5. Добавлен cfg.market.premature_profit_exit_penalty = 0.12 (gradient v11)
# 6. Добавлен cfg.market.profit_exit_threshold = 5 (gradient v11)
# 7. Добавлен cfg.market.holding_loss_penalty = 0.08 (gradient v11)
# 8. Добавлен cfg.market.loss_exit_threshold = 3 (gradient v11)
# 9. Отключен cfg.market.premature_exit_penalty = 0.0 (старый фиксированный)
# 10. Добавлен cfg.market.profit_holding_bonus = 0.0 (отключен, не было в v8)
# 11. Обновлено cfg.paths.config_name = "alpha_seed_404_v12"
# 12. ВСЕ остальные параметры из v8 СОХРАНЕНЫ без изменений
