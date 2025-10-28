# configs/alpha_pullback.py
from configs.alpha import cfg as base_cfg
cfg = base_cfg

cfg.random_seed = 303

cfg.trainlog.episodes = 320_000
cfg.trainlog.num_val_ep = 2_500
cfg.trainlog.val_freq = 1_000

cfg.rl.batch_size = 64
cfg.rl.learning_rate = 3e-4
cfg.rl.gamma = 0.993
cfg.rl.n_step = 3
cfg.rl.train_start = 12_000
cfg.rl.target_update_steps = 2_500
cfg.rl.grad_clip_norm = 1.0

cfg.per.buffer_size = 180_000
cfg.vec.num_envs = 2

cfg.detector.context_minutes = 45
cfg.detector.window_minutes = 9
cfg.detector.cooldown_minutes = 18
cfg.detector.use_lookahead = False

cfg.signals.long_action_threshold = 0.0072
cfg.signals.short_action_threshold = 0.0070
cfg.signals.close_action_threshold = 0.011

cfg.risk.take_profit = None
cfg.risk.trailing_stop = 0.019
cfg.risk.trailing_stop_min = 0.0047
cfg.risk.delta_p_hysteresis = 0.0019

cfg.backtest.position_fraction = 0.38

cfg.selection_strategy = "ensemble_q_filter"
cfg.ensemble_n_samples = 5
cfg.ensemble_max_sigma = 0.01

cfg.agent_history_len = 20
cfg.agent_session_len = 8
cfg.ACTION_HISTORY_LEN = 2
cfg.cnn_maps = [64, 64, 96]
cfg.kernels = [5, 3, 3]
cfg.strides = [1, 1, 1]
cfg.dropout_p = 0.05
cfg.dense_val = [96, 48]
cfg.dense_adv = [96, 48]
