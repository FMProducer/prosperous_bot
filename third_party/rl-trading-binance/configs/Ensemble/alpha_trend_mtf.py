# configs/alpha_trend_mtf.py
from configs.alpha import cfg as base_cfg
cfg = base_cfg

cfg.random_seed = 202

cfg.trainlog.episodes = 500_000
cfg.trainlog.num_val_ep = 3_500
cfg.trainlog.val_freq = 1_000

cfg.rl.batch_size = 128
cfg.rl.learning_rate = 2e-4
cfg.rl.gamma = 0.997
cfg.rl.n_step = 5
cfg.rl.train_start = 18_000
cfg.rl.target_update_steps = 4_000
cfg.rl.grad_clip_norm = 1.0

cfg.per.buffer_size = 300_000
cfg.vec.num_envs = 3

cfg.detector.context_minutes = 120
cfg.detector.window_minutes = 20
cfg.detector.cooldown_minutes = 40
cfg.detector.use_lookahead = False

cfg.signals.long_action_threshold = 0.0080
cfg.signals.short_action_threshold = 0.0080
cfg.signals.close_action_threshold = 0.014

cfg.risk.take_profit = None
cfg.risk.trailing_stop = 0.019
cfg.risk.trailing_stop_min = 0.0047
cfg.risk.delta_p_hysteresis = 0.0019

cfg.backtest.position_fraction = 0.40

cfg.selection_strategy = "ensemble_q_filter"
cfg.ensemble_n_samples = 5
cfg.ensemble_max_sigma = 0.01

cfg.agent_history_len = 60
cfg.agent_session_len = 20
cfg.ACTION_HISTORY_LEN = 3
cfg.cnn_maps = [32, 64, 128]
cfg.kernels = [7, 5, 3]
cfg.strides = [2, 1, 1]
cfg.dropout_p = 0.10
cfg.dense_val = [128, 64]
cfg.dense_adv = [128, 64]
