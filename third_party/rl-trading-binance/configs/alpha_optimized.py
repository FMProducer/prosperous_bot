# configs/alpha_optimized.py
# Этот файл был сгенерирован автоматически на основе лучших результатов
# оптимизации от 2025-10-26 14:40:25.

from config import MasterConfig

cfg = MasterConfig()

ACTION_HISTORY_LEN = 3

cfg.model.cnn_maps = [32, 64, 128]
cfg.model.cnn_kernels = [7, 5, 3]
cfg.model.cnn_strides = [2, 1, 1]
cfg.model.dense_val = [128, 64]
cfg.model.dense_adv = [128, 64]
cfg.model.additional_feats = 4 + ACTION_HISTORY_LEN * 4
cfg.model.dropout_p = 0.1

cfg.trainlog.num_val_ep = 3500
cfg.trainlog.val_freq = 1000
cfg.trainlog.episodes = 55_000
cfg.trainlog.plot_top_n = 10

cfg.per.buffer_size = 230_000

cfg.rl.batch_size = 64
cfg.rl.learning_rate = 1e-4
cfg.rl.train_start = 10_000

cfg.seq.agent_history_len = 30
cfg.seq.agent_session_len = 10
if not hasattr(cfg.seq, "pre_signal_len"):
    cfg.seq.pre_signal_len = cfg.seq.agent_history_len
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 2
cfg.backtest.position_fraction = 0.5
cfg.backtest.selection_strategy = "advantage_based_filter"

# --- OPTIMIZED PARAMETERS ---
# Best trial #1: PnL=1.71%, Accuracy=46.84%, trades=79.0
cfg.backtest.long_action_threshold = 0.020973010239653107
cfg.backtest.short_action_threshold = 0.006528846367079477
cfg.backtest.close_action_threshold = 0.0180067279703063 # (from original alpha.py, not optimized)

cfg.backtest.use_risk_management = True
cfg.backtest.stop_loss = 0.014933274013593328
cfg.backtest.take_profit = 0.03145626824160851
cfg.backtest.trailing_stop = 0.014552769233872755
# --- END OF OPTIMIZED PARAMETERS ---

cfg.backtest.ensemble_n_samples = 5
cfg.backtest.ensemble_max_sigma = 0.01
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.plot_backtest_balance_curve = True
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-10-01T00:00:00Z"}
cfg.random_seed = 25

cfg.logging.per_trial_logs = False
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False

cfg.perf.use_amp = False
cfg.perf.amp_dtype = "float16"
cfg.perf.compile_mode = None
cfg.perf.compile_dynamic = False
cfg.perf.dataloader_num_workers = 4
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
cfg.perf.cudnn_benchmark = True

cfg.vec.num_envs = 4
cfg.vec.backend = "dummy"
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True

cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paths.norm_stats_path = "C:\\Python\\Prosperous_Bot\\third_party\\rl-trading-binance\\output\\alpha\\norm_stats.json"

cfg.paper.source = "database"
cfg.paper.symbols = "ALL"

cfg.backtest.data_source = "find_spikes"
cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/val_data_fair_2m.npz"
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"

cfg.paths.model_path = r"C:\Python\Prosperous_Bot\output\alpha\saved_models\rl_binance_futures_trading_date_20251025_time_025141\best.pth"

cfg.detector.context_minutes = 30
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False
cfg.detector.abs_change_pct = 5.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 30