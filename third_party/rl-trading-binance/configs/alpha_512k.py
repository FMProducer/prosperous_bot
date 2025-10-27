# configs/alpha_512k.py
from config import MasterConfig

cfg = MasterConfig()

ACTION_HISTORY_LEN = 3

cfg.model.cnn_maps = [32, 64, 128]
cfg.model.cnn_kernels = [7, 5, 3]
cfg.model.cnn_strides = [2, 1, 1]
cfg.model.dense_val = [128, 64]
cfg.model.dense_adv = [128, 64]
# 4 + action_history_len * num_actions
cfg.model.additional_feats = 4 + ACTION_HISTORY_LEN * 4
# 0 ≤ p < 0.5; typical values are 0.1–0.2
cfg.model.dropout_p = 0.1
# Для устойчивого отбора чекпоинтов на GTX 1070 + 6C/12T
cfg.trainlog.num_val_ep = 3500
cfg.trainlog.val_freq = 1000
# gradient steps ~ 512_000
cfg.trainlog.episodes = 110_000
cfg.trainlog.plot_top_n = 10

cfg.per.buffer_size = 230_000

cfg.rl.batch_size = 64
cfg.rl.learning_rate = 1e-4
cfg.rl.train_start = 10_000

# Удлинённый контекст/сессии для повышения качества (см. коммиты от 2025-10-12)
cfg.seq.agent_history_len = 30
cfg.seq.agent_session_len = 10
# NB: pre_signal_len используется в utils.compute_metrics; согласуем с agent_history_len при отсутствии явной настройки.
if not hasattr(cfg.seq, "pre_signal_len"):
    cfg.seq.pre_signal_len = cfg.seq.agent_history_len
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 2
cfg.backtest.position_fraction = 0.5
# ["advantage_based_filter", "ensemble_q_filter"]
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.0078056307732368
cfg.backtest.short_action_threshold = 0.0091301259923296
cfg.backtest.close_action_threshold = 0.0180067279703063
cfg.backtest.ensemble_n_samples = 5
# maximum allowed variance (uncertainty) (range: 0.001 to 0.015)
cfg.backtest.ensemble_max_sigma = 0.01
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
# use_risk_management (from Trial #187)
cfg.backtest.use_risk_management = True
cfg.backtest.stop_loss = 0.0153228603723445
cfg.backtest.take_profit = 0.0447149987567144
cfg.backtest.trailing_stop = 0.018986
cfg.backtest.plot_backtest_balance_curve = True
# --- NEW: Linear Trailing Stop Parameters for Optuna ---
# d_min: нижний пол для отступа трейла (напр. 0.2–0.6%)
cfg.backtest.trailing_stop_min = 0.004740
# Множитель для fee_buf = fee_buffer_mult * fee (обычно ~2.0)
cfg.backtest.fee_buffer_mult = 2.0
# --- NEW: Hysteresis for TSL updates ---
cfg.backtest.delta_p_hysteresis = 0.001890
# --- NEW: Explicit time range for backtesting ---
# This ensures the backtest runs on the correct, unseen data period.
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-10-01T00:00:00Z"}
cfg.random_seed = 25 # или любое другое целое число

cfg.logging.per_trial_logs = False
# 1000,  default = None
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False

# --------------------------- 
# ⚡ Performance (hardware-tuned for GTX 1070 + i5-6600)
# Управление ускорением ТОЛЬКО конфигом, чтобы не ломать кодовую базу.
# AMP: экономия VRAM и потенциальный прирост на свертках; на Pascal (GTX 1070) и новее FP16 даёт ускорение и экономию памяти.
cfg.perf.use_amp = False
cfg.perf.amp_dtype = "float16"
# torch.compile: снижает overhead Python-графа; pemilihan "reduce-overhead" — наиболее безопасный.
cfg.perf.compile_mode = None
cfg.perf.compile_dynamic = False
# DataLoader: загрузка с CPU (4 физ. ядра). Для коротких сессий — умеренные значения.
cfg.perf.dataloader_num_workers = 4
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
# CuDNN Heuristics
cfg.perf.cudnn_benchmark = True

# ---- Vectorized Environments ----
# Увеличим количество параллельных сред для ускорения сбора данных.
# На Windows/спавн backend "subproc" может оказаться медленнее из-за накладных расходов spawn.
cfg.vec.num_envs = 4
# По умолчанию используем DummyVecEnv (часто быстрее для "лёгких" env).
# SubprocVecEnv включает прицельно под тяжёлые env/на Linux.
cfg.vec.backend = "dummy"
cfg.vec.start_method = "spawn"
# Масштабировать скорость убывания epsilon на количество параллельных сред.
# Это восстанавливает паритет поведения между single-env и vec-env по числу env-шагов до той же ε.
cfg.vec.scale_epsilon_by_envs = True

# python train.py configs/alpha_512k.py
# python test_agent.py configs/alpha_512k.py
# python backtest_engine.py configs/alpha_512k.py
# python optimize_cfg.py configs/alpha_512k.py

# Mini run with 10 short sessions
# python optimize_cfg.py configs/alpha_512k.py --trials 100 --jobs 1

# Notes: Default metric is values_0; default direction is max.
# python get_info_from_optuna.py configs/alpha_512k.py --n-best-trials 10

# rm -r output/alpha_512k

# Main workflow:
# Step                              Command
# 1. Train the model:               python train.py configs/alpha_512k.py
# 2. Update the cache:              python backtest_engine.py configs/alpha_512k.py  | When running a backtest, set backtest_mode = True
# 3. Run optimization:              python optimize_cfg.py configs/alpha_512k.py
# 4. Show and save top-n trials:    python get_info_from_optuna.py configs/alpha_512k.py

cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paths.norm_stats_path = "C:\\Python\\Prosperous_Bot\\output\\alpha_512k\\norm_stats.json"
# 
cfg.paper.source = "database"
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
# test_data_path можно пока не трогать или приравнять к backtest
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz" 
# --- Явное указание пути к модели для бэктеста ---
# Этот параметр теперь является единственным способом указать модель для бэктеста.
# Путь должен указывать на конкретный .pth файл.
# ВАЖНО: После тренировки модели, обновите этот путь на актуальный.
cfg.paths.model_path = r"C:\Python\Prosperous_Bot\output\alpha_512k\saved_models\<ИМЯ_ПАПКИ_С_ДАТОЙ>\best.pth"

# --- NEW: Spike Detector Configuration ---
cfg.detector.context_minutes = 30
cfg.detector.window_minutes = 10
cfg.detector.use_lookahead = False # IMPORTANT: This should be True for backtesting/dataset creation
cfg.detector.abs_change_pct = 5.0
cfg.detector.contrast_min = 5.0
cfg.detector.cooldown_minutes = 30
