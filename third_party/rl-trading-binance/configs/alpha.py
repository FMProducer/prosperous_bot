# configs/alpha.py
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
# gradient steps ~ 241_000 ~ episodes = 24_000
cfg.trainlog.episodes = 55_000
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
cfg.backtest.long_action_threshold = 0.012695  # Снижено с 0.012695
cfg.backtest.short_action_threshold = 0.009902 # Снижено с 0.009902
cfg.backtest.close_action_threshold = 0.001141
cfg.backtest.ensemble_n_samples = 5
# maximum allowed variance (uncertainty) (range: 0.001 to 0.015)
cfg.backtest.ensemble_max_sigma = 0.01
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
# use_risk_management
cfg.backtest.use_risk_management = False
cfg.backtest.stop_loss = 0.01
cfg.backtest.take_profit = 0.02
cfg.backtest.trailing_stop = 0.005
cfg.backtest.plot_backtest_balance_curve = True

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
# torch.compile: снижает overhead Python-графа; режим "reduce-overhead" — наиболее безопасный.
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

# python train.py configs/alpha.py
# python test_agent.py configs/alpha.py
# python backtest_engine.py configs/alpha.py
# python optimize_cfg.py configs/alpha.py

# Mini run with 10 short sessions
# python optimize_cfg.py configs/alpha.py --trials 100 --jobs 1

# Notes: Default metric is values_0; default direction is max.
# python get_info_from_optuna.py configs/alpha.py --n-best-trials 10

# rm -r output/alpha

# Main workflow:
# Step                              Command
# 1. Train the model:               python train.py configs/...
# 2. Update the cache:              python backtest_engine.py configs/...  | When running a backtest, set backtest_mode = True
# 3. Run optimization:              python optimize_cfg.py configs/...
# 4. Show and save top-n trials:    python get_info_from_optuna.py configs/...

data = {
    "source": "stream_sim_db",
    "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"},
    # Базовый (демо) режим: 30-10 — полная совместимость с README (Demo) :contentReference[oaicite:10]{index=10}
    "ctx_minutes": 30,
    # Порог и кулдаун используются и офлайн, и при потоковом построении индекса
    "trigger": {"abs_change_pct": 5.0, "cooldown_minutes": 30},
    "resample_1t": True,
    # Включить построение индекса окон из БД (если нет заранее подготовленного CSV)
    "build_index_from_db": True,
    # Новый режим индекса: скользящее окно на КАЖДОМ минутном баре
    "index_mode": "sliding",
    "sliding_stride_minutes": 1,
    "symbols": ["OMUSDT","1000RATSUSDT"],
    "session_minutes": 10,
     # Детектор всплесков: для строгой репликации backtest оставляем look-ahead включённым
    "detector": {
        # Полный режим: 90-10 детекция (контекст 90, окно оценки 10), но сам инференс/сессия задаётся agent_session_len (выше)
        "context_minutes": 30,
        "window_minutes": 10,
        "use_lookahead": True,       # True — как в бэктесте; False — реал-режим без заглядывания вперёд
        "abs_change_pct": 5.0,       # |ΔP| over window, %
        "contrast_min": 5.0,         # (|ΔP| / avg_abs_ret_pre) ≥ contrast_min
        "cooldown_minutes": 30
    },
    # Если файл провайдера лежит рядом (db_provider.py), используем прямой импорт:
    "db_provider": "db_provider:get_feed",
    # ---- Inference (строгий режим без фоллбэка) ----
    "inference": {
        "policy_loader": "inference_adapter:load_policy",  # module:function
        "checkpoint_path": r"C:\Python\Prosperous_Bot\third_party\FMProducer\fmproducer_1_eval\saved_models\session_1\best.pth",
        "strict": True   # True: без рабочей политики торги пропускаются (никакого Follow-Context)
    },
    # ---- Paper trading (RT/ASAP) ----
    "paper_trader": {
        "mode": "asap",  # "realtime" | "asap"
        "cap_windows_per_symbol": 0  # 0 = без лимита; иначе макс. окон/день/тикер
    },
    "exec": {
        "base_capital_usdt": 10000.0,  # общий капитал (для риска/позиции)
        "risk_per_trade_pct": 1.0,     # риск на сделку, %
        "fee_bps": 2.0,                # комиссия в б.п. (0.01% = 1 б.п.)
        "slippage_bps": 1.0,           # проскальзывание (одна сторона) в б.п.
        "max_concurrent": 4            # ограничение на одновременные позиции
    }
}

cfg.db.dsn = "postgresql://postgres:9691@localhost:5432/marketdata"
cfg.paths.norm_stats_path = "C:\\Python\\Prosperous_Bot\\third_party\\rl-trading-binance\\output\\alpha\\norm_stats.json"
