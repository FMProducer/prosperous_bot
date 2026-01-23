# configs/alpha.py — Все параметры RL-тренировки в одном месте
import torch
from config import cfg  # noqa: F401
from pathlib import Path
import json  # Для fallback norm_stats если нужно
import types

# --- DYNAMIC PATHS SETUP ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# --- AGENT MODE SELECTOR ---
# UNIVERSAL:  Trade both directions (Default)
# LONG_ONLY:  Force Long trades only (Train specialist)
# SHORT_ONLY: Force Short trades only (Train specialist)
# AGENT_MODE = "UNIVERSAL" 
# AGENT_MODE = "LONG_ONLY"
AGENT_MODE = "SHORT_ONLY"

if AGENT_MODE == "UNIVERSAL":
    cfg.paths.config_name = "alpha_seed_406_ohlcv"
else:
    cfg.paths.config_name = f"alpha_seed_406_ohlcv_{AGENT_MODE}"


print(f"CONFIG LOADED: AGENT_MODE = {AGENT_MODE}")

cfg.paths.model_dir = f"output/{cfg.paths.config_name}/saved_models"
cfg.paths.plot_dir = f"output/{cfg.paths.config_name}/plots"

# --- ИЗМЕНЕНИЯ ДЛЯ OHLCV (5 каналов) ---

# 1. Указываем каналы данных
# Важно: 'date' обычно исключают из input-features нейросети.
# Если вы оставите 'date', модель может переобучиться на конкретные временные метки.
# Рекомендуемый список (5 каналов):
cfg.data.datachannels = ["open", "high", "low", "close", "volume"]

# Для справки: какие из них считать ценой, а какие объемом (для нормализации)
cfg.data.pricechannels = ["open", "high", "low", "close"]
cfg.data.volumechannels = ["volume"]
cfg.data.otherchannels = [] # "date" и другие меты здесь не нужны для обучения

# 2. Настраиваем размеры
cfg.num_channels = len(cfg.data.datachannels)  # Будет 5
cfg.state_shape = (cfg.num_channels, 90, 1)    # (5, 90, 1)
cfg.seq.state_shape = cfg.state_shape

# 3. Настройка входного слоя CNN
# Поскольку каналов стало меньше (5 вместо 10), первый слой CNN должен это учитывать.
# Обычно это происходит автоматически, но если maps заданы жестко, проверим:
# cfg.model.cnn_maps = [64, ...] - первый параметр это выходные каналы, входные берутся из state_shape.
# Менять параметры модели (cnn_maps) НЕ нужно, они адаптируются под input shape.

# 4. ВНИМАНИЕ: Данные (NPZ файлы)
# Ваши файлы data/train_data_fair_8m.npz скорее всего содержат 10 каналов.
# Чтобы скрипт train.py "понял", что нужно брать только 5, он должен поддерживать фильтрацию.
# Если он просто читает все подряд, вам придется пересоздать датасет (npz) только с 5 колонками.

cfg.seq.full_seq_len = 150      # 90 контекст + 60 сессия
cfg.seq.agent_history_len = 90  # Context window (история)
cfg.seq.agent_session_len = 60  # Trading session length (60 шагов)
cfg.seq.action_history_len = 0  # Recent actions feat (Disabled to prevent IndexError)
cfg.seq.pre_signal_len = 90     # Старт эпизода после 90 баров истории
cfg.seq.post_signal_len = 60

# Явно фиксируем длину входного окна истории для env/model
cfg.seq.input_history_len = 90
cfg.episodes_per_epoch = 10000  # Sampling для memory (full 24k fallback) # This line was not in the diff but seems to belong with this block.
cfg.paths.train_data_path = "data/train_data_fair_8m.npz"
cfg.paths.val_data_path = "data/backtest_data_fair_2m.npz"  # Или data/val_data_fair_2m.npz
cfg.paths.test_data_path = "data/backtest_data_fair_2m.npz"  # Или data/backtest_data_fair_2m.npz
cfg.paths.norm_stats_path = str(BASE_DIR / "norm_stats.json")
cfg.paths.model_path = ""

# Model: ActorCritic CNN (dilated 1D Conv для ~60-min receptive)
cfg.model.cnn_maps = [64, 96, 128, 128, 96, 64]  # +1 layer
cfg.model.cnn_kernels = [3, 3, 3, 3, 3, 3]  # +1 layer
cfg.model.cnn_dilations = [1, 2, 4, 8, 16, 28]  # RF=87 bars (96.7% coverage)
cfg.model.cnn_strides = [1, 1, 1, 1, 1, 1]  # +1 layer
cfg.model.dense_val = [128, 64, 32]  # Value head
cfg.model.dense_adv = [128, 64, 32]  # Advantage/policy head
cfg.model.additional_feats = 4  # Pos(1) + unrealized(1) + time(2) + action_history(0) = 4
cfg.model.dropout_p = 0.10

# Market Config - ДОБАВЬТЕ ЭТУ СТРОКУ
cfg.market.num_actions = 3  # Discrete: 0=hold, 1=buy, 2=sell. Close отключен.

# Market/Position Sizing (для обучения, НЕ только backtest!)
cfg.market.position_fraction = 0.10  # 10% баланса на сделку
cfg.market.transaction_fee = 0.0004  # Уже есть ниже, но явно здесь
cfg.market.slippage = 0.0002
cfg.market.allow_opposite_trades = False # Запрещаем закрытие противоположной сделкой
# cfg.market.max_trades_per_episode = 1    # Caused ValueError in Pydantic
MAX_TRADES_PER_EPISODE = 1 # 1 сделка на сессию (60 баров). Запрет перезахода после TSL.

# --- MODE CONFIGURATION ---
if AGENT_MODE == "LONG_ONLY":
    cfg.market.allowed_directions = ['LONG']
    # IMPORTANT: Do not filter data for LONG, as the environment does not invert it.
    cfg.market.filter_direction = None
    
elif AGENT_MODE == "SHORT_ONLY":
    cfg.market.allowed_directions = ['SHORT']
    # CRITICAL: Enable filter_direction to trigger "Mirror World" logic in the environment.
    cfg.market.filter_direction = 'SHORT'
    
else: # UNIVERSAL
    cfg.market.allowed_directions = ['LONG', 'SHORT']
    cfg.market.filter_direction = None # All data is used, no inversion.

# num_actions остается 3, чтобы сохранить совместимость весов модели!
# 0=Wait, 1=Buy, 2=Sell. В режиме SHORT_ONLY агент просто не будет нажимать 1.

# RL/DQN Params (custom agent)
cfg.rl.lr = 0.0003  # Снижаем скорость обучения для большей стабильности
cfg.rl.gamma = 0.99         # Выше для длинного горизонта
cfg.rl.n_step = 5   # Чуть больше для лучшего связывания наград
cfg.rl.batch_size = 64  # Увеличиваем батч для более стабильного градиента
cfg.rl.train_start = 15000  # Значительно увеличиваем warmup, чтобы собрать разнообразный опыт перед обучением
cfg.rl.target_update_freq = 5000   # Чаще для длинных эпизодов
cfg.rl.max_gradient_norm = 1.0  # Clip grads

# DQN-specific (PER/epsilon)
cfg.per.buffer_size = 1000000
cfg.per.per_alpha = 0.6
cfg.per.per_beta_start = 0.4
cfg.per.per_beta_frames = 250000  # Синхронизируем с новым total_timesteps
cfg.per.per_eps = 1e-6
cfg.eps.eps_start = 1.0
cfg.eps.eps_end = 0.05
cfg.eps.eps_decay_frames = 180000  # Заканчиваем исследование раньше (под новый бюджет)

# Env/Vectorized
cfg.vec.num_envs = 12             # параллельные среды
cfg.vec.backend = "subproc"        # сначала DummyVecEnv, потом можно subproc
cfg.vec.start_method = "spawn"
cfg.vec.scale_epsilon_by_envs = True  # Adjust eps decay

# Training Log/Validation
cfg.trainlog.num_val_ep = 512  # Уменьшаем для быстрой валидации, 100 достаточно
max_episodes_per_symbol = 2

# При 4 env один эпизод даёт ~4× больше шагов.
# Чтобы общий бюджет шагов остался ≈600k, эпизодов можно делать ~в 4 раза меньше.
cfg.trainlog.episodes = 5000       # Меньше (60-bar episodes дольше)
cfg.trainlog.total_timesteps = 250000  # Сокращаем общий бюджет шагов

# Валидация: масштабируем по эпизодам, чтобы частота и прогрев соответствовали новому числу эпизодов.
cfg.trainlog.val_freq = 200             # Валидируемся чуть реже
cfg.trainlog.validation_warmup_steps = 15000       # норма 450000 (значительно уменьшено)
cfg.trainlog.plot_top_n = 10
cfg.trainlog.available_metrics = [
    "Validation_mean_reward", "Validation_mean_pnl", "Validation_win_rate",
    "Validation_profit_factor", "Validation_max_drawdown", "Validation_net_pnl",
    "Validation_sharpe", "Validation_sortino"
]
cfg.trainlog.val_selection_metrics = ["Validation_net_pnl", "Validation_sortino"]
cfg.trainlog.early_stopping_patience = 25  # Увеличиваем терпение (10 * 200 = 2000 эпизодов, ~40% обучения)

# Validation Gate (multi-crit; deny bad models)
cfg.validation_gate = {
    "min_sharpe": 0.01,
    "min_sortino": 0.01,
    "min_profit_factor": 1.01,      # Чуть выше безубытка
    "max_drawdown_at_most": -0.10,  # Ограничиваем просадку 30% (было -1.17)
    "min_win_rate": 0.45,  # Чуть ниже (SHORT сложнее)
    "min_trades": 500,      # Снижаем порог сделок для короткой валидации
    "deny_inf_pf": True,
    "deny_zero_drawdown": True
}

# Top-K checkpoint saving
cfg.trainlog.save_top_k = 10  # Сохранять топ-10 моделей
cfg.trainlog.checkpoint_metric = "Validation_sortino"  # Основная метрика: суммарная прибыль
cfg.trainlog.save_mode = "max"  # Максимизировать метрику

# --- Shaped Rewards & Penalties ---

# --- Bonuses ---
cfg.market.new_equity_peak_reward = 0.0   # Награда за достижение нового максимума эквити
cfg.market.perfect_entry_reward = 0.0   # Награда за прибыльную сделку, которая не уходила в минус
cfg.market.risk_reward_ratio_reward = 0.0   # Награда за сделку с высоким соотношением риск/прибыль
cfg.market.good_exit_bonus = 0.0   # Награда за хороший выход (закрытие сделки с >=80% от пиковой прибыли)
cfg.market.fast_exit_bonus = 0.0   # Награда за быстрый выход (< 20 шагов)
# --- Penalties ---
cfg.market.bankruptcy_penalty = 10.0   # Штраф за банкротство
cfg.market.bankruptcy_threshold = 0.0
cfg.market.bankruptcy_slippage_penalty = 0.05   # Штрафное проскальзывание при принудительной ликвидации
cfg.market.max_drawdown_threshold = -0.10   # Штраф за превышение максимальной просадки (MaxDD)
cfg.market.max_drawdown_penalty_type = "proportional"
cfg.market.max_drawdown_penalty = 1.0
cfg.market.continuous_pain_penalty_ratio = 0.0   # Штраф за удержание убыточной позиции (каждый шаг)
cfg.market.inaction_penalty_ratio = 0.001   # Штраф за бездействие (когда нет открытых позиций)
cfg.market.time_sl_penalty_ratio = 0.01   # Штраф за Time SL
cfg.market.low_balance_penalty = 1.0   # Штраф за попытку торговли с низким балансом
cfg.market.holding_penalty_multiplier = 0.0   # Множитель для прогрессивного штрафа за удержание убыточной позиции
cfg.market.greed_penalty_multiplier = 0.0   # "Штраф за жадность" (незафиксированная прибыль)
cfg.market.premature_profit_exit_penalty = 0.0   # Штраф за ранний выход из ПРИБЫЛЬНОЙ позиции (< profit_exit_threshold шагов)
cfg.market.holding_loss_penalty = 0.0   # Штраф за долгое удержание УБЫТОЧНОЙ позиции (> loss_exit_threshold шагов)
cfg.market.premature_exit_penalty = 0.0  # Легаси параметр
cfg.market.profit_holding_bonus = 0.0    # Легаси параметр, заменен на асимметричную логику
# --- Thresholds for Shaped Rewards ---
cfg.market.risk_reward_ratio_threshold = 0.0   # Порог для соотношения риск/прибыль (3:1)
cfg.market.profit_exit_threshold = 0   # Порог времени удержания для прибыльных позиций (минимум для выхода без штрафа) bars
cfg.market.greed_penalty_threshold = 0.4   # Порог отката прибыли для штрафа за жадность, разрешаем откат на 60% (удерживаем 40%)
cfg.market.exit_quality_threshold = 0.8   # Порог качества выхода для бонуса (0.80 = 80% от максимальной прибыли)
cfg.market.loss_exit_threshold = 70   # Порог времени удержания для убыточных позиций (максимум для выхода без штрафа) bars
cfg.market.holding_penalty_threshold = 70   # Порог времени удержания для начала прогрессивного штрафа (шагов)
cfg.market.fast_exit_threshold = 70  # Порог для бонуса за быстрый выход (шагов)
cfg.market.premature_exit_threshold = 70   # Порог для штрафа за преждевременный выход (шагов)

# Backtest/Paper Trader
cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 4
cfg.backtest.position_fraction = 0.10
cfg.backtest.order_size_usdt = 0.0
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.0  # 0.015
cfg.backtest.short_action_threshold = 0.0  # -0.015 Negative for short
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.use_risk_management = True # Включаем для работы TSL
cfg.backtest.trailing_stop = 0.07519862504113693
cfg.backtest.exec_delay_bars = 1
cfg.backtest.plot_backtest_balance_curve = True
cfg.backtest.trailing_stop_min = 0.0008225518697224519
cfg.backtest.fee_buffer_mult = 2.5
cfg.backtest.delta_p_hysteresis = 0.0015218098435784326
cfg.backtest.time_range = {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-09-30T23:59:00Z"}

# Perf/Perf (GTX1070 opt)
cfg.perf.use_amp = False  # ОТКЛЮЧЕНО: float16 слишком рискован для RL из-за возможного обнуления градиентов.
cfg.perf.amp_dtype = "float16" # Для стабильности лучше float32 (use_amp=False) или bfloat16 на новых GPU.

cfg.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.perf.compile_mode = None  # None - no torch.compile
cfg.perf.dataloader_num_workers = 0  # Windows safe
cfg.perf.pin_memory = True
cfg.perf.persistent_workers = False
cfg.perf.prefetch_factor = 2
cfg.perf.cudnn_benchmark = True

# MC-Dropout (ensemble; off by default)
mc_dropout_cfg = types.SimpleNamespace()
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
cfg.paper.leverage = 1.0
cfg.paper.symbols = "ALL"  # Or "ALL" or list from tickers.txt
cfg.backtest.data_source = "npz"  # For test/backtest

# Random/Logging
cfg.random_seed = 406
cfg.logging.per_trial_logs = True
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False
cfg.deterministic = False

# Bundle (for saving artifacts)
bundle_cfg = types.SimpleNamespace()
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

# --- DYNAMIC PATHS SETUP ---
# Определяем базовую директорию проекта относительно этого конфиг-файла
# Ожидаемая структура: <root>/third_party/rl-trading-binance/configs/
try:
    # Path(__file__).parent -> configs
    # .parent -> rl-trading-binance
    # .parent -> third_party
    # .parent -> <root>
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    # Fallback для интерактивных сред, где __file__ не определен
    BASE_DIR = Path.cwd()

# --- ПУТИ К МОДЕЛЯМ И АРТЕФАКТАМ ---
# Эти пути строятся динамически для обеспечения переносимости.
# Замените имена папок с временными метками на актуальные.
long_model_dir = BASE_DIR / "output" / "alpha_seed_406_ohlcv_LONG_ONLY" / "saved_models" / "rl_binance_futures_trading_date_20260110_time_133651"
short_model_dir = BASE_DIR / "output" / "alpha_seed_406_ohlcv_SHORT_ONLY" / "saved_models" / "rl_binance_futures_trading_date_20260110_time_191747"
single_model_dir = BASE_DIR / "output" / "alpha_seed_406_ohlcv_UNIVERSAL" / "saved_models" / "rl_binance_futures_trading_date_20251120_time_015257"

# Для валидации одиночного агента (раскомментируйте, если нужно)
# cfg.paths.model_path = single_model_dir / "best.pth"
# cfg.paths.norm_stats_path = single_model_dir / "norm_stats.json"

# Workflow notes (run from root):
# python train.py --config alpha.py --total_timesteps 10000  # Test
# python train.py --config alpha.py  # Full
# python paper_trader_q.py --model rl_model.pth --config alpha.py  # Backtest

# --- ENSEMBLE CONFIGURATION ---
cfg.ensemble = types.SimpleNamespace()

# Пути для валидации ансамбля
cfg.ensemble.long_model_path = long_model_dir / "best.pth"
cfg.ensemble.short_model_path = short_model_dir / "best.pth"
# Статистика нормализации обычно одинакова для long/short специалистов
cfg.ensemble.norm_stats_path = long_model_dir / "norm_stats.json"

# --- Ensemble Behavior ---
cfg.ensemble.enable_long = True
cfg.ensemble.enable_short = True
cfg.ensemble.use_confidence = True  # False = простое голосование (Argmax), True = порог уверенности Q
cfg.ensemble.long_threshold = 0.00365  # Порог уверенности для LONG 0.00365
cfg.ensemble.short_threshold = 0.00365 # Порог уверенности для SHORT 0.00365
cfg.ensemble.weights = [1.0, 1.0] # Веса [Long, Short] (пока 50/50)

# Маски признаков для согласования с additional_feats=4
cfg.ensemble.long_features_mask = [0, 1, 2, 3]
cfg.ensemble.short_features_mask = [0, 1, 2, 3]

# --- НОВЫЙ ПАРАМЕТР ---
# Если True, отключает логику, при которой открытие позиции одним агентом
# принудительно закрывает позицию другого. Сделки закрываются только по окончании сессии.
cfg.ensemble.disable_cross_close = True

# --- Soft Conflict Resolution ---
# Коэффициент уверенности для разрешения конфликтов.
# Пример: 1.2 означает, что Q-value "победителя" должно быть на 20% выше.
cfg.ensemble.confidence_ratio = 1.2