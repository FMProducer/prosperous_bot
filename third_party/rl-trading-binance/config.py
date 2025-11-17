# config.py
import os
from typing import Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo


class DeviceConfig(BaseModel):
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    class Config:
        arbitrary_types_allowed = True

    @field_validator("device", mode='before')
    @classmethod
    def validate_device(cls, v):
        if isinstance(v, str):
            return torch.device(v)
        return v



class PathConfig(BaseModel):
    config_name: str = "alpha"
    base_output_dir: str = "output"
    extra_model_dir: Optional[str] = None
    extra_cache_dir: Optional[str] = None

    train_data_path: str = "data/train_data.npz"
    val_data_path: str = "data/val_data.npz"
    test_data_path: str = "data/test_data.npz"
    backtest_data_path: str = "data/backtest_data.npz"
    norm_stats_path: Optional[str] = None
    model_path: Optional[str] = None # Явный путь к файлу модели (.pth)
    # Сделаем model_dir и plot_dir изменяемыми полями
    model_dir: Optional[str] = None
    plot_dir: Optional[str] = None

    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_output_dir, self.config_name)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")

    @field_validator("model_dir", "plot_dir", mode='before')
    @classmethod
    def set_default_dirs(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            values = info.data
            output_dir = os.path.join(values.get("base_output_dir", "output"), values.get("config_name", "alpha"))
            if info.field_name == "model_dir":
                return os.path.join(output_dir, "saved_models")
            if info.field_name == "plot_dir":
                return os.path.join(output_dir, "plots")
        return v

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.output_dir, "backtest_qval_cache")


class VecConfig(BaseModel):
    """
    Параметры векторизации окружений (Vectorized Environments).
    По умолчанию включаем 4 копии тренеровочной среды и синхронный backend.
    """
    num_envs: int = 4  # Количество параллельных сред (SubprocVecEnv)
    backend: Literal["dummy", "subproc"] = "dummy"  # "dummy" = 1 процесс, синхронно
    start_method: Literal["spawn", "fork", "forkserver"] = "spawn"  # безопасно на всех ОС
    # Флаг для масштабирования убывания эпсилон в зависимости от кол-ва сред.
    # Восстанавливает паритет шагов исследования между single-env и vec-env.
    scale_epsilon_by_envs: bool = False


class DataConfig(BaseModel):
    num_channels: int = 10
    expected_channels: List[str] = ["open", "high", "volume_weighted_average", "low", "close", "volume", "num_trades"]
    data_channels: List[str] = expected_channels.copy()
    price_channels: List[str] = ["open", "high", "volume_weighted_average", "low", "close"]
    volume_channels: List[str] = ["volume", "num_trades"]
    other_channels: List[str] = []
    plot_examples: int = 1
    plot_channel_idx: int = 4


from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

class SequenceConfig(BaseModel):
    full_seq_len: int = 150
    pre_signal_len: int = 90
    post_signal_len: int = 60
    agent_history_len: int = 30
    agent_session_len: int = 10
    action_history_len: int = 3
    state_shape: tuple = (10, 150, 1)

    @property
    def num_features(self) -> int:
        return len(DataConfig().data_channels)

    @property
    def input_history_len(self) -> int:
        return self.agent_history_len - 1

    @input_history_len.setter
    def input_history_len(self, value: int):
        self.agent_history_len = value

    @property
    def flat_state_size(self) -> int:
        return self.input_history_len * self.num_features + 4

    @field_validator("full_seq_len")
    def validate_full_seq_len(cls, v, values):
        if "pre_signal_len" in values and "post_signal_len" in values:
            assert v == values["pre_signal_len"] + values["post_signal_len"], "FULL_SEQ_LEN mismatch"
        return v


class MarketConfig(BaseModel):
    initial_balance: float = 10_000.0
    transaction_fee: float = 0.0004
    # 0.01% – 0.05% (1–5 bps -> basis points) 1 bps = 0.01% = 0.0001
    slippage: float = 0.0005 / 2
    num_actions: int = 4
    inaction_penalty_ratio: float = 0.001
    bankruptcy_threshold: float = 0.0
    bankruptcy_penalty: float = 1.0


class RLConfig(BaseModel):
    lr: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    batch_size: int = 16
    target_update_freq: int = 100
    train_start: int = 10_000
    max_gradient_norm: float = 1.0
    n_step: int = 5
    gamma_n_step_buffer: float = 0.99


class PERConfig(BaseModel):
    buffer_size: int = 230_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 20_000
    per_eps: float = 1e-6


class EpsilonConfig(BaseModel):
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frames: int = 50_000


class ModelConfig(BaseModel):
    cnn_maps: List[int] = [32, 64, 128]
    cnn_kernels: List[int] = [7, 5, 3]
    cnn_strides: List[int] = [2, 1, 1]
    cnn_dilations: List[int] = [1, 2, 4, 8]  # Receptive ~60 min
    dense_val: List[int] = [128, 64]
    dense_adv: List[int] = [128, 64]
    additional_feats: int = 16  # 4 + action_history_len * num_actions
    dropout_p: float = 0.1


class TrainLogConfig(BaseModel):
    episodes: int = 55_000
    episodes_per_epoch: int = 10000  # Для sampling/memory
    total_timesteps: int = 1000000
    validate_model: bool = True
    val_freq: int = 1000
    num_val_ep: int = 3500
    validation_warmup_steps: int = 0
    available_metrics: List[str] = [
        "Validation_mean_reward",
        "Validation_mean_pnl",
        "Validation_win_rate",
        "Validation_profit_factor",
        "Validation_max_drawdown",
        "Validation_sharpe",
        "Validation_sortino",
        "Validation_all_pnls",
    ]
    # Single- или Multi-объективный выбор (лексикографический порядок при списке)
    val_selection_metrics: Union[str, List[str]] = "Validation_mean_pnl"
    # Направление сравнения и минимальное улучшение
    val_selection_direction: Literal["max", "min"] = "max"
    val_min_delta: float = 0.0
    test_selection_metrics: str = "Test_all_pnls"
    plot_moving_avg_window: int = 10
    plot_top_n: int = 10
    plot_metric: str = "pnl"
    iterations: int = 10_000
    early_stopping_patience: int = 20

    @field_validator("val_selection_metrics")
    def check_val_metric(cls, v, values):
        allowed = set(values.get("available_metrics", []))
        if isinstance(v, str):
            assert v in allowed, "Selected metric not in AVAILABLE_METRICS"
        elif isinstance(v, (list, tuple)):
            assert all(isinstance(x, str) and x in allowed for x in v), \
                "All selection metrics must be in AVAILABLE_METRICS"
        else:
            raise TypeError("val_selection_metrics must be str or list[str]")
        return v


class DebugConfig(BaseModel):
    debug_max_size_data: Optional[int] = None
    use_final_model: bool = False


class SmartExplorationConfig(BaseModel):
    use_strategy: bool = False
    mode: Literal["random", "softmax"] = "softmax"
    temperature: float = 1.0
    top_k: Optional[int] = None
    initial_threshold: float = 0.6
    percentile_for_value: float = 70.0
    window_size_for_value: int = 500
    min_size_get_dynamic_thresh: int = 50


class BacktestConfig(BaseModel):
    continuous_data: bool = False # Flag to use the continuous data loader
    ticker_name: Optional[str] = None
    volatility_threshold: Optional[float] = None
    position_fraction: float = 0.5
    order_size_usdt: float = 0.0
    max_parallel_sessions: int = 2
    return_qvals: bool = True
    use_cache: bool = True
    clear_disk_cache: bool = True
    long_action_threshold: float = 0.012695
    short_action_threshold: float = 0.009902
    close_action_threshold: float = 0.001141
    use_risk_management: bool = False
    stop_loss: float = 0.01
    take_profit: float = 0.02
    trailing_stop: float = 0.005
    trailing_stop_min: Optional[float] = None
    fee_buffer_mult: Optional[float] = None
    delta_p_hysteresis: Optional[float] = None
    selection_strategy: Literal["advantage_based_filter", "ensemble_q_filter"] = "advantage_based_filter"
    plot_backtest_balance_curve: bool = True
    data_source: Literal["npz_keys", "find_spikes"] = "npz_keys"
    ensemble_n_samples: int = 5
    ensemble_max_sigma: float = 0.01
    time_range: Optional[Dict[str, str]] = None
    exec_delay_bars: int = 1


class PaperTraderConfig(BaseModel):
    source: Literal["websocket", "database"] = "websocket"
    db_source_speed: float = 0.0  # Seconds to sleep between simulated minutes. 0.0 for max speed.
    symbols: Optional[Union[List[str], Literal["ALL"]]] = None # None or empty list means use all from tickers.txt
    leverage: float = 1.0 # Новое: плечо для бумажной торговли

class DetectorConfig(BaseModel):
    """Spike detector parameters for finding trading signals."""
    context_minutes: int = 30
    window_minutes: int = 10
    use_lookahead: bool = False  # Must be False for live trading/paper trading
    abs_change_pct: float = 5.0
    contrast_min: float = 5.0
    cooldown_minutes: int = 30


class LoggingConfig(BaseModel):
    per_trial_logs: bool = False


class DbConfig(BaseModel):
    dsn: str = "postgresql://postgres:9691@localhost:5432/marketdata?sslmode=disable"


class PerformanceConfig(BaseModel):
    """
    Переключатели производительности, управляемые из configs/*.py.
    Никакой бизнес-логики — только флаги/параметры, которые затем
    используются в train.py/agent.py (AMP, torch.compile, DataLoader).
    """
    # AMP (автокаст и GradScaler)
    use_amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    # torch.compile (PyTorch 2.x)
    compile_mode: Optional[Literal["default", "reduce-overhead", "max-autotune"]] = None
    compile_dynamic: bool = True
    # CUDA/CuDNN
    cudnn_benchmark: bool = True
    # Параметры загрузки данных
    dataloader_num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None


class MasterConfig(BaseModel):
    project_name: str = "rl_binance_futures_trading"
    render_mode: Optional[str] = None
    random_seed: int = 25
    global_env_seed: int = 17
    backtest_mode: bool = True

    device: DeviceConfig = DeviceConfig()
    paths: PathConfig = PathConfig()
    vec: VecConfig = VecConfig()
    data: DataConfig = DataConfig()
    seq: SequenceConfig = SequenceConfig()
    market: MarketConfig = MarketConfig()
    rl: RLConfig = RLConfig()
    per: PERConfig = PERConfig()
    eps: EpsilonConfig = EpsilonConfig()
    model: ModelConfig = ModelConfig()
    trainlog: TrainLogConfig = TrainLogConfig()
    debug: DebugConfig = DebugConfig()
    smart: SmartExplorationConfig = SmartExplorationConfig()
    backtest: BacktestConfig = BacktestConfig()
    paper: PaperTraderConfig = PaperTraderConfig()
    logging: LoggingConfig = LoggingConfig()
    detector: DetectorConfig = DetectorConfig()
    perf: PerformanceConfig = PerformanceConfig()
    db: DbConfig = DbConfig()

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


cfg = MasterConfig()


assert cfg.seq.action_history_len <= cfg.seq.agent_session_len, "ACTION_HISTORY_LEN > AGENT_SESSION_LEN"
