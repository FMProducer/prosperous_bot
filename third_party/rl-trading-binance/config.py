# config.py
import os
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, Field, validator


class DeviceConfig(BaseModel):
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    class Config:
        arbitrary_types_allowed = True


class PathConfig(BaseModel):
    config_name: str = "alpha"
    base_output_dir: str = "output"
    extra_model_dir: Optional[str] = None
    extra_cache_dir: Optional[str] = None

    train_data_path: str = "data/train_data.npz"
    val_data_path: str = "data/val_data.npz"
    test_data_path: str = "data/test_data.npz"
    backtest_data_path: str = "data/backtest_data.npz"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_output_dir, self.config_name)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.output_dir, "saved_models")

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.output_dir, "plots")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.output_dir, "backtest_qval_cache")


class DataConfig(BaseModel):
    expected_channels: List[str] = ["open", "high", "volume_weighted_average", "low", "close", "volume", "num_trades"]
    data_channels: List[str] = expected_channels.copy()
    price_channels: List[str] = ["open", "high", "volume_weighted_average", "low", "close"]
    volume_channels: List[str] = ["volume", "num_trades"]
    other_channels: List[str] = []
    plot_examples: int = 1
    plot_channel_idx: int = 4


class SequenceConfig(BaseModel):
    full_seq_len: int = 150
    pre_signal_len: int = 90
    post_signal_len: int = 60
    agent_history_len: int = 30
    agent_session_len: int = 10
    action_history_len: int = 3

    @property
    def num_features(self) -> int:
        return len(DataConfig().data_channels)

    @property
    def input_history_len(self) -> int:
        return self.agent_history_len - 1

    @property
    def flat_state_size(self) -> int:
        return self.input_history_len * self.num_features + 4

    @validator("full_seq_len")
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


class RLConfig(BaseModel):
    gamma: float = 0.99
    learning_rate: float = 1e-4
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
    dense_val: List[int] = [128, 64]
    dense_adv: List[int] = [128, 64]
    additional_feats: int = 16  # 4 + action_history_len * num_actions
    dropout_p: float = 0.1


class TrainLogConfig(BaseModel):
    episodes: int = 55_000
    validate_model: bool = True
    val_freq: int = 1000
    num_val_ep: int = 3500
    available_metrics: List[str] = [
        "Validation_mean_reward",
        "Validation_mean_pnl",
        "Validation_win_rate",
        "Validation_all_pnls",
    ]
    val_selection_metrics: str = "Validation_mean_pnl"
    test_selection_metrics: str = "Test_all_pnls"
    plot_moving_avg_window: int = 10
    plot_top_n: int = 10
    plot_metric: str = "pnl"
    iterations: int = 10_000

    @validator("val_selection_metrics")
    def check_val_metric(cls, v, values):
        if "available_metrics" in values:
            assert v in values["available_metrics"], "Selected metric not in AVAILABLE_METRICS"
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
    selection_strategy: Literal["advantage_based_filter", "ensemble_q_filter"] = "advantage_based_filter"
    plot_backtest_balance_curve: bool = True
    ensemble_n_samples: int = 5
    ensemble_max_sigma: float = 0.01


class LoggingConfig(BaseModel):
    per_trial_logs: bool = False


class MasterConfig(BaseModel):
    project_name: str = "rl_binance_futures_trading"
    render_mode: Optional[str] = None
    num_envs: int = 2
    random_seed: int = 25
    global_env_seed: int = 17
    backtest_mode: bool = True

    device: DeviceConfig = DeviceConfig()
    paths: PathConfig = PathConfig()
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
    logging: LoggingConfig = LoggingConfig()


cfg = MasterConfig()


assert cfg.seq.action_history_len <= cfg.seq.agent_session_len, "ACTION_HISTORY_LEN > AGENT_SESSION_LEN"
assert 1 / cfg.backtest.max_parallel_sessions >= cfg.backtest.position_fraction
