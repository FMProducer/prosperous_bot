import sys
import os
import logging
import logging.handlers
import importlib.util
from pathlib import Path
import json
import numpy as np  # type: ignore
import pandas as pd
from pandas import DataFrame
import torch  # type: ignore
from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
import onnxruntime as ort
import threading
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Tuple
from collections import deque
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


try:
    from freqtrade.persistence import Trade  # type: ignore
except ImportError:
    class Trade:
        id: int
        pair: str
        is_open: bool
        is_short: bool
        open_date_utc: Optional[datetime] = None
        close_rate_requested: Optional[float] = None
        open_rate: float = 0.0
        close_profit: Optional[float] = None
        
        def calc_profit(self, rate: float) -> float: return 0.0

        @classmethod
        def get_trades(cls, filters): return []

        @classmethod
        def get_open_trades(cls): return []

from datetime import datetime, timezone, timedelta

# --- 1. ПУТИ ---
strategy_file = Path(__file__).resolve()
project_root = strategy_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter, merge_informative_pair  # type: ignore
except ImportError:
    class IStrategy:
        dp: Any = None
        config: Dict[str, Any]
        logger: Any = None
        def __init__(self, config: dict):
            self.config = config
            self.logger = logging.getLogger(__name__)
    class DecimalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0.0)
    class IntParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0)
    class CategoricalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', args[0][0] if args and args[0] else None)
    def merge_informative_pair(dataframe, informative, timeframe, informative_timeframe, ffill=True):
        return dataframe

# Agent imports
try:
    from agent import D3QN_PER_Agent  # type: ignore
except ImportError as e:
    raise e

class CustomD3QNStrategy4z(IStrategy):
    config: Dict[str, Any]
    dp: Any
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short: bool = True 
    startup_candle_count: int = 180

    informative_timeframe = '1m'
    informative_timeframe_global = '1m'
    
    minimal_roi = {"0": 0.5, "30": 0.01, "45": 0}
    stoploss = -0.99 
    trailing_stop = False
    use_custom_stoploss = True
    
    order_types = {
        'entry': 'market', 'exit': 'market', 'stoploss': 'market', 'stoploss_on_exchange': False
    }
    
    d0 = DecimalParameter(0.01, 1.0, default=0.99, space='sell', optimize=False, load=False)
    d_min = DecimalParameter(0.0005, 0.005, default=0.001, space='sell', optimize=False, load=False)
    hysteresis = DecimalParameter(0.001, 0.01, default=0.005, space='sell', optimize=False, load=False)
    p_target = DecimalParameter(0.007, 0.1, default=0.04, space='sell', optimize=False, load=False)
    tsl_exponent = DecimalParameter(0.95, 1.1, default=1.003, space='sell', optimize=False, load=False)

    rl_long_threshold_opt = IntParameter(2, 2, default=2, space='buy', optimize=False, load=False)
    rl_short_threshold_opt = IntParameter(2, 2, default=2, space='sell', optimize=False, load=False)

    global_ema_timeframe = CategoricalParameter(['1m', '3m', '5m', '15m', '30m', '1h'], default='1m', space='buy', optimize=False, load=False)
    ema_fast_period = IntParameter(10, 15, default=12, space='buy', optimize=False, load=False)
    global_ema_period = IntParameter(5, 60, default=25, space='buy', optimize=False, load=False)
    min_quote_volume_usd = DecimalParameter(0, 500000, default=1000, space='buy', optimize=False, load=False)
    dd_aggression_k = DecimalParameter(0.0, 2.0, default=2.0, space='buy', optimize=False, load=False)
    rl_epsilon_long = DecimalParameter(0.0, 0.01, default=0.001, space='buy', optimize=False, load=False)
    rl_epsilon_short = DecimalParameter(0.95, 1.0, default=0.98, space='sell', optimize=False, load=False)

    vol_window = IntParameter(28, 34, default=31, space='buy', optimize=False, load=False)
    cvd_window = IntParameter(75, 90, default=83, space='buy', optimize=False, load=False)

    vol_f1_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    vol_f2_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=False)
    vol_f3_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)

    vol_f1_surge = DecimalParameter(1.05, 3.0, default=2.627, space='buy', optimize=False, load=False)
    vol_f1_pct = DecimalParameter(51.0, 80.0, default=79.172, space='buy', optimize=False, load=False)
    vol_f2_cvd_spike = DecimalParameter(1.0, 2.5, default=1.024, space='buy', optimize=False, load=False)
    vol_f2_gap = DecimalParameter(1.1, 2.5, default=1.959, space='buy', optimize=False, load=False)
    vol_f3_peak = DecimalParameter(4.5, 6.0, default=5.857, space='sell', optimize=False, load=False)
    vol_f3_fade = DecimalParameter(0.2, 0.3, default=0.276, space='sell', optimize=False, load=False)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.project_root = project_root 
        
        self._batch_cache = {}
        self._last_batch_ts = 0.0
        self._batch_lock = threading.Lock()
        self._is_batch_processing = False
        
        if load_dotenv:
            load_dotenv(dotenv_path=project_root / '.env')

        self.enable_long_1 = config.get('rl_enable_long_1', True)
        self.enable_long_2 = config.get('rl_enable_long_2', True)
        self.enable_short_1 = config.get('rl_enable_short_1', True)
        self.enable_short_2 = config.get('rl_enable_short_2', True)
        
        self._long_1_agent = None
        self._long_2_agent = None
        self._short_1_agent = None
        self._short_2_agent = None
        
        self.tsl_memory = {}
        self.q_normalization = config.get('rl_ensemble', {}).get('q_normalization', {})

    @property
    def long_1_agent(self):
        if not self.enable_long_1: return None
        if self._long_1_agent is None:
            path = project_root / self.config.get('rl_ensemble', {}).get('model_paths', {}).get('long_1', '')
            self.cfg_long_1 = self._load_py_config(next(path.glob("*.py")))
            self._long_1_agent = self._create_agent_from_config(self.cfg_long_1)
            self._load_weights(self._long_1_agent, path / "best.onnx", "L1")
        return self._long_1_agent

    @property
    def long_2_agent(self):
        if not self.enable_long_2: return None
        if self._long_2_agent is None:
            path = project_root / self.config.get('rl_ensemble', {}).get('model_paths', {}).get('long_2', '')
            self.cfg_long_2 = self._load_py_config(next(path.glob("*.py")))
            self._long_2_agent = self._create_agent_from_config(self.cfg_long_2)
            self._load_weights(self._long_2_agent, path / "best.onnx", "L2")
        return self._long_2_agent

    @property
    def short_1_agent(self):
        if not self.enable_short_1: return None
        if self._short_1_agent is None:
            path = project_root / self.config.get('rl_ensemble', {}).get('model_paths', {}).get('short_1', '')
            self.cfg_short_1 = self._load_py_config(next(path.glob("*.py")))
            mirror = getattr(self.cfg_short_1.market, 'mirror_mode', False)
            self._short_1_agent = self._create_agent_from_config(self.cfg_short_1, mirror_mode=mirror)
            self._load_weights(self._short_1_agent, path / "best.onnx", "S1")
        return self._short_1_agent

    @property
    def short_2_agent(self):
        if not self.enable_short_2: return None
        if self._short_2_agent is None:
            path = project_root / self.config.get('rl_ensemble', {}).get('model_paths', {}).get('short_2', '')
            self.cfg_short_2 = self._load_py_config(next(path.glob("*.py")))
            mirror = getattr(self.cfg_short_2.market, 'mirror_mode', False)
            self._short_2_agent = self._create_agent_from_config(self.cfg_short_2, mirror_mode=mirror)
            self._load_weights(self._short_2_agent, path / "best.onnx", "S2")
        return self._short_2_agent

    def _load_py_config(self, file_path):
        spec = importlib.util.spec_from_file_location("mod_cfg", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.cfg

    def _create_agent_from_config(self, cfg, mirror_mode=False):
        agent = D3QN_PER_Agent(
            state_shape=cfg.seq.state_shape, action_dim=cfg.market.num_actions,
            cnn_maps=cfg.model.cnn_maps, cnn_kernels=cfg.model.cnn_kernels,
            cnn_strides=cfg.model.cnn_strides, 
            cnn_dilations=getattr(cfg.model, 'cnn_dilations', [1]*len(cfg.model.cnn_maps)),
            dense_val=cfg.model.dense_val, dense_adv=cfg.model.dense_adv,
            additional_feats=cfg.model.additional_feats, dropout_model=cfg.model.dropout_p,
            device=torch.device("cpu"), gamma=cfg.rl.gamma, learning_rate=cfg.rl.lr,
            batch_size=cfg.rl.batch_size, buffer_size=cfg.per.buffer_size,
            target_update_freq=cfg.rl.target_update_freq, train_start=cfg.rl.train_start,
            per_alpha=cfg.per.per_alpha, per_beta_start=cfg.per.per_beta_start,
            per_beta_frames=cfg.per.per_beta_frames, eps_start=cfg.eps.eps_start,
            eps_end=cfg.eps.eps_end, eps_frames=cfg.eps.eps_decay_frames,
            epsilon=0.0, max_gradient_norm=cfg.rl.max_gradient_norm
        )
        agent.mirror_mode = mirror_mode
        return agent

    def _load_weights(self, agent, onnx_path, name):
        if onnx_path.exists():
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.config.get('cpu_threads', 12)
            agent.ort_session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=['CPUExecutionProvider'])
            self.logger.info(f"[OK] {name} ONNX Loaded")

    def _run_global_batch_inference(self, current_time: datetime):
        if self._is_batch_processing: return
        self._is_batch_processing = True
        try:
            whitelist = self.dp.current_whitelist()
            self._batch_cache = {}
            tasks = {"long_1": [], "long_2": [], "short_1": [], "short_2": []}
            pairs_in_batch = {"long_1": [], "long_2": [], "short_1": [], "short_2": []}

            for pair in whitelist:
                df = self.dp.get_pair_dataframe(pair, self.timeframe)
                if df is None or len(df) < 180: continue
                raw_cols = ['open', 'high', 'low', 'close', 'volume']
                z_slice = df[raw_cols].iloc[-180:].copy()
                z_slice['volume'] = np.log1p(z_slice['volume'])
                mean, std = z_slice.mean().values, z_slice.std().values + 1e-6
                normalized = (z_slice.iloc[-90:].values - mean) / std
                
                def prep_input(data, invert=False):
                    d = data.copy()
                    if invert:
                        d[:, :4] *= -1.0
                        d[:, [1, 2]] = d[:, [2, 1]]
                    img = d.T.flatten()
                    return np.expand_dims(np.concatenate([img, np.zeros(4, dtype=np.float32)]), axis=0).astype(np.float32)

                if self.enable_long_1: 
                    tasks["long_1"].append(prep_input(normalized))
                    pairs_in_batch["long_1"].append(pair)
                if self.enable_long_2:
                    tasks["long_2"].append(prep_input(normalized))
                    pairs_in_batch["long_2"].append(pair)
                if self.enable_short_1:
                    tasks["short_1"].append(prep_input(normalized, self.short_1_agent.mirror_mode))
                    pairs_in_batch["short_1"].append(pair)
                if self.enable_short_2:
                    tasks["short_2"].append(prep_input(normalized, self.short_2_agent.mirror_mode))
                    pairs_in_batch["short_2"].append(pair)

            for m_name, model_tasks in tasks.items():
                if not model_tasks: continue
                batch_input = np.vstack(model_tasks).astype(np.float32)
                agent = getattr(self, f"{m_name}_agent")
                if agent and agent.ort_session:
                    res = agent.ort_session.run(None, {agent.ort_session.get_inputs()[0].name: batch_input})[0]
                    for i, pair in enumerate(pairs_in_batch[m_name]):
                        if pair not in self._batch_cache: self._batch_cache[pair] = {}
                        self._batch_cache[pair][m_name] = res[i:i+1, :]
            self.logger.info(f"[BATCH] INFERENCE COMPLETE: {len(whitelist)} pairs")
        finally:
            self._is_batch_processing = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['close']
        is_live = self.config.get('runmode') in ['live', 'dry_run']
        vol_sma_window = 1440 if not is_live else 200 
        dataframe['quote_volume_sma'] = dataframe['quote_volume'].rolling(window=vol_sma_window, min_periods=100).mean().fillna(0)

        high_low_range = dataframe['high'] - dataframe['low']
        buy_pressure = np.where(high_low_range > 0, (dataframe['close'] - dataframe['low']) / high_low_range, 0.5)
        dataframe['buy_vol'] = dataframe['volume'] * buy_pressure
        dataframe['sell_vol'] = dataframe['volume'] * (1.0 - buy_pressure)
        dataframe['vol_sma_dyn'] = dataframe['volume'].rolling(window=31, min_periods=15).mean()
        dataframe['surge_ratio'] = dataframe['volume'] / (dataframe['vol_sma_dyn'] + 1e-8)
        dataframe['buy_vol_pct'] = (dataframe['buy_vol'] / (dataframe['volume'] + 1e-8)) * 100
        dataframe['sell_vol_pct'] = (dataframe['sell_vol'] / (dataframe['volume'] + 1e-8)) * 100
        dataframe['cvd'] = (dataframe['buy_vol'] - dataframe['sell_vol']).cumsum()
        dataframe['cvd_ma'] = dataframe['cvd'].rolling(window=83, min_periods=40).mean()
        dataframe['gap_pct_long'] = dataframe['buy_vol'] / (dataframe['sell_vol'] + 1e-8)
        dataframe['gap_pct_short'] = dataframe['sell_vol'] / (dataframe['buy_vol'] + 1e-8)

        dataframe['volume_orig'] = dataframe['volume'].copy()
        dataframe['volume'] = np.log1p(dataframe['volume'])
        
        if not is_live:
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                rolling = dataframe[col].rolling(window=90, min_periods=90)
                dataframe[f'{col}_z'] = ((dataframe[col] - rolling.mean()) / (rolling.std(ddof=0) + 1e-6)).astype(np.float32)
            dataframe.fillna(0.0, inplace=True)

        ema_period = int(self.ema_fast_period.value)
        dataframe['ema_fast'] = dataframe['close'].ewm(span=ema_period, adjust=False).mean()
        dataframe['st_regime_local'] = np.where(dataframe['close'] > dataframe['ema_fast'], 1, -1)
        dataframe['st_regime_global'] = dataframe['st_regime_local']
        return dataframe

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str) -> Optional[torch.Tensor]:
        window = 90
        should_invert = (side == "SHORT") and ((model_num == 1 and self.short_1_agent.mirror_mode) or (model_num == 2 and self.short_2_agent.mirror_mode))
        if self.config.get('runmode') in ('live', 'dry_run'):
            if 'open_z' in dataframe.columns:
                last_window = dataframe[['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']].iloc[-window:].values.copy().astype(np.float32)
            else:
                raw_cols = ['open', 'high', 'low', 'close', 'volume']
                df_slice = dataframe[raw_cols].iloc[-180:].copy()
                df_slice['volume'] = np.log1p(df_slice['volume'])
                mean, std = df_slice.mean().values, df_slice.std().values + 1e-6
                last_window = ((df_slice.iloc[-window:].values - mean) / std).astype(np.float32)
            if should_invert:
                last_window[:, :4] *= -1.0
                last_window[:, [1, 2]] = last_window[:, [2, 1]]
            img = last_window.T.flatten()
            return np.expand_dims(np.concatenate([img, np.zeros(4, dtype=np.float32)]), axis=0).astype(np.float32)
        else:
            if 'open_z' not in dataframe.columns: return None
            z_data = dataframe[['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']].values.astype(np.float32)
            windows = sliding_window_view(z_data, window_shape=(window, 5)).squeeze(1).copy()
            if should_invert:
                windows[:, :4] *= -1.0
                windows[:, [1, 2]] = windows[:, [2, 1]]
            img_batch = windows.transpose(0, 2, 1).reshape(len(windows), -1)
            return np.concatenate([img_batch, np.zeros((len(windows), 4), dtype=np.float32)], axis=1).astype(np.float32)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty: return dataframe
        dataframe['impulse_long_ok'] = (dataframe['close'] / dataframe['open']) < 1.005
        dataframe['impulse_short_ok'] = (dataframe['close'] / dataframe['open']) > 0.995
        
        # Ускоряем: Привязка к системному времени (строго 1 раз в минуту для всего вайтлиста)
        time_now_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).timestamp()
        with self._batch_lock:
            if self._last_batch_ts != time_now_ts and self.config.get('runmode') in ['live', 'dry_run']:
                self._run_global_batch_inference(datetime.now(timezone.utc))
                self._last_batch_ts = time_now_ts

        if len(dataframe) < self.startup_candle_count: return dataframe
        q_values = self._batch_cache.get(metadata['pair'], {})
        if not q_values and self.config.get('runmode') not in ['live', 'dry_run']:
            asset_name = metadata['pair'].split(':')[0].replace('/', '')
            for m in ['long_1', 'long_2', 'short_1', 'short_2']:
                side, num = m.split('_')
                tensor = self.get_model_input(dataframe, metadata['pair'], side.upper(), int(num), asset_name)
                if tensor is not None:
                    agent = getattr(self, f"{m}_agent")
                    if agent and agent.ort_session:
                        q_values[m] = agent.ort_session.run(None, {agent.ort_session.get_inputs()[0].name: tensor})[0]

        if not q_values: return dataframe

        def get_sig(name, idx, side_eps):
            if name not in q_values: return 0
            q = q_values[name]
            adv = q[:, idx] - q[:, 0]
            norm_cfg = self.q_normalization.get(name, {})
            q_min, q_max = norm_cfg.get('q_min', 0.0), norm_cfg.get('q_max', 0.01)
            thr = q_min + (q_max - q_min) * side_eps
            return (adv > thr).astype(np.int8)

        sig_l1 = get_sig("long_1", 1, self.rl_epsilon_long.value)
        sig_l2 = get_sig("long_2", 1, self.rl_epsilon_long.value)
        sig_s1 = get_sig("short_1", 1, self.rl_epsilon_short.value)
        sig_s2 = get_sig("short_2", 1, self.rl_epsilon_short.value)

        l_vote = sig_l1 + sig_l2
        s_vote = sig_s1 + sig_s2
        l_final = (l_vote >= self.rl_long_threshold_opt.value) & (s_vote == 0)
        s_final = (s_vote >= self.rl_short_threshold_opt.value) & (l_vote == 0)
        
        l_final &= dataframe['impulse_long_ok'].iloc[-len(l_final):].values
        s_final &= dataframe['impulse_short_ok'].iloc[-len(s_final):].values
        
        if self.config.get('runmode') in ['live', 'dry_run']:
            l_final &= (dataframe['st_regime_global'].iloc[-1] > 0)
            s_final &= (dataframe['st_regime_global'].iloc[-1] < 0)

        target_idx = dataframe.index[-len(l_final):]
        dataframe.loc[target_idx, 'enter_long'] = l_final.astype(np.int8)
        dataframe.loc[target_idx, 'enter_short'] = s_final.astype(np.int8)
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        # ЖЕСТКИЙ ЛИМИТ 20/20 (только для Live/Dry-run)
        open_trades = Trade.get_open_trades()
        if side == 'long':
            long_count = len([t for t in open_trades if not t.is_short])
            return long_count < 20
        else:
            short_count = len([t for t in open_trades if t.is_short])
            return short_count < 20

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        leverage_dict = self.config.get('leverage', {})
        return float(leverage_dict.get(pair, leverage_dict.get('*', 1.0)))

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        trade_id = trade.id
        if trade_id not in self.tsl_memory: self.tsl_memory[trade_id] = current_profit
        self.tsl_memory[trade_id] = max(self.tsl_memory[trade_id], current_profit)
        calc_p = self.tsl_memory[trade_id]
        if calc_p <= 0.0008: return -self.d0.value
        p_norm = min(1.0, (calc_p - 0.0008) / self.p_target.value)
        d_eff = self.d0.value - (self.d0.value - self.d_min.value) * (p_norm**self.tsl_exponent.value)
        return -max(self.d_min.value, d_eff)
