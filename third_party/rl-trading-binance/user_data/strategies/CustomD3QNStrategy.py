import sys
import json
import logging
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from numpy.lib.stride_tricks import sliding_window_view
try:
    from freqtrade.persistence import Trade  # type: ignore
except ImportError:
    class Trade: pass
from datetime import datetime

# --- 1. НАСТРОЙКА ПУТЕЙ ---
strategy_file = Path(__file__).resolve()
# Robustly find the project root (assuming we are in user_data/strategies)
if strategy_file.parent.name == 'strategies':
    project_root = strategy_file.parent.parent.parent
else:
    project_root = strategy_file.parent.parent.parent # Fallback

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Freqtrade imports
try:
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter  # type: ignore
except ImportError:
    logging.getLogger(__name__).error("Could not import freqtrade.strategy")
    class IStrategy: pass
    class DecimalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0.0)
    class IntParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0)

logger = logging.getLogger(__name__)

# Agent imports
try:
    from agent import D3QN_PER_Agent  # type: ignore
except ImportError as e:
    logger.error(f"CRITICAL: Could not import D3QN Agent! Check path: {project_root}")
    raise e

class CustomD3QNStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short = True
    startup_candle_count: int = 100
    minimal_roi = {"0": 100}
    stoploss = -0.99        # Заглушка, работает custom_stoploss
    trailing_stop = False   # Встроенный выключаем
    use_custom_stoploss = True # Явно разрешаем

    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Параметры для кастомного трейлинга (TSL)
    # ОБЯЗАТЕЛЬНО добавь space='stoploss'
    d0 = DecimalParameter(0.01, 0.10, default=0.075, space='stoploss', load=True)
    d_min = DecimalParameter(0.001, 0.05, default=0.01, space='stoploss', load=True)
    hysteresis = DecimalParameter(0.001, 0.02, default=0.002, space='stoploss', load=True)

    # Пороги агента обычно относятся к точкам входа
    long_threshold = DecimalParameter(0.0, 0.05, default=0.003, space='buy', load=True)
    short_threshold = DecimalParameter(0.0, 0.05, default=0.003, space='sell', load=True)

    # --- FreqUI PLOT CONFIG ---
    plot_config = {
        'main_plot': {},
        'subplots': {}
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.device = torch.device("cpu")
        
        # Определяем корень проекта относительно этого файла
        if Path(__file__).parent.name == 'strategies':
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(__file__).parent.parent.parent
        
        self.tsl_memory = {} 
        
        # --- ПУТИ К МОДЕЛЯМ ---
        self.short_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260118_time_225844"
        self.short_model_pth = self.short_model_dir / "best.pth"
        
        self.long_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260118_time_220542"
        self.long_model_pth = self.long_model_dir / "best.pth"
        
        # --- 1. ЗАГРУЗКА КОНФИГОВ ---
        cfg_file_short = self._find_config_file(self.short_model_dir)
        if not cfg_file_short: raise FileNotFoundError(f"Config .py not found in {self.short_model_dir}")
        logger.info(f"Loading SHORT config from {cfg_file_short}")
        self.cfg_short = self._load_py_config(cfg_file_short)

        cfg_file_long = self._find_config_file(self.long_model_dir)
        if not cfg_file_long: raise FileNotFoundError(f"Config .py not found in {self.long_model_dir}")
        logger.info(f"Loading LONG config from {cfg_file_long}")
        self.cfg_long = self._load_py_config(cfg_file_long)
        
        # --- 2. ЗАГРУЗКА NORM_STATS ---
        ns_path_short = self.short_model_dir / "norm_stats.json"
        if ns_path_short.exists():
            with open(ns_path_short, 'r') as f:
                self.norm_stats_short = json.load(f)
        else:
            raise FileNotFoundError(f"norm_stats.json missing in {self.short_model_dir}")

        ns_path_long = self.long_model_dir / "norm_stats.json"
        if ns_path_long.exists():
            with open(ns_path_long, 'r') as f:
                self.norm_stats_long = json.load(f)
        else:
             raise FileNotFoundError(f"norm_stats.json missing in {self.long_model_dir}")

        # --- 3. ИНИЦИАЛИЗАЦИЯ АГЕНТОВ ---
        self.long_agent = self._create_agent_from_config(self.cfg_long)
        self.short_agent = self._create_agent_from_config(self.cfg_short)

        # --- 4. ЗАГРУЗКА ВЕСОВ ---
        self._load_weights(self.long_agent, self.long_model_pth, "LONG")
        self._load_weights(self.short_agent, self.short_model_pth, "SHORT")

    def _find_config_file(self, dir_path: Path):
        for file in dir_path.glob("*.py"):
            if "alpha" in file.name or "config" in file.name:
                return file
        return None

    def _load_py_config(self, path: Path):
        unique_module_name = f"config_{path.parent.name}_{path.name}"
        spec = importlib.util.spec_from_file_location(unique_module_name, path)
        if spec is None or spec.loader is None: raise ImportError(f"Cannot load config from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = mod
        spec.loader.exec_module(mod)
        return mod.cfg

    def _create_agent_from_config(self, cfg):
        return D3QN_PER_Agent(
            state_shape=cfg.seq.state_shape,
            action_dim=cfg.market.num_actions,
            cnn_maps=cfg.model.cnn_maps,
            cnn_kernels=cfg.model.cnn_kernels,
            cnn_strides=cfg.model.cnn_strides,
            cnn_dilations=cfg.model.cnn_dilations,
            dense_val=cfg.model.dense_val,
            dense_adv=cfg.model.dense_adv,
            additional_feats=cfg.model.additional_feats,
            dropout_model=cfg.model.dropout_p,
            device=self.device,
            gamma=cfg.rl.gamma,
            learning_rate=cfg.rl.lr,
            batch_size=cfg.rl.batch_size,
            buffer_size=cfg.per.buffer_size,
            target_update_freq=cfg.rl.target_update_freq,
            train_start=cfg.rl.train_start,
            per_alpha=cfg.per.per_alpha,
            per_beta_start=cfg.per.per_beta_start,
            per_beta_frames=cfg.per.per_beta_frames,
            eps_start=cfg.eps.eps_start,
            eps_end=cfg.eps.eps_end,
            eps_frames=cfg.eps.eps_decay_frames,
            epsilon=0.0,
            max_gradient_norm=cfg.rl.max_gradient_norm
        )

    def _load_weights(self, agent, path, name):
        try:
            agent.load_model(str(path))
            agent.policy_net.eval()
            logger.info(f"✅ {name} Agent loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name} Agent: {e}")
            raise e

    def feature_engineering(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # Расчет волатильности (с защитой от деления на 0)
        dataframe['volatility_90m'] = (
            (dataframe['high'].rolling(90).max() - dataframe['low'].rolling(90).min()) / 
            (dataframe['low'].rolling(90).min() + 1e-9)
        )
        
        # VWAP и объемы
        dataframe['vwap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['vwap']
        dataframe['num_trades'] = dataframe['volume'] 
        dataframe['taker_base'] = dataframe['volume'] * 0.5
        dataframe['taker_quote'] = dataframe['quote_volume'] * 0.5
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Заполняем NaN нулями, чтобы FreqUI не «вешался»
        dataframe = dataframe.fillna(0.0)
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.feature_engineering(dataframe)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        if trade.open_date_utc:
            duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration_min >= 60:
                return "timeout_60m"
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        d0_val = self.d0.value
        d_min_val = self.d_min.value
        hysteresis_val = self.hysteresis.value
        FEE_BUF = 0.0008
        
        p = current_profit 
        trade_id = trade.id
        if trade_id not in self.tsl_memory:
            self.tsl_memory[trade_id] = -999.0

        last_p = self.tsl_memory[trade_id]
        if p >= (last_p + hysteresis_val):
            self.tsl_memory[trade_id] = p

        # Используем запомненное значение (ступенчатое)
        calc_p = self.tsl_memory[trade_id]

        if calc_p <= FEE_BUF:
            d_eff = d0_val
        else:
            d_eff = d0_val - (calc_p - FEE_BUF)
            
        d_eff = max(d_min_val, d_eff)
        return -d_eff

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str):
        # 1. Данные (5 каналов)
        opens = dataframe['open'].values
        highs = dataframe['high'].values
        lows = dataframe['low'].values
        closes = dataframe['close'].values
        volumes = dataframe['volume'].values
        
        # Log Returns
        eps = 1e-9
        def calc_log_returns(arr):
            changes = arr[1:] / (arr[:-1] + eps)
            return np.log(np.maximum(changes, eps))

        r_opens = calc_log_returns(opens)
        r_highs = calc_log_returns(highs)
        r_lows = calc_log_returns(lows)
        r_closes = calc_log_returns(closes)
        r_volumes = np.log(volumes[1:] + 1.0)
        
        # Stack (5, N-1)
        data = np.stack([r_opens, r_highs, r_lows, r_closes, r_volumes])
        
        if side == "SHORT":
            data = data * -1.0
            
        # 2. Нормализация
        if side == "LONG":
            current_norm_stats = self.norm_stats_long
        else:
            current_norm_stats = self.norm_stats_short

        # Принудительно используем загруженные статистики
        if "means" in current_norm_stats and isinstance(current_norm_stats["means"], dict):
            stats = current_norm_stats
            target_channels = ["open", "high", "low", "close", "volume"]
            means_val = [stats["means"].get(ch, 0.0) for ch in target_channels]
            stds_val = [stats["stds"].get(ch, 1.0) for ch in target_channels]
            means = np.array(means_val).reshape(-1, 1)
            stds = np.array(stds_val).reshape(-1, 1)
            
            data = (data - means) / (stds + 1e-8)
        elif "mean" in current_norm_stats and isinstance(current_norm_stats["mean"], list):
            stats = current_norm_stats
            # Берем первые 5 значений
            means_val = stats["mean"][:5]
            stds_val = stats["std"][:5]
            means = np.array(means_val).reshape(-1, 1)
            stds = np.array(stds_val).reshape(-1, 1)
            
            data = (data - means) / (stds + 1e-8)
        else:
            # Fallback (на всякий случай)
            df_data = pd.DataFrame(data.T)
            rolling = df_data.rolling(window=200, min_periods=1)
            means = rolling.mean().values.T
            stds = rolling.std().values.T
            means[np.isnan(means)] = 0.0
            stds[np.isnan(stds)] = 1.0
            data = (data - means) / (stds + 1e-8)
            
        # Handle NaNs from Rolling Norm (start of window)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Safety: Защита от NaN/Inf перед подачей в модель
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Sliding window (5, N-1) -> (5, (N-1)-89, 90)
        if data.shape[1] < 90:
            return None
            
        windows = sliding_window_view(data, window_shape=90, axis=1)
        # Transpose to (Batch, 5, 90)
        windows = windows.transpose(1, 0, 2)
        
        # 4. Flatten & Extra Features
        batch_size = windows.shape[0]
        flat_feats = windows.reshape(batch_size, -1)
        add_feats = np.zeros((batch_size, 4), dtype=np.float32)
        combined = np.concatenate([flat_feats, add_feats], axis=1)
        
        input_tensor = torch.FloatTensor(combined).to(self.device)
        return input_tensor

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        
        # В бэктесте сразу разрешаем, так как Trade.get_trades() там не работает
        if self.config.get('runmode') not in ['live', 'dry_run']:
            return True

        try:
            from freqtrade.persistence import Trade  # type: ignore
            trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
            current_shorts = sum(1 for t in trades if t.is_short)
            current_longs = sum(1 for t in trades if not t.is_short)
            MAX_LONGS = 50
            MAX_SHORTS = 50
            if side == "long":
                if current_longs >= MAX_LONGS: return False
            elif side == "short":
                if current_shorts >= MAX_SHORTS: return False
        except Exception:
            return True
            
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        if trade.id in self.tsl_memory:
            del self.tsl_memory[trade.id]
        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Базовая защита
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # 2. Оптимизация инференса (Deep Inference vs Fast Mode)
        deep_inference = self.config.get('deep_inference', False)

        if self.config.get('runmode') in ['live', 'dry_run']:
            # Live: только необходимый минимум (90 + 1)
            df_input = dataframe.iloc[-91:].copy()
        elif not deep_inference:
            # Backtest Fast Mode: Ограничиваем инференс последними N свечами
            # Это позволяет быстро проверить работу стратегии без OOM на огромной истории
            lookback = 1000
            if len(dataframe) > lookback:
                df_input = dataframe.iloc[-lookback:].copy()
            else:
                df_input = dataframe
            
            # Сброс сигналов (так как мы не считаем историю)
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
        else:
            # Backtest Full Mode: Полный прогон (ВНИМАНИЕ: Требует много RAM)
            df_input = dataframe
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0

        # 3. Инференс
        tensor_l = self.get_model_input(df_input, metadata['pair'], side="LONG")
        tensor_s = self.get_model_input(df_input, metadata['pair'], side="SHORT")
        
        if tensor_l is None or tensor_s is None:
            return dataframe

        with torch.no_grad():
            # Предсказание (Batch, Actions)
            q_l = self.long_agent.policy_net(tensor_l).cpu().numpy()
            q_s = self.short_agent.policy_net(tensor_s).cpu().numpy()
            
            # Расчет преимущества (Advantage): Q(Buy) - Q(Hold)
            adv_l = q_l[:, 1] - q_l[:, 0]
            adv_s = q_s[:, 1] - q_s[:, 0]
            
            # Запись в DF
            n_predictions = len(adv_l)
            target_idx = slice(-n_predictions, None)
            
            # Убедимся, что колонки существуют
            if 'enter_long' not in dataframe.columns: dataframe['enter_long'] = 0
            if 'enter_short' not in dataframe.columns: dataframe['enter_short'] = 0
            
            dataframe['enter_long'] = dataframe['enter_long'].astype(np.int8)
            dataframe['enter_short'] = dataframe['enter_short'].astype(np.int8)
            
            # Маски сигналов на основе порогов
            mask_l = adv_l > self.long_threshold.value
            mask_s = adv_s > self.short_threshold.value
            
            # Разрешение конфликтов
            conflict = mask_l & mask_s
            if np.any(conflict):
                # Где конфликт, оставляем только тот, где адвантаж больше
                mask_l[conflict] = adv_l[conflict] > adv_s[conflict]
                mask_s[conflict] = adv_s[conflict] >= adv_l[conflict]
            
            dataframe.iloc[target_idx, dataframe.columns.get_loc('enter_long')] = mask_l.astype(np.int8)
            dataframe.iloc[target_idx, dataframe.columns.get_loc('enter_short')] = mask_s.astype(np.int8)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
