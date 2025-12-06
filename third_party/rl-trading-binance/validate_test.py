import os
import sys
# Set CUBLAS workspace config to ensure determinism, must be done before torch import
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import datetime
import logging
import glob
import time
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

try:
    from trading_environment import TradingEnvironment
    from agent import D3QN_PER_Agent
    from model import DuelingQNetwork 
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

class PerformanceConfig:
    def __init__(self):
        self.use_amp = True; self.amp_dtype = "float16"; self.compile_mode = False; self.compile_dynamic = False

def create_validation_episodes(
    val_sequences, val_keys, num_episodes=750, max_episodes_per_symbol=10, seed=404
):
    if not val_sequences:
        return [], []
    episodes_by_symbol = defaultdict(list)
    for i, key in enumerate(val_keys):
        symbol = key.split('_')[0]
        episodes_by_symbol[symbol].append(i)
    
    selected_indices = []
    for symbol, indices in episodes_by_symbol.items():
        n_samples = min(len(indices), max_episodes_per_symbol)
        random.seed(seed)
        selected_indices.extend(random.sample(indices, n_samples))
    
    if len(selected_indices) > num_episodes:
        random.seed(seed)
        final_indices = random.sample(selected_indices, num_episodes)
    else:
        final_indices = selected_indices
    
    random.seed(seed)
    random.shuffle(final_indices)
    
    final_sequences = [val_sequences[i] for i in final_indices]
    final_keys = [val_keys[i] for i in final_indices]
    
    final_symbols = {val_keys[i].split('_')[0] for i in final_indices}
    logging.info(f"Stratified sampling complete. Sampled episodes: {len(final_sequences)}, Symbol coverage: {len(final_symbols)}/{len(episodes_by_symbol)}")
    
    return final_sequences, final_keys

def load_config_from_path(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    loader = SourceFileLoader("config_module", config_path)
    config_module = loader.load_module()
    return config_module.cfg

def find_model_checkpoint(cfg=None):
    if cfg and hasattr(cfg, 'paths') and hasattr(cfg.paths, 'model_path') and os.path.exists(cfg.paths.model_path):
        logger.info(f"ℹ️ Using model path from config: {cfg.paths.model_path}")
        return cfg.paths.model_path
    
    model_dir_from_cfg = "."
    if cfg and hasattr(cfg, 'paths') and cfg.paths.model_path:
        model_dir_from_cfg = os.path.dirname(cfg.paths.model_path)

    search_path = os.path.join(model_dir_from_cfg, "best.pth")
    if os.path.exists(search_path):
        return search_path
    
    files = glob.glob(os.path.join(model_dir_from_cfg, "**", "best.pth"), recursive=True)
    if files:
        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"ℹ️ Found latest model checkpoint: {latest_file}")
        return latest_file
    return None

def load_true_config(model_path):
    if model_path is None: return None
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config_train.json")
    if os.path.exists(config_path):
        logger.info(f"ℹ️ Loading ground truth config from: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    logger.warning(f"⚠️ config_train.json not found in model directory.")
    return None

def load_and_normalize_data(npz_path, norm_stats_path, paper_symbols_cfg):
    logger.info(f"📂 Loading data from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data file not found: {npz_path}")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")

    with open(norm_stats_path, 'r') as f:
        all_stats = json.load(f)

    allowed_assets = paper_symbols_cfg
    if allowed_assets == "ALL":
        allowed_assets = None

    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    
    logger.info(f"Normalizing data for specified symbols: {allowed_assets or 'ALL'}")
    for key in tqdm(data_keys, desc="Normalizing validation data"):
        try:
            asset_name = key.split('_')[0]
        except IndexError:
            continue
        
        if allowed_assets and asset_name not in allowed_assets:
            continue

        asset_specific_stats = all_stats.get(asset_name)
        if asset_specific_stats is None:
            continue

        means = np.array(asset_specific_stats['mean'])
        stds = np.array(asset_specific_stats['std'])
        
        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means):
            continue
            
        seq = (seq - means) / (stds + 1e-8)
        sequences.append(seq)
        valid_keys.append(key)
    
    d.close()
    if not sequences:
        raise ValueError("No validation sequences were loaded. Check data path and symbol configuration.")
    logger.info(f"Prepared {len(sequences)} validation sequences.")
    
    return sequences, all_stats, valid_keys

def run_validation():
    cli_config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not cli_config_path:
        print("❌ Please provide the path to a config file.")
        return

    user_cfg_module = load_config_from_path(cli_config_path)
    
    model_path = find_model_checkpoint(user_cfg_module)
    if not model_path:
        print("❌ 'best.pth' model file not found.")
        return

    train_cfg_dict = load_true_config(model_path)
    if not train_cfg_dict:
        print("❌ Could not load the ground truth config_train.json from the model's directory.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    norm_stats_path = os.path.join(os.path.dirname(model_path), "norm_stats.json")
    
    val_data_path = train_cfg_dict.get("paths", {}).get("val_data_path", "data/val_data_fair_2m.npz")
    if not os.path.isabs(val_data_path):
        val_data_path = os.path.join(script_dir, val_data_path)

    logger.info(f"ℹ️ Using Validation Data: {val_data_path}")
    logger.info(f"ℹ️ Using Normalization Stats: {norm_stats_path}")

    paper_symbols = train_cfg_dict.get("paper", {}).get("symbols", "ALL")
    sequences, all_stats, keys = load_and_normalize_data(val_data_path, norm_stats_path, paper_symbols)
    
    trainlog_cfg = train_cfg_dict.get("trainlog", {})
    sequences, keys = create_validation_episodes(
        val_sequences=sequences,
        val_keys=keys,
        num_episodes=trainlog_cfg.get("num_val_ep", 750),
        seed=train_cfg_dict.get("random_seed", 404)
    )

    seq_cfg = train_cfg_dict.get("seq", {})
    data_cfg = train_cfg_dict.get("data", {})
    market_cfg = train_cfg_dict.get("market", {})
    backtest_cfg = train_cfg_dict.get("backtest", {})
    model_cfg = train_cfg_dict.get("model", {})
    rl_cfg = train_cfg_dict.get("rl", {})
    per_cfg = train_cfg_dict.get("per", {})
    eps_cfg = train_cfg_dict.get("eps", {})

    input_history_len = seq_cfg.get("input_history_len") or seq_cfg.get("agent_history_len", 90)

    env_params = {
        "full_seq_len": seq_cfg.get("full_seq_len", 150),
        "pre_signal_len": seq_cfg.get("pre_signal_len", 90),
        "agent_history_len": seq_cfg.get("agent_history_len", 90),
        "agent_session_len": seq_cfg.get("agent_session_len", 60),
        "input_history_len": input_history_len,
        "initial_balance": market_cfg.get("initial_balance", 10000.0),
        "transaction_fee": market_cfg.get("transaction_fee"),
        "slippage": market_cfg.get("slippage", 0.0002),
        "num_actions": market_cfg.get("num_actions", 4),
        "inaction_penalty_ratio": market_cfg.get("inaction_penalty_ratio", 0.0),
        "position_fraction": market_cfg.get("position_fraction", 0.1),
        "order_size_usdt": backtest_cfg.get("order_size_usdt", 0.0),
        "backtest_mode": True,
        "use_risk_management": backtest_cfg.get("use_risk_management", False),
        "cnn_format": False,
        "num_features": data_cfg.get("numchannels", 10),
        "action_history_len": seq_cfg.get("action_history_len", 2),
        "datachannels": data_cfg.get("datachannels", []),
        "pricechannels": data_cfg.get("pricechannels", []),
        "volumechannels": data_cfg.get("volumechannels", []),
        "otherchannels": data_cfg.get("otherchannels", []),
    }
    
    flat_state_size = (env_params["agent_history_len"] * env_params["num_features"]) + 4 + (env_params["num_actions"] * env_params["action_history_len"])
    env_params["flat_state_size"] = flat_state_size

    backtest_kwargs = {
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": backtest_cfg.get("trailing_stop"),
        "trailing_stop_min": backtest_cfg.get("trailing_stop_min"),
        "fee_buffer_mult": backtest_cfg.get("fee_buffer_mult"),
        "delta_p_hysteresis": backtest_cfg.get("delta_p_hysteresis"),
    }

    logger.info("🔧 Initializing TradingEnvironment...")
    env = TradingEnvironment(sequences=sequences, stats=all_stats, keys=keys, render_mode=None, **env_params)

    agent = D3QN_PER_Agent(
        state_shape=train_cfg_dict.get("state_shape", [10, 90, 1]),
        action_dim=env_params["num_actions"],
        cnn_maps=model_cfg.get("cnn_maps", []),
        cnn_kernels=model_cfg.get("cnn_kernels", []),
        cnn_strides=model_cfg.get("cnn_strides", []),
        cnn_dilations=model_cfg.get("cnn_dilations", []),
        dense_val=model_cfg.get("dense_val", []),
        dense_adv=model_cfg.get("dense_adv", []),
        additional_feats=model_cfg.get("additional_feats", 12),
        dropout_model=model_cfg.get("dropout_p", 0.0),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma=rl_cfg.get("gamma", 0.99),
        learning_rate=rl_cfg.get("lr", 1e-4),
        batch_size=rl_cfg.get("batch_size", 32),
        buffer_size=per_cfg.get("buffer_size", 100000),
        target_update_freq=rl_cfg.get("target_update_freq", 1000),
        train_start=rl_cfg.get("train_start", 1000),
        max_gradient_norm=rl_cfg.get("max_gradient_norm", 1.0),
        per_alpha=per_cfg.get("per_alpha", 0.6),
        per_beta_start=per_cfg.get("per_beta_start", 0.4),
        per_beta_frames=per_cfg.get("per_beta_frames", 100000),
        eps_start=eps_cfg.get("eps_start", 1.0),
        eps_end=eps_cfg.get("eps_end", 0.05),
        eps_frames=eps_cfg.get("eps_decay_frames", 100000),
        epsilon=0.0,
        perf_cfg=PerformanceConfig()
    )
    
    logger.info("🤖 Initializing Agent...")
    agent.load_model(model_path, strict=True)

    logger.info("🚀 Starting Backtest Validation...")
    
    all_trades_info = []
    total_bars_processed = 0
    start_time = time.time()
    
    logging.getLogger().setLevel(logging.ERROR)
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        
        signal_dt_for_step = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ticker_name = "UNKNOWN"
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                if len(key_parts) > 1:
                    start_dt_str = key_parts[1]
                    signal_dt_for_step = datetime.datetime.fromisoformat(start_dt_str.replace("Z", "+00:00"))
            except (IndexError, AttributeError, ValueError) as e:
                logger.warning(f"Could not parse ticker/date from key: {keys[i]} due to {e}")

        while not done:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.backtest_step(
                action=action, 
                signal_dt=signal_dt_for_step,
                ticker=ticker_name, 
                **backtest_kwargs
            )
            done = terminated or truncated
            obs = next_obs
            total_bars_processed += 1
            
            if info.get("position_closed", False):
                all_trades_info.append(info)
        
        pbar.set_postfix({
            "PnL": f"{sum(t.get('trade_realized_pnl', 0.0) for t in all_trades_info):,.0f}", 
            "Trds": len(all_trades_info)
        })

    logging.getLogger().setLevel(logging.INFO)

    # --- Metrics Calculation ---
    total_duration = time.time() - start_time
    total_trades = len(all_trades_info)
    win_count = sum(1 for t in all_trades_info if t.get('trade_realized_pnl', 0.0) > 0)
    loss_count = total_trades - win_count
    wr_ratio = win_count / max(1, total_trades)

    gross_pnl = sum(t.get('trade_realized_pnl', 0.0) + t.get('trade_commission', 0.0) for t in all_trades_info)
    net_pnl = sum(t.get('trade_realized_pnl', 0.0) for t in all_trades_info)
    total_commission = sum(t.get('trade_commission', 0.0) for t in all_trades_info)
    avg_pnl_per_trade = net_pnl / max(1, total_trades)

    trade_pnls = [t.get('trade_realized_pnl', 0.0) for t in all_trades_info]
    best_trade = max(trade_pnls) if trade_pnls else 0.0
    worst_trade = min(trade_pnls) if trade_pnls else 0.0

    long_trades = sum(1 for t in all_trades_info if t.get('direction') == 'LONG')
    short_trades = sum(1 for t in all_trades_info if t.get('direction') == 'SHORT')

    holding_times = [t.get('holding_duration_bars', 0) for t in all_trades_info]
    avg_holding_time = np.mean(holding_times) if holding_times else 0.0
    max_holding_time = max(holding_times) if holding_times else 0.0
    min_holding_time = min(holding_times) if holding_times else 0.0

    bars_per_day = 1440
    trading_time_days = total_bars_processed / bars_per_day if bars_per_day > 0 else 0.0
    pnl_per_day = net_pnl / max(1, trading_time_days)

    initial_balance = env_params["initial_balance"]
    roi_percent = (net_pnl / initial_balance) * 100
    roi_annualized = roi_percent * (365.0 / trading_time_days) if trading_time_days > 0 else 0.0
    
    pos_pnls = [p for p in trade_pnls if p > 0]
    neg_pnls = [p for p in trade_pnls if p < 0]
    avg_win_size = np.mean(pos_pnls) if pos_pnls else 0.0
    avg_loss_size = np.mean(neg_pnls) if neg_pnls else 0.0
    win_loss_ratio = abs(avg_win_size / avg_loss_size) if avg_loss_size != 0 else float('inf')
    expectancy = (wr_ratio * avg_win_size) + ((1 - wr_ratio) * avg_loss_size)
    
    profit_factor = sum(pos_pnls) / max(1e-9, abs(sum(neg_pnls)))

    # Max Drawdown
    equity_curve = np.cumsum([initial_balance] + trade_pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Sharpe & Sortino
    returns = np.array(trade_pnls) / initial_balance
    if len(returns) > 1:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        downside_std = np.std(returns[returns < 0], ddof=1) if len(returns[returns < 0]) > 1 else 1e-9
        sharpe = (mean_r / std_r) if std_r > 1e-9 else 0.0
        sortino = (mean_r / downside_std) if downside_std > 1e-9 else 0.0
    else:
        sharpe, sortino = 0.0, 0.0

    tsl_hits = sum(1 for t in all_trades_info if t.get('tsl_triggered', False))

    # --- Print Results ---
    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*44)
    print(f"Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}, Win: {win_count}, Loss: {loss_count}) | WinRate: {wr_ratio:.2%} | PF: {profit_factor:.4f}")
    print(f"Gross PnL: {gross_pnl:.2f} | Net PnL: {net_pnl:.2f} | Commission: {total_commission:.2f} | Avg/Trade: {avg_pnl_per_trade:.2f}")
    print(f"Best Trade: {best_trade:+.2f} | Worst Trade: {worst_trade:+.2f} | MaxDD: {abs(max_dd):.2%} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}")
    print(f"Avg Hold: {avg_holding_time:.2f} bars | Min Hold: {min_holding_time} bars | Max Hold: {max_holding_time} bars")
    print(f"Duration: {total_duration:.2f}s | Bars: {total_bars_processed} | Trading Days: {trading_time_days:.1f}")
    print(f"PnL/Day: {pnl_per_day:.2f} USDT | ROI: {roi_percent:.2f}% | Annualized ROI: {roi_annualized:.1f}%")
    print(f"Commission: {(total_commission / max(1e-9, abs(gross_pnl)))*100:.1f}% of gross | Avg Win: {avg_win_size:.2f} | Avg Loss: {avg_loss_size:.2f} | W/L Ratio: {win_loss_ratio:.2f}")
    print(f"Expectancy/Trade: {expectancy:.2f} USDT")
    print(f"TSL hits: {tsl_hits} ({tsl_hits/max(1, total_trades):.2%})")
    print("="*44)


def run_validation_with_config(config_path: str, model_path: str = None, override_params: dict = None) -> dict:
    """
    Программный запуск валидации с возможностью override параметров.
    
    Args:
        config_path: путь к конфиг-файлу
        model_path: путь к модели (опционально, будет искать автоматически)
        override_params: dict с параметрами для override (например, market.* параметры)
    
    Returns:
        dict с метриками: sharpe, sortino, profit_factor, max_drawdown, win_rate, trades, net_pnl
    """
    user_cfg_module = load_config_from_path(config_path)
    
    # === OVERRIDE PARAMETERS (для Optuna) ===
    if override_params:
        for path, value in override_params.items():
            parts = path.split('.')
            obj = user_cfg_module
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            logger.debug(f"[Override] {path} = {value}")
    
    if not model_path:
        model_path = find_model_checkpoint(user_cfg_module)
    
    if not model_path:
        raise FileNotFoundError("'best.pth' model file not found.")
    
    train_cfg_dict = load_true_config(model_path)
    if not train_cfg_dict:
        raise FileNotFoundError("Could not load config_train.json from model's directory.")
    
    # === MERGE OVERRIDES INTO train_cfg_dict ===
    if override_params:
        for path, value in override_params.items():
            parts = path.split('.')
            obj = train_cfg_dict
            for part in parts[:-1]:
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]
            obj[parts[-1]] = value
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    norm_stats_path = os.path.join(os.path.dirname(model_path), "norm_stats.json")
    val_data_path = train_cfg_dict.get("paths", {}).get("val_data_path", "data/val_data_fair_2m.npz")
    
    if not os.path.isabs(val_data_path):
        val_data_path = os.path.join(script_dir, val_data_path)
    
    paper_symbols = train_cfg_dict.get("paper", {}).get("symbols", "ALL")
    sequences, all_stats, keys = load_and_normalize_data(val_data_path, norm_stats_path, paper_symbols)
    
    trainlog_cfg = train_cfg_dict.get("trainlog", {})
    sequences, keys = create_validation_episodes(
        val_sequences=sequences,
        val_keys=keys,
        num_episodes=trainlog_cfg.get("num_val_ep", 150),  # Для Optuna используем меньше эпизодов
        seed=train_cfg_dict.get("random_seed", 404)
    )
    
    seq_cfg = train_cfg_dict.get("seq", {})
    data_cfg = train_cfg_dict.get("data", {})
    market_cfg = train_cfg_dict.get("market", {})
    backtest_cfg = train_cfg_dict.get("backtest", {})
    model_cfg = train_cfg_dict.get("model", {})
    rl_cfg = train_cfg_dict.get("rl", {})
    per_cfg = train_cfg_dict.get("per", {})
    eps_cfg = train_cfg_dict.get("eps", {})
    
    input_history_len = seq_cfg.get("input_history_len") or seq_cfg.get("agent_history_len", 90)
    
    env_params = {
        "full_seq_len": seq_cfg.get("full_seq_len", 150),
        "pre_signal_len": seq_cfg.get("pre_signal_len", 90),
        "agent_history_len": seq_cfg.get("agent_history_len", 90),
        "agent_session_len": seq_cfg.get("agent_session_len", 60),
        "input_history_len": input_history_len,
        "initial_balance": market_cfg.get("initial_balance", 10000.0),
        "transaction_fee": market_cfg.get("transaction_fee"),
        "slippage": market_cfg.get("slippage", 0.0002),
        "num_actions": market_cfg.get("num_actions", 4),
        "inaction_penalty_ratio": market_cfg.get("inaction_penalty_ratio", 0.0),
        "position_fraction": market_cfg.get("position_fraction", 0.1),
        "order_size_usdt": backtest_cfg.get("order_size_usdt", 0.0),
        "backtest_mode": True,
        "use_risk_management": backtest_cfg.get("use_risk_management", False),
        "cnn_format": False,
        "num_features": data_cfg.get("numchannels", 10),
        "action_history_len": seq_cfg.get("action_history_len", 2),
        "datachannels": data_cfg.get("datachannels", []),
        "pricechannels": data_cfg.get("pricechannels", []),
        "volumechannels": data_cfg.get("volumechannels", []),
        "otherchannels": data_cfg.get("otherchannels", []),
        # === НАГРАДЫ/ШТРАФЫ (из market_cfg) ===
        "new_equity_peak_reward": market_cfg.get("new_equity_peak_reward", 0.005),
        "perfect_entry_reward": market_cfg.get("perfect_entry_reward", 0.05),
        "risk_reward_ratio_threshold": market_cfg.get("risk_reward_ratio_threshold", 3.0),
        "risk_reward_ratio_reward": market_cfg.get("risk_reward_ratio_reward", 0.075),
        "good_exit_bonus": market_cfg.get("good_exit_bonus", 0.30),
        "fast_exit_bonus": market_cfg.get("fast_exit_bonus", 0.10),
        "bankruptcy_threshold": market_cfg.get("bankruptcy_threshold", 0.0),
        "bankruptcy_penalty": market_cfg.get("bankruptcy_penalty", 1.0),
        "bankruptcy_slippage_penalty": market_cfg.get("bankruptcy_slippage_penalty", 0.05),
        "max_drawdown_threshold": market_cfg.get("max_drawdown_threshold", -0.20),
        "max_drawdown_penalty_type": market_cfg.get("max_drawdown_penalty_type", "proportional"),
        "max_drawdown_penalty": market_cfg.get("max_drawdown_penalty", 1.0),
        "continuous_pain_penalty_ratio": market_cfg.get("continuous_pain_penalty_ratio", 0.12),
        "low_balance_penalty": market_cfg.get("low_balance_penalty", 0.01),
        "holding_penalty_multiplier": market_cfg.get("holding_penalty_multiplier", 0.2),
        "greed_penalty_multiplier": market_cfg.get("greed_penalty_multiplier", 0.1),
        "premature_exit_penalty": market_cfg.get("premature_exit_penalty", 0.25),
        "holding_penalty_threshold": market_cfg.get("holding_penalty_threshold", 15),
        "greed_penalty_threshold": market_cfg.get("greed_penalty_threshold", 0.50),
        "exit_quality_threshold": market_cfg.get("exit_quality_threshold", 0.80),
        "fast_exit_threshold": market_cfg.get("fast_exit_threshold", 20),
        "premature_exit_threshold": market_cfg.get("premature_exit_threshold", 8),
        "profit_exit_threshold": market_cfg.get("profit_exit_threshold", 8),
        "loss_exit_threshold": market_cfg.get("loss_exit_threshold", 3),
    }
    
    flat_state_size = (env_params["agent_history_len"] * env_params["num_features"]) + 4 + (env_params["num_actions"] * env_params["action_history_len"])
    env_params["flat_state_size"] = flat_state_size
    
    backtest_kwargs = {
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": backtest_cfg.get("trailing_stop"),
        "trailing_stop_min": backtest_cfg.get("trailing_stop_min"),
        "fee_buffer_mult": backtest_cfg.get("fee_buffer_mult"),
        "delta_p_hysteresis": backtest_cfg.get("delta_p_hysteresis"),
    }
    
    # Создание окружения и агента (код остается как в оригинале)
    env = TradingEnvironment(sequences=sequences, stats=all_stats, keys=keys, render_mode=None, **env_params)
    
    agent = D3QN_PER_Agent(
        state_shape=train_cfg_dict.get("state_shape", [10, 90, 1]),
        action_dim=env_params["num_actions"],
        cnn_maps=model_cfg.get("cnn_maps", []),
        cnn_kernels=model_cfg.get("cnn_kernels", []),
        cnn_strides=model_cfg.get("cnn_strides", []),
        cnn_dilations=model_cfg.get("cnn_dilations", []),
        dense_val=model_cfg.get("dense_val", []),
        dense_adv=model_cfg.get("dense_adv", []),
        additional_feats=model_cfg.get("additional_feats", 12),
        dropout_model=model_cfg.get("dropout_p", 0.0),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma=rl_cfg.get("gamma", 0.99),
        learning_rate=rl_cfg.get("lr", 1e-4),
        batch_size=rl_cfg.get("batch_size", 32),
        buffer_size=per_cfg.get("buffer_size", 100000),
        target_update_freq=rl_cfg.get("target_update_freq", 1000),
        train_start=rl_cfg.get("train_start", 1000),
        max_gradient_norm=rl_cfg.get("max_gradient_norm", 1.0),
        per_alpha=per_cfg.get("per_alpha", 0.6),
        per_beta_start=per_cfg.get("per_beta_start", 0.4),
        per_beta_frames=per_cfg.get("per_beta_frames", 100000),
        eps_start=eps_cfg.get("eps_start", 1.0),
        eps_end=eps_cfg.get("eps_end", 0.05),
        eps_frames=eps_cfg.get("eps_decay_frames", 100000),
        epsilon=0.0,
        perf_cfg=PerformanceConfig()
    )
    
    agent.load_model(model_path, strict=True)
    
    # Запуск симуляции (без прогресс-бара для Optuna)
    all_trades_info = []
    total_bars_processed = 0
    
    for i in range(len(sequences)):
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        signal_dt_for_step = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ticker_name = "UNKNOWN"
        
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                if len(key_parts) > 1:
                    start_dt_str = key_parts[1]
                    signal_dt_for_step = datetime.datetime.fromisoformat(start_dt_str.replace("Z", "+00:00"))
            except (IndexError, AttributeError, ValueError):
                pass
        
        while not done:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.backtest_step(
                action=action,
                signal_dt=signal_dt_for_step,
                ticker=ticker_name,
                **backtest_kwargs
            )
            done = terminated or truncated
            obs = next_obs
            total_bars_processed += 1
            
            if info.get("position_closed", False):
                all_trades_info.append(info)
    
    # Расчет метрик (код как в оригинале)
    total_trades = len(all_trades_info)
    win_count = sum(1 for t in all_trades_info if t.get('trade_realized_pnl', 0.0) > 0)
    wr_ratio = win_count / max(1, total_trades)
    
    trade_pnls = [t.get('trade_realized_pnl', 0.0) for t in all_trades_info]
    net_pnl = sum(trade_pnls)
    
    pos_pnls = [p for p in trade_pnls if p > 0]
    neg_pnls = [p for p in trade_pnls if p < 0]
    profit_factor = sum(pos_pnls) / max(1e-9, abs(sum(neg_pnls)))
    
    # Max Drawdown
    equity_curve = np.cumsum([env_params["initial_balance"]] + trade_pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    # Sharpe & Sortino
    returns = np.array(trade_pnls) / env_params["initial_balance"]
    if len(returns) > 1:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        downside_std = np.std(returns[returns < 0], ddof=1) if len(returns[returns < 0]) > 1 else 1e-9
        sharpe = (mean_r / std_r) if std_r > 1e-9 else 0.0
        sortino = (mean_r / downside_std) if downside_std > 1e-9 else 0.0
    else:
        sharpe, sortino = 0.0, 0.0
    
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_dd),
        "win_rate": float(wr_ratio),
        "trades": int(total_trades),
        "net_pnl": float(net_pnl)
    }


if __name__ == "__main__":
    run_validation()
