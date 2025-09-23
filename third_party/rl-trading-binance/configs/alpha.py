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
# For Val 1500 | Test 3500
cfg.trainlog.num_val_ep = 3500
cfg.trainlog.val_freq = 1000
# gradient steps ~ 241_000 ~ episodes = 24_000
cfg.trainlog.episodes = 55_000
cfg.trainlog.plot_top_n = 10

cfg.per.buffer_size = 230_000

cfg.rl.batch_size = 16
cfg.rl.learning_rate = 1e-4
cfg.rl.train_start = 10_000

cfg.seq.agent_history_len = 30
cfg.seq.agent_session_len = 10
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 2
cfg.backtest.position_fraction = 0.5
# ["advantage_based_filter", "ensemble_q_filter"]
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.012695
cfg.backtest.short_action_threshold = 0.009902
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

