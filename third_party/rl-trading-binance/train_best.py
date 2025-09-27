from config import MasterConfig

cfg = MasterConfig()

# --- Best parameters from Optuna ---
cfg.rl.learning_rate = 6.725065824995981e-05
cfg.rl.gamma = 0.9882016250031588
cfg.model.dropout_p = 0.14331986869518515
cfg.backtest.long_action_threshold = 0.021682168428365414
cfg.backtest.short_action_threshold = 0.018711084727234838
cfg.seq.agent_history_len = 30
cfg.rl.batch_size = 16

# --- Training settings ---
cfg.trainlog.episodes = 55_000  # Full training schedule
cfg.paths.config_name = "production_model_v1" # A descriptive name for the new model
