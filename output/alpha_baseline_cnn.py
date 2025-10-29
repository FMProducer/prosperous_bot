import random

from config import MasterConfig, cfg
from find_best_matching_cnn_configs import find_matching_cnn_configs

num_target_params = 256_000
input_channels = len(cfg.data.data_channels)
max_layers = len(cfg.model.cnn_maps)
top_nearest_configs = 5
top_nearests = find_matching_cnn_configs(
    num_target_params, input_channels, max_layers, top_nearest_configs=top_nearest_configs
)

random_top_n = random.choice(top_nearests)
cfg = MasterConfig(
    paths={"config_name": "alpha_baseline_cnn"},
    trainlog={"iterations": 50_000, "val_freq": 1000},
    model={"cnn_maps": random_top_n[0], "cnn_kernels": random_top_n[1]},
)


# python baseline_cnn_classifier.py configs/alpha_baseline_cnn.py
# rm -r output/alpha_baseline_cnn
