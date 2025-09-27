import copy
from config import cfg as default_cfg

# Create a deep copy to avoid modifying the original default_cfg
cfg = copy.deepcopy(default_cfg)

# Point to the temporary directory where the FMProducer 1 model is located
cfg.paths.config_name = "fmproducer_1_eval"
