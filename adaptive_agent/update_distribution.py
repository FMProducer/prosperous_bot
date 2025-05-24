
import json
import logging

logger = logging.getLogger(__name__)

def update_distribution(config_signal, signal):
    distribution = config_signal["asset_distributions"][signal]

    new_config = {
        "target_weights": {
            symbol: weight for symbol, weight in distribution.items()
            if symbol.endswith("USDT")
        }
    }

    with open("config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    logger.info("Updated target_weights in config.json for signal: %s", signal)
