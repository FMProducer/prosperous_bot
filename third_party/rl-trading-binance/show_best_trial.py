import optuna
import logging
import copy
from config import cfg as default_cfg
from pprint import pprint

# --- Configuration ---
STUDY_NAME = "rl_trading_optimization"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[logging.StreamHandler()])

def show_best_trial_config(study_name: str, storage_name: str):
    """
    Loads an Optuna study, finds the best trial, and prints its complete configuration.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_trial = study.best_trial
        
        logging.info("-" * 50)
        logging.info(f"Study '{study_name}' loaded successfully.")
        logging.info(f"Best trial number: {best_trial.number}")
        logging.info(f"Best trial value (Sharpe): {best_trial.value:.4f}")
        logging.info("-" * 50)

        # Create a config object with the best parameters
        cfg = copy.deepcopy(default_cfg)
        for key, value in best_trial.params.items():
            # This is a bit of a hack to set nested attributes
            # e.g., key = "learning_rate" -> cfg.rl.learning_rate = value
            # e.g., key = "dropout_p" -> cfg.model.dropout_p = value
            if hasattr(cfg.rl, key):
                setattr(cfg.rl, key, value)
            elif hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
            elif hasattr(cfg.seq, key):
                setattr(cfg.seq, key, value)
            elif hasattr(cfg.backtest, key):
                setattr(cfg.backtest, key, value)

        logging.info("Complete configuration for the best trial:")
        
        print("\n--- RLConfig ---")
        pprint(cfg.rl.model_dump())
        
        print("\n--- ModelConfig ---")
        pprint(cfg.model.model_dump())
        
        print("\n--- SequenceConfig ---")
        pprint(cfg.seq.model_dump())
        
        print("\n--- MarketConfig ---")
        pprint(cfg.market.model_dump())
        
        print("\n--- PERConfig ---")
        pprint(cfg.per.model_dump())
        
        print("\n--- EpsilonConfig ---")
        pprint(cfg.eps.model_dump())

        print("\n" + "-" * 50)

    except Exception as e:
        logging.error(f"Failed to load or process the study: {e}", exc_info=True)

if __name__ == "__main__":
    show_best_trial_config(STUDY_NAME, STORAGE_NAME)
