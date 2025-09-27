
import optuna
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- Load the Optuna study ---
study_name = "rl_trading_optimization"
storage_name = f"sqlite:///{study_name}.db"

try:
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
    )

    # --- Print the results ---
    logging.info(f"--- Results for study '{study_name}' ---")
    logging.info(f"Number of finished trials: {len(study.trials)}")

    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info(f"  Value (Sharpe): {best_trial.value}")

    logging.info("  Best Parameters: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Print top 5 trials
    logging.info("\n--- Top 5 Trials (sorted by Sharpe) ---")
    best_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    best_trials.sort(key=lambda t: t.value, reverse=True)

    for i, trial in enumerate(best_trials[:5]):
        logging.info(f"\nRank {i+1}:")
        logging.info(f"  Value (Sharpe): {trial.value}")
        logging.info("  Parameters:")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

except Exception as e:
    logging.error(f"Could not load study '{study_name}' from {storage_name}. Error: {e}")
    logging.error("Please ensure you have run 'optimize.py' and the database file 'rl_trading_optimization.db' exists.")
    sys.exit(1)
