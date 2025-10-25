
import optuna
import logging
import argparse
from pprint import pprint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[logging.StreamHandler()])

def show_trial(db_path: str, study_name: str, trial_number: int):
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        
        logging.info(f"Study '{study_name}' loaded successfully from {db_path}.")
        
        trial = study.get_trials(deepcopy=False)[trial_number]

        print(f"\n--- Trial #{trial.number} ---")
        print(f"  Values: {trial.values}")
        print("  Params:")
        pprint(trial.params)
        print("-" * 50)

    except Exception as e:
        logging.error(f"Failed to load or process the study: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a specific trial from an Optuna study.")
    parser.add_argument("db_path", type=str, help="Path to the Optuna sqlite database file.")
    parser.add_argument("study_name", type=str, help="The name of the study to load.")
    parser.add_argument("trial_number", type=int, help="The number of the trial to show.")
    args = parser.parse_args()
    
    show_trial(args.db_path, args.study_name, args.trial_number)
