import optuna
import logging
import argparse
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[logging.StreamHandler()])

def save_all_trials(db_path: str, study_name: str):
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        
        logging.info(f"Study '{study_name}' loaded successfully from {db_path}.")
        
        df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))

        # Save to files
        base_dir = Path(db_path).parent
        csv_path = base_dir / "all_trials.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Warning: failed to save outputs: {e}", file=sys.stderr)

        print(f"\nSaved all trials to: {csv_path}")

    except Exception as e:
        logging.error(f"Failed to load or process the study: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save all trials from an Optuna study to a CSV file.")
    parser.add_argument("db_path", type=str, help="Path to the Optuna sqlite database file.")
    parser.add_argument("study_name", type=str, help="The name of the study to load.")
    args = parser.parse_args()
    
    save_all_trials(args.db_path, args.study_name)
