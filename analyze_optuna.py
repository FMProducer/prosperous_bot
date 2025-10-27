import optuna
import pandas as pd
import sys

# Redirect stderr to stdout to capture all output
sys.stderr = sys.stdout

study_db_path = "sqlite:///C:\\Python\\Prosperous_Bot\\output\\alpha\\optuna_papertrader_20251027_215635\\optuna.db"
study_name = "papertrade_opt_20251027_215635" # Found from previous run

try:
    study = optuna.load_study(study_name=study_name, storage=study_db_path)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials found in the study.")
    else:
        best_trials = study.best_trials[:10]

        print(f"Study name: {study.study_name}")
        # Handle multi-objective directions
        directions = [d.name for d in study.directions]
        print(f"Directions: {directions}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of completed trials: {len(completed_trials)}")
        print("\nTop 10 Pareto Front Trials:")

        data = []
        # Create column names for the multiple values
        value_cols = [f'value_{i}' for i in range(len(study.directions))]

        for trial in best_trials:
            row = {'number': trial.number}
            # Add each objective value to the row
            for i, value in enumerate(trial.values):
                row[f'value_{i}'] = value
            row.update(trial.params)
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Reorder columns to have number, values, then params
        param_cols = [c for c in df.columns if c.startswith('d')] # d0, d_min, etc.
        col_order = ['number'] + value_cols + sorted(param_cols)
        df = df[col_order]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))

except Exception as e:
    print(f"An error occurred: {e}")
