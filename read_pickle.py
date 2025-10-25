
import pickle

file_path = "c:\\Python\\Prosperous_Bot\\output\\alpha\\optuna_cfg_optimization_results\\trial_caches\\trial_187\\qval_cache.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data)
