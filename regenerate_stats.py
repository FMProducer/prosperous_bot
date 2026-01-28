import json
import os
import numpy as np
from tqdm import tqdm
import logging

# --- Configuration ---
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Path to your training data
NPZ_FILE_PATH = r"third_party\rl-trading-binance\data\train_data_fair_8m.npz"

# Path where the output stats file will be saved
OUTPUT_STATS_PATH = r"third_party\rl-trading-binance\output\alpha_trend_mtf\norm_stats.json"

# Channel configuration (must match the model's training config)
USE_CHANNELS = [
    "open", "high", "volume_weighted_average", "low", "close", "volume", "num_trades"
]
PRICE_CHANNELS = [
    "open", "high", "volume_weighted_average", "low", "close"
]
VOLUME_CHANNELS = ["volume", "num_trades"]
OTHER_CHANNELS = []

# --- Logic (Corrected) ---

def calculate_normalization_stats(
    sequences: list[np.ndarray],
    use_channels: list[str],
    price_channels: list[str],
    volume_channels: list[str],
    other_channels: list[str],
) -> dict[str, dict[str, float]]:
    """
    Calculates normalization statistics based on the logic from the project's utils.py.
    """
    stats: dict[str, dict[str, float]] = {"means": {}, "stds": {}}
    if not sequences:
        logging.warning("Empty training set for normalization stats")
        return stats

    # This dictionary will hold the flattened, transformed data for each channel
    data_accum: dict[str, list[float]] = {ch: [] for ch in use_channels}

    logging.info("Step 1: Accumulating and transforming data from sequences...")
    for seq in tqdm(sequences, desc="Processing sequences", leave=False):
        for idx, ch in enumerate(use_channels):
            # Extract the column for the current channel
            channel_data = seq[:, idx].astype(np.float64)
            
            transformed_vals = None
            if ch in price_channels:
                # Log returns for price channels
                if len(channel_data) > 1:
                    changes = channel_data[1:] / (channel_data[:-1] + 1e-9)
                    transformed_vals = np.log(np.maximum(changes, 1e-9))
            elif ch in volume_channels:
                # Log transform for volume channels
                transformed_vals = np.log1p(channel_data)
            elif ch in other_channels:
                transformed_vals = channel_data
            else:
                continue
            
            if transformed_vals is not None:
                # Append only finite values to the accumulator
                finite_vals = transformed_vals[np.isfinite(transformed_vals)]
                data_accum[ch].extend(finite_vals.tolist())

    logging.info("Step 2: Calculating mean and standard deviation for each channel...")
    for ch, values in tqdm(data_accum.items(), desc="Calculating stats", leave=False):
        if not values:
            logging.warning(f"No data for stats on channel {ch}, defaulting to mean=0, std=1")
            stats["means"][ch] = 0.0
            stats["stds"][ch] = 1.0
        else:
            # Convert list to numpy array for calculation
            values_arr = np.array(values, dtype=np.float32)
            mean, std = float(values_arr.mean()), float(values_arr.std())
            
            # Prevent division by zero if standard deviation is too small
            if std < 1e-7:
                logging.debug(f"Std too small for {ch}, setting to 1.0")
                std = 1.0
            
            stats["means"][ch] = mean
            stats["stds"][ch] = std
            
    logging.info("Normalization statistics computed successfully.")
    return stats

def main():
    """
    Main function to load data, calculate stats, and save them.
    """
    logging.info(f"Loading dataset from {NPZ_FILE_PATH}")
    
    try:
        with np.load(NPZ_FILE_PATH, allow_pickle=True) as data:
            # Extract all arrays from the NPZ file, ignoring metadata keys
            sequences = [data[key] for key in data.files if not key.startswith('_')]
        logging.info(f"Loaded {len(sequences)} sequences from the dataset.")
    except FileNotFoundError:
        logging.error(f"FATAL: Training data not found at {NPZ_FILE_PATH}. Cannot proceed.")
        return
    except Exception as e:
        logging.error(f"FATAL: An error occurred while loading the NPZ file: {e}")
        return

    if not sequences:
        logging.error("FATAL: No data sequences found in the NPZ file.")
        return

    # Calculate the stats using the corrected logic
    final_stats = calculate_normalization_stats(
        sequences, USE_CHANNELS, PRICE_CHANNELS, VOLUME_CHANNELS, OTHER_CHANNELS
    )

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_STATS_PATH)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    # Save the stats to a JSON file
    with open(OUTPUT_STATS_PATH, 'w') as f:
        json.dump(final_stats, f, indent=4)
    
    logging.info(f"Successfully saved normalization stats to: {OUTPUT_STATS_PATH}")
    logging.info("You can now re-run the paper_trader.py script.")

if __name__ == "__main__":
    main()