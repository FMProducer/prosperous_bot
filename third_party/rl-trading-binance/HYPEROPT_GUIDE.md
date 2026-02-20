# 🚀 Hyperopt Guide for RL-Trading-Binance

This guide explains how to optimize the **ensemble parameters** of `CustomD3QNStrategy4z` using Freqtrade's Hyperopt.

**Important:** We are NOT optimizing the Neural Network weights here. We are optimizing the **voting logic**, **filters**, and **risk management** parameters that sit *on top* of the pre-trained RL models.

## 1. Preparation

### 1.1. Check Strategy Parameters
Ensure the parameters you want to optimize are set to `optimize=True` in `user_data/strategies/CustomD3QNStrategy4z.py`.

**Key Targets for Optimization:**
- **Voting Thresholds:** `rl_long_threshold_opt`, `rl_short_threshold_opt` (How many models must agree).
- **Regime Filter:** `supertrend_period`, `supertrend_multiplier` (Global trend definition).
- **Dynamic Epsilon:** `dd_aggression_k` (Sensitivity to drawdown).
- **Liquidity:** `min_quote_volume_usd`.

**Example in Code:**
```python
rl_long_threshold_opt = IntParameter(1, 2, default=1, space='buy', optimize=True, load=True)
```

### 1.2. Configuration
Ensure `user_data/config_rl4z.json` is ready.
- **Runmode:** Hyperopt overrides `runmode`, but ensure paths to models (`long_1_model_dir`, etc.) are correct in the strategy.
- **Pairs:** Use a representative list of pairs in `pair_whitelist`.

## 2. Running Hyperopt

### 2.1. Standard Command
Run hyperopt with a focus on **Sortino Ratio** (best for risk-adjusted returns) or **Sharpe**.

```bash
freqtrade hyperopt --config user_data/config_rl4z.json \
                   --strategy CustomD3QNStrategy4z \
                   --hyperopt-loss SortinoHyperOptLoss \
                   --spaces buy sell \
                   --epochs 100 \
                   --timerange 20240101-
```

**Flags:**
- `--spaces buy sell`: Optimizes parameters in `buy` and `sell` spaces (where our params are defined).
- `--hyperopt-loss`: `SortinoHyperOptLoss` is recommended for RL strategies to penalize downside volatility.
- `--epochs`: Start with 100-500 to test speed.

### 2.2. Performance & Caching
The strategy uses `self.q_value_cache` to cache RL inference results based on `(pair, date)`.
- **First Epoch:** Will be slow (computing Q-values via PyTorch for all candles).
- **Subsequent Epochs:** Will be **extremely fast** (using RAM cache).
- **Memory Warning:** The cache stores `(Batch, Actions)` floats for every candle. If running on many pairs/long timeframe, ensure you have enough RAM (16GB+ recommended).

## 3. Analysis & Application

### 3.1. Interpret Results
Look for a balance between:
- **High Win Rate:** > 55%
- **Profit Factor:** > 1.5
- **Drawdown:** < 20%

### 3.2. Update Strategy
Once the best parameters are found, Freqtrade will output a json block.
1. Copy the `params` block.
2. Paste it into the `user_data/strategies/CustomD3QNStrategy4z.json` file (if using separate storage) OR update the `default` values in the python file directly.

**Example Update:**
```python
# Before
dd_aggression_k = DecimalParameter(0.1, 2.0, default=0.55, ...)

# After (if best was 1.2)
dd_aggression_k = DecimalParameter(0.1, 2.0, default=1.2, ...)
```

## 4. Troubleshooting

- **"High Memory Usage"**: Reduce `timerange` or number of pairs. The Q-value cache is aggressive.
- **"No Trades"**: Check if `min_quote_volume_usd` range is too high in the parameter definition.
- **"Objective not improving"**: Try a different Loss Function (e.g., `SharpeHyperOptLossDaily` or `CalmarHyperOptLoss`).

## 5. Agent Checklist
1. Verify `optimize=True` in strategy code for target parameters.
2. Construct command with correct config path (`user_data/config_rl4z.json`).
3. Warn user about RAM usage due to caching mechanism.
4. Suggest `SortinoHyperOptLoss` as the default objective.