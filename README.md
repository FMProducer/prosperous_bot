# RL Trading System for Binance Futures

High-performance Reinforcement Learning trading system integrated with Freqtrade.

## 🏗 Architecture

### Core Components
- **Framework**: Freqtrade (Execution Engine) + PyTorch (RL Models).
- **Strategy**: `CustomD3QNStrategy4z.py` - A sophisticated ensemble strategy.
- **Agent**: `D3QN_PER_Agent` (Dueling Double DQN with Prioritized Experience Replay).

### 🧠 Ensemble Logic (2+2)
The system employs an ensemble of **4 independent neural networks**:
- **2 Long-only Models**: Trained specifically for long entries.
- **2 Short-only Models**: Trained for short entries (some using Mirror Mode).
- **Inference**: Parallel execution using `ThreadPoolExecutor` for low latency.
- **Voting System**:
  - Models cast votes if their normalized Q-Advantage exceeds a dynamic threshold.
  - **Veto Mechanism**: A strong signal from the opposite side blocks entry (e.g., Short models can veto a Long entry).

## 🛡️ Risk Management & Dynamic Features

### 1. Dynamic Epsilon (Adaptive Thresholds)
The voting threshold (`epsilon`) is not static. It adapts to market conditions and portfolio performance:
- **Defensive Mode**: If the portfolio experiences drawdown, the threshold increases, requiring higher model confidence to trade.
- **Aggressive Mode**: During profitable periods, thresholds remain at base levels.
- **Logic**: `_update_dynamic_epsilon` monitors Unrealized PnL.

### 2. Dynamic Slot Allocation
The bot dynamically reallocates `max_open_trades` between Long and Short sides based on recent performance:
- If Longs are profitable and Shorts are losing, the system allocates more slots to Longs.
- **Logic**: `_update_slot_allocation` calculates ratios based on PnL.

### 3. Market Regime Filter
- Uses **Supertrend (15m)** to determine the global market trend.
- **Bullish Regime**: Only Long signals are processed.
- **Bearish Regime**: Only Short signals are processed.

### 4. Q-Value Normalization
- Raw Q-values are normalized to a `[0, 1]` scale using auto-tuned `q_min` and `q_max` percentiles.
- This ensures consistent voting behavior across different model training epochs.

## 📊 Data & Features
- **Timeframe**: 1m (Execution), 15m (Informative).
- **Input Features**: 10 channels (Open, High, Low, Close, Volume, QuoteVolume, Trades, TakerBase, TakerQuote, VWAP).
- **Preprocessing**: Z-score normalization (window=90).

## 🚀 Usage
Run with Freqtrade:
```bash
freqtrade trade --config user_data/config_rl4z.json --strategy CustomD3QNStrategy4z
```
