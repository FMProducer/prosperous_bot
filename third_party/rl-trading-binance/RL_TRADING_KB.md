# RL Trading Bot Knowledge Base

## Architecture
- **Main Strategy**: CustomD3QNStrategy4z.py (Freqtrade IStrategy)
- **Agent**: D3QN_PER_Agent (agent.py, PyTorch Dueling DQN + PER)
- **Ensemble**: 2 LONG + 2 SHORT models
- **Key Methods**: populate_entry_trend → _parallel_inference → _compute_ensemble_decision

## Decision Logic
- **Input**: 1-minute OHLCV + Technical Features (Supertrend, etc.)
- **Output**: Action probabilities (LONG/SHORT/NEUTRAL)
- **Ensemble Logic**:
  - LONG models: 100% weight if probability > 0.5
  - SHORT models: 100% weight if probability > 0.5
  - Final Decision: LONG if (LONG_VOTE + SHORT_VOTE) > 0, else NEUTRAL

## Key Files
- `CustomD3QNStrategy4z.py`: Strategy logic and feature engineering
- `agent.py`: D3QN + PER agent implementation
- `config.json`: Hyperparameters and configuration
- `requirements.txt`: Dependencies

## Training
- Use `freqtrade hyperopt` with custom hyperopt_loss
- Train on historical data with appropriate timeframes
- Validate with walk-forward analysis

## Deployment
- Run with `freqtrade trade --config config.json --strategy CustomD3QNStrategy4z`
- Monitor performance and adjust hyperparameters as needed