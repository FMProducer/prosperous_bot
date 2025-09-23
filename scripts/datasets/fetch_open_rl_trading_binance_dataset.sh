#!/usr/bin/env bash
set -euo pipefail
python - << 'PY'
from datasets import load_dataset
# Пример: тренировочный split (см. карточку HF)
ds = load_dataset("ResearchRL/open-rl-trading-binance-dataset", split="train_data")
print(ds)
PY
