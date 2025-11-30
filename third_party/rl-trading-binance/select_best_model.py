#!/usr/bin/env python3
"""
Скрипт для пост-селекции лучшей модели из топ-K чекпоинтов
с применением строгих критериев отбора.

Usage:
    python select_best_model.py --checkpoint-dir output/alpha_seed_404_v7/saved_models/rl_binance_futures_trading_date_20251120_time_015257/checkpoints --min-trades 200
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def select_best_checkpoint(
    checkpoint_dir: str,
    min_trades: int = 200,
    min_sharpe: float = 0.5,
    min_profit_factor: float = 1.0,
    max_drawdown_threshold: float = -0.20,
    metric: str = "Validation_sortino"
) -> Optional[Path]:
    """
    Отфильтровать чекпоинты по строгим критериям и выбрать лучший. 
    
    Args:
        checkpoint_dir: Путь к директории с чекпоинтами
        min_trades: Минимальное количество сделок
        min_sharpe: Минимальный Sharpe Ratio
        min_profit_factor: Минимальный Profit Factor
        max_drawdown_threshold: Максимально допустимая просадка (отрицательное число)
        metric: Метрика для ранжирования
    
    Returns:
        Путь к лучшей модели или None
    """
    
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logging.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = []
    
    # Собрать все чекпоинты с метаданными
    for json_file in checkpoint_dir.glob("checkpoint_*.json"):
        pth_file = json_file.with_suffix('.pth')
        
        if not pth_file.exists():
            logging.warning(f"Missing .pth file for {json_file.name}, skipping")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {})
            checkpoints.append({
                'path': pth_file,
                'episode': data.get('episode', 0),
                'metrics': metrics
            })
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")
            continue
    
    if not checkpoints:
        logging.error("No valid checkpoints found in directory")
        return None
    
    logging.info(f"Found {len(checkpoints)} total checkpoints")
    
    # Применить строгий фильтр
    filtered = []
    
    for ckpt in checkpoints:
        m = ckpt['metrics']
        
        trades = m.get('Validation_trades', 0)
        sharpe = m.get('Validation_sharpe', -999)
        pf = m.get('Validation_profit_factor', 0)
        dd = m.get('Validation_max_drawdown', 0)
        
        # Строгие критерии
        passes = (
            trades >= min_trades and
            sharpe >= min_sharpe and
            pf >= min_profit_factor and
            dd >= max_drawdown_threshold
        )
        
        if passes:
            filtered.append(ckpt)
        else:
            logging.debug(
                f"Checkpoint rejected: ep={ckpt['episode']} "
                f"(trades={trades}, sharpe={sharpe:.3f}, pf={pf:.3f}, dd={dd:.2%})"
            )
    
    logging.info(f"Checkpoints passing strict filter: {len(filtered)}/{len(checkpoints)}")
    
    # Fallback: если нет моделей с min_trades >= 200, попробовать с >= 100
    if not filtered:
        logging.warning(f"No checkpoints with min_trades >= {min_trades}. Trying fallback >= 100...")
        
        for ckpt in checkpoints:
            m = ckpt['metrics']
            if (m.get('Validation_trades', 0) >= 100 and
                m.get('Validation_sharpe', -999) >= 0.0 and
                m.get('Validation_profit_factor', 0) >= 0.8):
                filtered.append(ckpt)
    
    if not filtered:
        logging.error("No valid checkpoints found even with fallback criteria!")
        return None
    
    # Сортировать по целевой метрике
    filtered.sort(key=lambda x: x['metrics'].get(metric, -999), reverse=True)
    
    best = filtered[0]
    
    # Вывести отчет
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL (из {len(filtered)} валидных / {len(checkpoints)} всего):")
    print("="*70)
    print(f"Файл: {best['path'].name}")
    print(f"Эпизод: {best['episode']}")
    print("\nМетрики:")
    
    key_metrics = [
        'Validation_sortino',
        'Validation_sharpe',
        'Validation_profit_factor',
        'Validation_max_drawdown',
        'Validation_trades',
        'Validation_win_rate'
    ]
    
    for key in key_metrics:
        val = best['metrics'].get(key, 'N/A')
        if isinstance(val, float):
            if 'rate' in key or 'drawdown' in key:
                print(f"  {key}: {val:.2%}" if val != 'N/A' else f"  {key}: {val}")
            else:
                print(f"  {key}: {val:.4f}" if val != 'N/A' else f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")
    
    print("="*70 + "\n")
    
    return best['path']


def main():
    parser = argparse.ArgumentParser(
        description="Select best model from top-K checkpoints with strict criteria"
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to checkpoints directory"
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=200,
        help="Minimum number of trades required (default: 200)"
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.5,
        help="Minimum Sharpe Ratio (default: 0.5)"
    )
    parser.add_argument(
        "--min-pf",
        type=float,
        default=1.0,
        help="Minimum Profit Factor (default: 1.0)"
    )
    parser.add_argument(
        "--max-dd",
        type=float,
        default=-0.20,
        help="Maximum drawdown threshold as negative decimal (default: -0.20 = -20%%)"
    )
    parser.add_argument(
        "--metric",
        default="Validation_sortino",
        help="Metric for ranking (default: Validation_sortino)"
    )
    parser.add_argument(
        "--copy-to-best",
        action="store_true",
        help="Copy selected checkpoint to best.pth in parent directory"
    )
    
    args = parser.parse_args()
    
    best_path = select_best_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        min_profit_factor=args.min_pf,
        max_drawdown_threshold=args.max_dd,
        metric=args.metric
    )
    
    if best_path:
        print(f"✅ Best model: {best_path}")
        
        if args.copy_to_best:
            parent_dir = best_path.parent.parent
            dest = parent_dir / "best.pth"
            shutil.copy2(best_path, dest)
            print(f"✅ Copied to: {dest}")
    else:
        print("❌ No suitable model found")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
