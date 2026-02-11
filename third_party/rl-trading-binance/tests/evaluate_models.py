import json
import subprocess
import sys
from pathlib import Path
import copy
import re

def parse_metrics(output):
    metrics = {'profit_pct': 0.0, 'profit_factor': 0.0, 'trades': 0}
    
    # Удаляем цветовые коды (ANSI escape sequences)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    # 1. Total Profit %
    # │ Total profit %                │ -13.46%                        │
    m_profit = re.search(r"[│|]\s*Total profit %\s*[│|]\s*([-\d\.]+)%", clean_output)
    if m_profit:
        metrics['profit_pct'] = float(m_profit.group(1))
        
    # 2. Profit Factor
    # │ Profit factor                 │ 0.46                           │
    m_pf = re.search(r"[│|]\s*Profit factor\s*[│|]\s*([-\d\.]+)", clean_output)
    if m_pf:
        metrics['profit_factor'] = float(m_pf.group(1))
        
    # 3. Trades
    # │ Total/Daily Avg Trades        │ 6323 / 451.64                  │
    m_trades = re.search(r"[│|]\s*Total/Daily Avg Trades\s*[│|]\s*(\d+)", clean_output)
    if m_trades:
        metrics['trades'] = int(m_trades.group(1))
        
    return metrics

def main():
    # Определяем корень проекта
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # Путь к базовому конфигу (используем backtest конфиг, так как он уже настроен)
    config_src = project_root / "user_data" / "config_rl4z_backtest.json"
    
    if not config_src.exists():
        print(f"❌ Config not found: {config_src}")
        return

    print(f"📂 Loading base config: {config_src}")
    with open(config_src, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # Список моделей для проверки
    models = ["long_1", "long_2", "short_1", "short_2"]
    
    # Обработка аргументов: перехватываем --pairs-file
    raw_args = sys.argv[1:]
    freqtrade_args = []
    pairs_from_file = []
    
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg == "--pairs-file":
            if i + 1 < len(raw_args):
                pairs_file_path = Path(raw_args[i+1])
                if pairs_file_path.exists():
                    print(f"📂 Reading pairs from {pairs_file_path}")
                    with open(pairs_file_path, 'r', encoding='utf-8') as f:
                        pairs_from_file = [line.strip() for line in f if line.strip()]
                else:
                    print(f"⚠️ Pairs file not found: {pairs_file_path}")
                i += 2
            else:
                print("⚠️ --pairs-file provided without a file path")
                i += 1
        else:
            freqtrade_args.append(arg)
            i += 1
    
    print(f"⏱️  Freqtrade args: {' '.join(freqtrade_args)}")

    all_results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"🧪 EVALUATING MODEL: {model.upper()}")
        print(f"{'='*60}")
        
        # Создаем временный конфиг
        temp_config = copy.deepcopy(base_config)
        
        # Внедряем пары из файла, если есть
        if pairs_from_file:
            if 'exchange' not in temp_config:
                temp_config['exchange'] = {}
            temp_config['exchange']['pair_whitelist'] = pairs_from_file
            # Ensure StaticPairList is used
            temp_config['pairlists'] = [{"method": "StaticPairList", "allow_inactive": True}]

        # 1. Отключаем все модели
        temp_config['rl_enable_long_1'] = False
        temp_config['rl_enable_long_2'] = False
        temp_config['rl_enable_short_1'] = False
        temp_config['rl_enable_short_2'] = False
        
        # 2. Включаем только текущую
        temp_config[f'rl_enable_{model}'] = True
        
        # 3. Снижаем порог голосования до 1 (так как голосует только одна модель)
        temp_config['rl_long_threshold'] = 1
        temp_config['rl_short_threshold'] = 1
        
        # 4. Отключаем вето (нет смысла при одной модели)
        temp_config['rl_enable_veto'] = False
        
        # Сохраняем временный конфиг
        temp_config_path = project_root / f"user_data/config_eval_{model}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(temp_config, f, indent=4)
            
        # Формируем команду запуска
        cmd = [
            "freqtrade", "backtesting",
            "--config", str(temp_config_path),
            "--strategy", "CustomD3QNStrategy4z",
            "--cache", "day" # Используем кэш для скорости
        ] + freqtrade_args
        
        try:
            # Запускаем freqtrade (вывод пойдет в консоль)
            # capture_output=True позволяет перехватить вывод для парсинга
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace'
            )
            print(result.stdout) # Выводим лог пользователю
            
            # Парсим метрики
            metrics = parse_metrics(result.stdout)
            all_results[model] = metrics
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating {model}")
            print(e.stdout)
            print(e.stderr)
        finally:
            # Удаляем временный конфиг
            if temp_config_path.exists():
                temp_config_path.unlink()

    # Вывод итоговой таблицы
    print("\n" + "="*65)
    print(f"{'MODEL':<15} | {'PROFIT %':<10} | {'PF':<6} | {'TRADES':<8}")
    print("-" * 65)
    for model in models:
        if model in all_results:
            m = all_results[model]
            print(f"{model:<15} | {m['profit_pct']:>9.2f}% | {m['profit_factor']:>6.2f} | {m['trades']:>8}")
        else:
            print(f"{model:<15} | {'N/A':>10} | {'N/A':>6} | {'N/A':>8}")
    print("="*65)

if __name__ == "__main__":
    main()
