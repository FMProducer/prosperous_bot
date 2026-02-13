import json
from pathlib import Path

# Пути к файлам
base_dir = Path(__file__).parent.resolve()
pairs_file = base_dir / "pairs_fast.txt"
config_file = base_dir / "config_rl4z_backtest.json"

if pairs_file.exists() and config_file.exists():
    # 1. Читаем пары из txt
    with open(pairs_file, "r", encoding="utf-8") as f:
        # Фильтруем пустые строки и комментарии
        pairs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    
    # 2. Читаем конфиг
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 3. Обновляем whitelist
    config["exchange"]["pair_whitelist"] = pairs
    
    # 4. Сохраняем обратно
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ Успешно! В конфиг {config_file.name} записано {len(pairs)} пар.")
else:
    print(f"❌ Ошибка: Файлы не найдены.\nПроверьте пути:\n{pairs_file}\n{config_file}")
