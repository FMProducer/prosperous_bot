import os
import glob
import json
import subprocess
import time

def reset_system():
    print("--- FULL SYSTEM RESET INITIATED ---")
    
    # 1. Останавливаем все процессы через PM2
    print("Stopping and deleting all PM2 processes...")
    try:
        # Убиваем демон полностью, чтобы снять блокировки с файлов логов
        subprocess.run(["pm2", "kill"], check=False)
        time.sleep(2)
    except:
        pass

    # 2. Очищаем папку логов
    print("Clearing logs directory...")
    log_files = glob.glob("logs/*")
    for f in log_files:
        try:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                import shutil
                shutil.rmtree(f)
        except Exception as e:
            print(f"Could not remove log file {f}: {e}")

    # 3. Удаляем файлы состояний
    print("Removing state files...")
    state_files = glob.glob("state_*.json") + glob.glob("paper_state_*.json")
    for f in state_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Could not remove state file {f}: {e}")

    # 4. Обнуляем конфиг (ставим пустые тикеры)
    print("Resetting config.json...")
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            config["tickers"] = []
            config["base_ticker"] = "BTCUSDT"
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error resetting config: {e}")

    # 5. Удаляем историю графиков
    print("Removing equity history...")
    history_file = os.path.join("logs", "equity_history.csv")
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
        except:
            pass

    print("\n--- RESET COMPLETE ---")
    print("To start the system again, run:")
    print("pm2 start supervisor_service.py --name supervisor-service")
    print("pm2 save")

if __name__ == "__main__":
    reset_system()
