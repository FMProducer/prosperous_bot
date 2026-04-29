import subprocess
import time
import sys
import os
import json
from dotenv import load_dotenv

# Загрузка окружения
load_dotenv()

def get_sleep_interval():
    """Читает supervisor_interval_days из config.json и переводит в секунды"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        days = config.get("supervisor_interval_days", 0.125)
        # Возвращаем интервал в секундах
        return max(300, int(days * 86400)) # Минимум 5 минут, чтобы не спамить
    except Exception as e:
        print(f"Error reading config for interval: {e}")
        return 3600

def main():
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "supervisor.py")
    
    print(f"Supervisor Service started.")
    
    while True:
        # Считываем актуальный интервал перед каждым циклом ожидания
        interval_sec = get_sleep_interval()
        
        print(f"--- Running Supervisor Cycle at {time.ctime()} ---")
        try:
            # Запускаем supervisor.py и ждем завершения
            subprocess.run([python_exe, script_path], check=True)
            print(f"Cycle completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cycle failed with error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        print(f"Sleeping for {interval_sec / 3600:.1f} hours...")
        time.sleep(interval_sec)

if __name__ == "__main__":
    main()
