import subprocess
import time
import sys
import os

def main():
    # Путь к интерпретатору python и скрипту супервизора
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "supervisor.py")
    
    print(f"Supervisor Service started. Cycle: 4 hours.")
    
    while True:
        print(f"--- Running Supervisor Cycle at {time.ctime()} ---")
        try:
            # Запускаем supervisor.py и ждем завершения
            subprocess.run([python_exe, script_path], check=True)
            print(f"Cycle completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cycle failed with error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        # Ждем 4 часа (14400 секунд)
        print(f"Sleeping for 4 hours...")
        time.sleep(14400)

if __name__ == "__main__":
    main()
