import asyncio
import sys
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Загрузка окружения
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("SupervisorService")

def get_sleep_interval():
    """Читает supervisor_interval_days из config.json и переводит в секунды"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            days = config.get("supervisor_interval_days", 0.125)
            # Возвращаем интервал в секундах
            return max(300, int(days * 86400)) # Минимум 5 минут, чтобы не спамить
    except Exception as e:
        logger.error(f"Error reading config for interval: {e}")
    return 3600

async def main():
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "supervisor.py")
    
    logger.info("Supervisor Service started (Async Mode).")
    
    while True:
        # Считываем актуальный интервал перед каждым циклом
        interval_sec = get_sleep_interval()
        
        logger.info(f"--- Running Supervisor Cycle at {datetime.now().strftime('%c')} ---")
        try:
            # Запускаем supervisor.py асинхронно и ждем завершения
            proc = await asyncio.create_subprocess_exec(
                python_exe, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Мы можем читать вывод в реальном времени или дождаться окончания
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info("Cycle completed successfully.")
                if stdout:
                    output = stdout.decode('utf-8', errors='replace').strip()
                    for line in output.split('\n'):
                        logger.info(f"SUPERVISOR: {line}")
            else:
                logger.error(f"Cycle failed with exit code {proc.returncode}")
                if stderr:
                    error_output = stderr.decode('utf-8', errors='replace').strip()
                    for line in error_output.split('\n'):
                        logger.error(f"SUPERVISOR ERROR: {line}")
                    
        except Exception as e:
            logger.error(f"Unexpected error during cycle: {e}")
            
        logger.info(f"Sleeping for {interval_sec / 3600:.1f} hours...")
        await asyncio.sleep(interval_sec)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Supervisor Service stopped by user.")
