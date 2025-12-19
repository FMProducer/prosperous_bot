import os
import zipfile
import pathlib
import logging

# Настройки
PROJECT_PATH = pathlib.Path("C:/Python/Prosperous_Bot/third_party/rl-trading-binance")
ARCHIVE_NAME = pathlib.Path("C:/Python/Prosperous_Bot/rl-trading-binance-clean.zip")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_archive(project_path: pathlib.Path, archive_name: pathlib.Path):
    """
    Создает ZIP-архив проекта, исключая ненужные каталоги и файлы.
    """
    logging.info(f"Начинаем архивацию проекта из: {project_path}")
    logging.info(f"Архив будет создан: {archive_name}")

    excluded_dirs = ['.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'data', 'datasets', 'logs', 'checkpoints', 'tensorboard', 'wandb']
    included_extensions = ['.py', '.json', '.yaml', '.yml', '.toml', '.ini', '.md', '.txt', '.sh', '.bat']

    total_files = 0
    total_size = 0

    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_path):
            root_path = pathlib.Path(root)

            # Исключаем каталоги
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            # Пропускаем, если текущий каталог находится в исключенных
            if any(part in excluded_dirs for part in root_path.parts):
                continue

            for file in files:
                file_path = root_path / file  # Initialize file_path here
                if any(file.endswith(ext) for ext in included_extensions):
                    file_path = root_path / file
                    archive_path = file_path.relative_to(project_path)
                    zipf.write(file_path, archive_path)

                    total_files += 1
                    total_size += os.path.getsize(file_path)
                else:
                    logging.debug(f"Пропущен файл (расширение): {file_path}")

    logging.info(f"Добавлено файлов: {total_files}")
    logging.info(f"Итоговый размер архива: {total_size / (1024 * 1024):.2f} MB")
    logging.info("Архивация завершена.")

if __name__ == "__main__":
    create_archive(PROJECT_PATH, ARCHIVE_NAME)
