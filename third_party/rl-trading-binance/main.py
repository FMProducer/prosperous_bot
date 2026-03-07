# main.py
import argparse
import logging
import os
import time
from agent_router import AgentRouter

def main():
    """
    CLI для использования AgentRouter.
    """
    parser = argparse.ArgumentParser(description="Запуск задач через AgentRouter")
    parser.add_argument("--prompt", type=str, required=True, help="Текст задачи")
    parser.add_argument("--type", type=str, default="code_writing", help="Тип задачи (code_writing, design, complex_math, etc.)")
    parser.add_argument("--output", type=str, help="Путь к файлу для сохранения ответа")
    args = parser.parse_args()

    try:
        # 1. Инициализируем роутер.
        router = AgentRouter()
        logging.info("AgentRouter успешно инициализирован.")

        # 2. Определяем задачу из аргументов.
        task = {
            "id": f"task_{int(time.time())}",
            "type": args.type,
            "prompt": args.prompt
        }
        logging.info(f"Выполнение задачи: {task['id']} ({task['type']})")

        # 3. Выполняем задачу.
        result = router.execute_with_fallback(task)

        logging.info(f"Выполнение задачи завершено.")
        
        if result.get("success"):
            response_data = result.get("response", {})
            # Попытка извлечь текст ответа (формат OpenAI/OpenRouter)
            try:
                content = response_data["choices"][0]["message"]["content"]
                print("\n" + "="*40)
                print("ОТВЕТ МОДЕЛИ:")
                print("="*40)
                print(content)
                print("="*40 + "\n")

                # Сохранение в файл, если указан
                if args.output:
                    output_path = os.path.abspath(args.output)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logging.info(f"Ответ сохранен в файл: {output_path}")
            except (KeyError, IndexError, TypeError):
                logging.info(f"Результат (raw): {result}")
        else:
            logging.error(f"Ошибка выполнения: {result}")

    except Exception as e:
        logging.critical(f"В основном приложении произошла ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()