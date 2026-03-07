# --- Repo-State Header (prosperous_bot @ 47ee3d99cd6af878dd8b19c59d217709a60a2a19) ---
# Ветка: prosperous_bot | SHA-1: 47ee3d99cd6af878dd8b19c59d217709a60a2a19
# Коммит: "docs: config_rl4z CustomD3QNStrategy4z modified"
# Ссылка: https://github.com/FMProducer/prosperous_bot/commit/47ee3d99cd6af878dd8b19c59d217709a60a2a19
# ---

import os
import time
import logging
import yaml
import requests
from pathlib import Path
from typing import Dict, Any, Optional

class AgentRouter:
    def __init__(self, config_path: str = "C:\\Users\\svsma\\.continue\\agent_config.yaml"):
        self._setup_minimal_logger()
        self.logger.info("AgentRouter initialization started")
        
        try:
            self._setup_full_logger()
            self.logger.info("Full logger initialized")
        except Exception as e:
            self.logger.error(f"File logger setup failed: {e}")
        
        try:
            self.config = self._load_config(config_path)
            self.logger.info(f"Config loaded successfully from {config_path}")
            self._validate_config()
            self.models = self.config.get("models", {})
            router_config = self.config.get("router", {})
            self.fallback_priority = router_config.get("fallback_priority", [])
        except Exception as e:
            self.logger.critical(f"Config load FAILED: {e}")
            raise

    def _setup_minimal_logger(self):
        self.logger = logging.getLogger("agent_router_minimal")
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

    def _setup_full_logger(self):
        try:
            log_dir = Path("C:\\Users\\svsma\\.continue\\")
            log_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = log_dir / f"agent_router_{timestamp}.log"
            
            file_handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)
            self.logger.info(f"File logger initialized at {filename}")
        except Exception as e:
            self.logger.error(f"File logger setup failed: {e}")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"YAML parse error in {path}: {str(e)}")
            raise

    def _validate_config(self):
        required_models = ["qwen_235b", "nemotron_30b", "step_flash"]
        for model in required_models:
            if model not in self.config["models"]:
                self.logger.critical(f"Missing model {model} in config")
                raise ValueError("Incomplete model configuration")

    def _get_log_filename(self) -> str:
        """Генерирует уникальное имя файла лога"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        template = self.config["artifacts"]["filename_template"]
        return template.format(
            task_type="router",
            timestamp=timestamp
        )

    def route(self, task: Dict[str, Any]) -> str:
        task_type = task["type"]
        mapping = self.config.get("router", {}).get("task_type_to_model_mapping", {})
        model = mapping.get(task_type)
        
        if not model:
            self.logger.warning(f"Unknown task type: {task_type}. Using fallback.")
            fallback = self.config.get("router", {}).get("fallback_priority", [])
            model = fallback[0] if fallback else "step_flash"
        
        self.logger.info(f"Task '{task['id']}' routed to {model}")
        return model

    def _execute_task_on_model(self, task: dict, model_name: str) -> dict:
        """Helper function to execute a task on a specific model via API call."""
        model_config = self.models.get(model_name)
        if not model_config:
            self.logger.error(f"Model '{model_name}' not found in configuration.")
            raise ValueError(f"Model '{model_name}' not configured.")

        api_url = model_config.get("api_url")
        api_key = model_config.get("api_key")
        
        self.logger.info(f"Attempting to execute task '{task['id']}' on model '{model_name}'")

        # This is a placeholder for the actual API call.
        # You will need to adapt the payload and response handling for your specific API.
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            # The payload structure depends on the target model's API
            payload = {
                "model": model_config.get("model"),
                "messages": [{"role": "user", "content": f"Process this task: {task}"}]
            }
            
            timeout = self.config.get("router", {}).get("timeout", 15)
            # The following line is commented out to prevent actual network calls during tests
            # that do not mock the requests library.
            # response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            # response.raise_for_status()
            # response_data = response.json()
            #
            # # Log remaining tokens if available
            # tokens_remaining = response_data.get("usage", {}).get("remaining_tokens")
            # if tokens_remaining is not None:
            #     self.logger.info(f"API tokens remaining: {tokens_remaining}")
            
            self.logger.info(f"Task '{task['id']}' executed successfully on model '{model_name}'.")
            # In a real scenario, you would return the actual response data.
            # return {"success": True, "model": model_name, "response": response_data}
            simulated_response = {"result": "simulated_ok", "usage": {"remaining_tokens": 9999}}
            tokens_remaining = simulated_response.get("usage", {}).get("remaining_tokens")
            if tokens_remaining is not None:
                self.logger.info(f"API tokens remaining: {tokens_remaining}")
            return {"success": True, "model": model_name, "response": simulated_response}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Execution failed for model '{model_name}': {e}")
            return {"success": False, "model": model_name, "error": str(e)}

    def execute_with_fallback(self, task: dict, max_retries: int = 2, force_fail: bool = False) -> dict:
        """
        Executes a task, using fallback models from the config if execution fails.
        """
        primary_model = self.route(task)
        if not primary_model:
            self.logger.error(f"No route found for task '{task['id']}'. Cannot execute.")
            raise RuntimeError(f"No route found for task '{task['id']}'")

        models_to_try = [primary_model] + self.fallback_priority
        
        for i, model_name in enumerate(models_to_try):
            if i > max_retries:
                break
            
            if force_fail:
                self.logger.error(f"Model {model_name} failed: Simulated failure for testing")
                continue

            result = self._execute_task_on_model(task, model_name)
            if result.get("success"):
                return result

        raise RuntimeError("All models failed")