---
description: Проверка путей и загрузки моделей
---

Review the model loading logic in `CustomD3QNStrategy4z.py`.
1. Check how `project_root` is determined and if it handles different execution contexts (running from root vs user_data).
2. Verify the paths for `long_1_model_dir`, `long_2_model_dir`, etc.
3. Analyze `_find_config_file` and `_load_py_config` for potential import errors or path resolution issues.