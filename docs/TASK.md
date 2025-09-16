*** Begin Patch
*** Update File: tests/test_rebalance_signal_handling.py
@@
-import shutil
+import shutil
+import os
+import stat
+import time
+import gc
+
+# Best-effort removal tolerant to Windows locks (WinError 32).
+# При ошибках удаления пробуем снять read-only и повторяем несколько раз.
+def _robust_rmtree(path, tries: int = 5, delay: float = 0.2) -> None:
+    def _onerror(func, p, _exc_info):
+        try:
+            os.chmod(p, stat.S_IWRITE)
+            func(p)
+        except Exception:
+            # Игнорируем вторичную ошибку — повторим удаление выше
+            pass
+    for _ in range(tries):
+        try:
+            shutil.rmtree(path, onerror=_onerror)
+            return
+        except PermissionError:
+            # Сборка мусора помогает освободить открытые дескрипторы на Windows
+            gc.collect()
+            time.sleep(delay)
+    # Финальная попытка — если всё ещё занято, пусть бросит исключение
+    shutil.rmtree(path, onerror=_onerror)
@@
 class TestSignalHandling:
@@
     @classmethod
     def tearDownClass(cls):
-        if cls.test_data_dir.exists():
-            shutil.rmtree(cls.test_data_dir)
+        if getattr(cls, "test_data_dir", None) and cls.test_data_dir.exists():
+            _robust_rmtree(cls.test_data_dir)
*** End Patch
