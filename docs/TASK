TL;DR

Падение в --probe вызвано Windows-локом при удалении временной папки в tearDownClass. Даю минимальный патч к тесту: «робастный» rmtree с ретраями и снятием read-only, чтобы очистка каталога не валяла прогоны на Windows. После применения — запустите bash setup_env.sh --strict (порог 90% останется).

*** Begin Patch
*** Update File: tests/test_rebalance_signal_handling.py
@@
-import shutil
+import shutil, os, stat, time, gc
@@
-class TestSignalHandling:
+def _robust_rmtree(path, tries=5, delay=0.2):
+    """Best-effort removal tolerant to Windows locks (WinError 32)."""
+    def _onerror(func, p, _exc_info):
+        # try to drop read-only and retry the failed op
+        try:
+            os.chmod(p, stat.S_IWRITE)
+            func(p)
+        except Exception:
+            pass
+    for _ in range(tries):
+        try:
+            shutil.rmtree(path, onerror=_onerror)
+            return
+        except PermissionError:
+            gc.collect()
+            time.sleep(delay)
+    # final attempt
+    shutil.rmtree(path, onerror=_onerror)
+
+class TestSignalHandling:
@@
     @classmethod
     def tearDownClass(cls):
-        if cls.test_data_dir.exists():
-            shutil.rmtree(cls.test_data_dir)
+        if getattr(cls, "test_data_dir", None) and cls.test_data_dir.exists():
+            _robust_rmtree(cls.test_data_dir)
*** End Patch
