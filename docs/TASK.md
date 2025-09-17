Патч (точечный, ≤ 10 строк)

Файл: tests/test_exchange_property.py
Идея: добавить импорт settings, HealthCheck и применить декоратор к конкретному тесту пост-фактум (это устойчиво, даже если над функцией уже стоят другие декораторы @given(...)).

*** Begin Patch
*** Update File: tests/test_exchange_property.py
@@
-from hypothesis import given, strategies as st
+from hypothesis import given, strategies as st
+from hypothesis import settings, HealthCheck
@@
 # (остальной код и тесты остаются без изменений)
+
+# --- Hypothesis health-check suppression for property test with function-scoped fixture ---
+# В данном тесте фикстура `exch` имеет scope="function", а Hypothesis генерирует несколько входов
+# без повторного вызова фикстуры между ними; это допустимо для нашего случая.
+test_create_futures_order_property = settings(
+    suppress_health_check=[HealthCheck.function_scoped_fixture]
+)(test_create_futures_order_property)
*** End Patch
