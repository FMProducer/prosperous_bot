
TL;DR: Исправления в целом корректны: знак «сужения» теперь верный, TSL симметрично инициализируется для LONG/SHORT, гистерезис по Δp работает. Один важный недочёт: **при отсутствии нового прироста прибыли Δp текущий код может ослаблять уже подтянутый TSL** (возвращая «базовый» трейл). Нужно сделать TSL **монотонным**: не отдалять стоп от цены, если Δp не достигнут. Мини-фикс — 4 строки в LONG и 4 в SHORT. 

---

## Что проверил и что нашёл

| Шаг                        | Наблюдение                                                                                                                                                                              | Где в файле                                                                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Знак «сужения»             | `d_eff = min(max(d0 - max(0, p - fee_buf), d_min), d0)` — правильно: при росте p ширина **уменьшается**                                                                                 | LONG/SHORT расчёт `d_eff` в risk-блоке.                                                                                        |
| Симметричная инициализация | Базовый TSL задаётся сразу: LONG `trailing_max*(1-d0)`, SHORT `trailing_min*(1+d0)`                                                                                                     | Ветви LONG/SHORT перед «расширенной» логикой.                                                                                  |
| Гистерезис                 | Обновление TSL только при `p > p_last + delta_p_hysteresis` — корректно                                                                                                                 | Проверки `p_at_last_tsl_update` в обеих ветках.                                                                                |
| **Монотонность TSL**       | **Проблема:** если Δp не достигнут, вы заново присваиваете «базовый» TSL и **затираете** более строгий `pos['tsl_price']` из прошлого шага → стоп может отдалиться от цены (ослабление) | LONG/SHORT: базовый `tsl_price` вычисляется и затем всегда пишется в `pos['tsl_price']`, даже если продвинутый не обновлялся.  |

---

## Мини-патч (монотонность TSL, без изменения логики Δp/формул)

Идея: брать «строже из двух» — **текущий базовый** против **ранее зафиксированного** `pos['tsl_price']`.

* Для **LONG** «строже» = **выше** (берём `max(prev, base)`),
* Для **SHORT** «строже» = **ниже** (берём `min(prev, base)`).

```diff
diff --git a/third_party/rl-trading-binance/paper_trader.py b/third_party/rl-trading-binance/paper_trader.py
--- a/third_party/rl-trading-binance/paper_trader.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@ -300,8 +300,11 @@
                 if pos["direction"] == "LONG":
                     pos["trailing_max_price"] = max(pos.get("trailing_max_price", current_price), current_price)
-                    # Always set a base TSL for symmetric activation
-                    tsl_price = pos["trailing_max_price"] * (1 - d0)
+                    # Always set a base TSL for symmetric activation, but keep it monotonic (never loosen)
+                    base_tsl = pos["trailing_max_price"] * (1 - d0)
+                    prev_tsl = pos.get("tsl_price")
+                    tsl_price = max(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl
@@ -317,7 +320,7 @@
-                            advanced_tsl_price = max(pos["entry_price"] * (1 + fee_buf), pos["trailing_max_price"] * (1 - d_eff))
-                            tsl_price = max(tsl_price, advanced_tsl_price)
+                            advanced_tsl_price = max(pos["entry_price"] * (1 + fee_buf), pos["trailing_max_price"] * (1 - d_eff))
+                            tsl_price = max(tsl_price, advanced_tsl_price)  # still never loosen
 
                 elif pos["direction"] == "SHORT":
                     pos["trailing_min_price"] = min(pos.get("trailing_min_price", current_price), current_price)
-                    # Always set a base TSL for symmetric activation
-                    tsl_price = pos["trailing_min_price"] * (1 + d0)
+                    # Always set a base TSL for symmetric activation, but keep it monotonic (never loosen)
+                    base_tsl = pos["trailing_min_price"] * (1 + d0)
+                    prev_tsl = pos.get("tsl_price")
+                    tsl_price = min(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl
@@ -336,7 +339,7 @@
-                            advanced_tsl_price = min(pos["entry_price"] * (1 - fee_buf), pos["trailing_min_price"] * (1 + d_eff))
-                            tsl_price = min(tsl_price, advanced_tsl_price)
+                            advanced_tsl_price = min(pos["entry_price"] * (1 - fee_buf), pos["trailing_min_price"] * (1 + d_eff))
+                            tsl_price = min(tsl_price, advanced_tsl_price)  # still never loosen
```

Почему это важно: теперь при «тишине» (Δp не накопилась) мы **сохраняем более строгий** ранее зафиксированный уровень трейла, а не возвращаемся к более широкому базовому. Это устраняет «ослабление» стопа и соответствует сути гистерезиса. Логика Δp, формулы `d_eff`, кап по BE и причины выхода **не меняются**. 

---

## Влияние на KPI и риски

| Шаг              | Действие                                   | KPI/риск                                                                                                      |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Монотонность TSL | Исключаем ослабление стопа между апдейтами | PF/Sharpe ↑, Win-Rate чаще ↑ в «пиле»; риск — немного раньше фиксация в тренде (ожидаемо для строгого трейла) |