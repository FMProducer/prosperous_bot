--- third_party/rl-trading-binance/backtest_continuous.py
+++ third_party/rl-trading-binance/backtest_continuous.py
@@ -441,7 +441,36 @@
                     pass_adv = get_pass_advantage(action, confidence, cfg)
                     if pass_adv:
                         action = 0
-                else:
+                
+                elif cfg.backtest.selection_strategy == "ensemble_q_filter":
+                    q_mean, q_std = agent.predict_ensemble(
+                        state=obs,
+                        training=False,
+                        use_cache=cfg.backtest.use_cache,
+                        cache_key=cache_key,
+                        n_samples=cfg.backtest.ensemble_n_samples,
+                    )
+                    advantage = q_mean - q_mean[0]
+                    action = int(np.argmax(advantage))
+                    confidence = float(advantage[action])
+                    uncertainty = float(q_std[action])
+
+                    pass_adv = get_pass_advantage(action, confidence, cfg)
+                    pass_uncertainty = uncertainty >= cfg.backtest.ensemble_max_sigma
+                    if pass_adv and pass_uncertainty:
+                        # threshold per action (for logging)
+                        thr_val = (
+                            cfg.backtest.long_action_threshold if action == 1
+                            else cfg.backtest.short_action_threshold if action == 2
+                            else cfg.backtest.close_action_threshold
+                        )
+                        logging.info(
+                            f": REJECTED {['LONG', 'SHORT', 'CLOSE'][action-1]}, "
+                            f"confidence={confidence:.3f} < threshold={thr_val:.3f}, "
+                            f"uncertainty={uncertainty:.3f} > max_sigma_threshold={cfg.backtest.ensemble_max_sigma}"
+                        )
+                        action = 0
+                else:
                     action = agent.select_action(
                         state=obs,
                         training=False,
