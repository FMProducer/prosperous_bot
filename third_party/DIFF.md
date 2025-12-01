Ниже — приоритетные патчи в формате diff, отсортированы от самых важных к менее критичным.

***

## 1) Исправить сравнение Max Drawdown в validation gate

Цель: гарантировать, что более глубокая (более отрицательная) просадка считается хуже, и порог читается однозначно.[1][2]

```diff
- def _passes_gate(metrics: Dict[str, Any], gate) -> bool:
-     maxdd = metrics.get("Validationmaxdrawdown")
-     if gate.maxdrawdownatmost is not None and maxdd > gate.maxdrawdownatmost:
-         # ❌ это условие неверно для отрицательных MaxDD
-         return False
+ def _passes_gate(metrics: Dict[str, Any], gate) -> bool:
+     maxdd = metrics.get("Validationmaxdrawdown")
+     # maxdd и maxdrawdownatmost – отрицательные числа, например -0.20 для -20%.
+     # Отбрасываем модель, если фактическая просадка более отрицательна, чем порог.
+     if gate.maxdrawdownatmost is not None and maxdd < gate.maxdrawdownatmost:
+         return False
```

(Название функции возьмёшь из своего кода — патч иллюстративный; главное — заменить `>` на `<` и добавить комментарий.)

***

## 2) Сделать seed-детерминизм PyTorch управляемым через cfg.deterministic

Цель: настоящая воспроизводимость на GPU/CPU.[3][1]

В utils.py (функция установки сида):

```diff
-def set_random_seed(seed: int) -> None:
-    random.seed(seed)
-    np.random.seed(seed)
-    torch.manual_seed(seed)
-    if torch.cuda.is_available():
-        torch.cuda.manual_seed_all(seed)
-    torch.backends.cudnn.benchmark = True
+def set_random_seed(seed: int, deterministic: bool = False) -> None:
+    random.seed(seed)
+    np.random.seed(seed)
+    torch.manual_seed(seed)
+    if torch.cuda.is_available():
+        torch.cuda.manual_seed_all(seed)
+
+    if deterministic:
+        # Строгая детерминированность PyTorch
+        torch.backends.cudnn.benchmark = False
+        try:
+            torch.use_deterministic_algorithms(True, warn_only=True)
+        except Exception:
+            # Совместимость со старыми версиями torch
+            pass
+    else:
+        torch.backends.cudnn.benchmark = True
```

В train.py (там, где вызывается setrandomseed):[2][1]

```diff
-    setrandomseed(cfg.randomseed)
+    setrandomseed(cfg.randomseed, getattr(cfg, "deterministic", False))
```

***

## 3) Очистить load_and_prep_data от импорта cfg и сделать эпизодную выборку явной в train.py

Цель: убрать скрытую зависимость от глобального cfg и жёсткий seed внутри функции, плюс централизовать управление sampling.[1][2][3]

В train.py:

```diff
-def loadandprepdatanpzpath: str, splitname: str, normstats: dict,
-                       allowedassets: Optional[List[str]] = None
-                       ) -> Tuple[List[np.ndarray], List[str]]:
+def loadandprepdatanpzpath: str,
+                       splitname: str,
+                       normstats: dict,
+                       allowedassets: Optional[List[str]] = None
+                       ) -> Tuple[List[np.ndarray], List[str]]:
@@
-    episodesperepoch = len(sequences)
-    try:
-        from config import cfg
-        if hasattr(cfg, "episodesperepoch"):
-            episodesperepoch = getattr(cfg, "episodesperepoch")
-        else:
-            episodesperepoch = len(sequences)
-        if len(sequences) > episodesperepoch:
-            np.random.seed(25)
-            indices = np.random.choice(len(sequences), episodesperepoch, replace=False)
-            sequences = [sequences[i] for i in sorted(indices)]
-            validkeys = [validkeys[i] for i in sorted(indices)]
-            print(f"Sampled to {episodesperepoch} episodes")
-    except (ImportError, AttributeError):
-        print("Config not available, using full sequences")
+    # Никакого доступа к cfg и никакого внутреннего семплинга.
+    # Функция отвечает только за загрузку и Z-нормировку.
```

В main(cfg) (train.py) — добавить явное семплирование уже подготовленных seqs:[2][1]

```diff
-    trainseqs, trainkeys = loadandprepdata(cfg.paths.traindatapath, "Train",
-                                           normstats=normstats,
-                                           allowedassets=allowedassets)
+    trainseqs, trainkeys = loadandprepdata(
+        cfg.paths.traindatapath,
+        "Train",
+        normstats=normstats,
+        allowedassets=allowedassets,
+    )
+
+    episodes_per_epoch = getattr(cfg.trainlog, "episodesperepoch", None)
+    if episodes_per_epoch is not None and len(trainseqs) > episodes_per_epoch:
+        rng = np.random.default_rng(cfg.randomseed)
+        indices = rng.choice(len(trainseqs), episodes_per_epoch, replace=False)
+        indices = sorted(indices.tolist())
+        trainseqs = [trainseqs[i] for i in indices]
+        trainkeys = [trainkeys[i] for i in indices]
+        logging.info(
+            "Sampled train set down to %d episodes from %d",
+            episodes_per_epoch,
+            len(indices),
+        )
```

(Аналогично можешь добавить опциональный sampling для валидации, если понадобится.)

***

## 4) Сделать compute_norm_stats управляемым через конфиг, без магических чисел

Цель: вынести `num_samples_per_asset` и seed в cfg, убрать хардкод.[3][1][2]

В config.py (добавить небольшой блок, если хочешь отдельно конфигурировать):

```diff
 class DataConfig(BaseModel):
     numchannels: int = 10
@@
-    otherchannels: List[str] = ["numtrades"]
+    otherchannels: List[str] = ["numtrades"]
+    norm_num_samples_per_asset: int = 1000
+    norm_seed: int = 25
```

В train.py (функция compute_norm_stats):[1]

```diff
-def computenormstats(npzpath: str, numsamplesperasset: int = 1000, seed: int = 25) -> dict:
-    np.random.seed(seed)
+def computenormstats(npzpath: str, cfg: MasterConfig) -> dict:
+    numsamplesperasset = cfg.data.norm_num_samples_per_asset
+    seed = cfg.data.norm_seed
+    np.random.seed(seed)
@@
-    Path("normstats.json").write_text(json.dumps(allstats, indent=2))
+    Path("normstats.json").write_text(json.dumps(allstats, indent=2))
```

И там, где вызывается:[1]

```diff
-    if forcerecompute or normstats is None:
-        logging.info(f"{cfg.paths.traindatapath} normstats: compute_norm_stats")
-        normstats = compute_norm_stats(cfg.paths.traindatapath)
+    if forcerecompute or normstats is None:
+        logging.info("%s normstats: compute_norm_stats", cfg.paths.traindatapath)
+        normstats = compute_norm_stats(cfg.paths.traindatapath, cfg)
```

***

## 5) Упростить частоту вызова agent.learn в rollout_vectorized_episode

Цель: убрать двойной вызов learn на каждый шаг и сделать поведение предсказуемым.[4][1]

В train.py, внутри `rolloutvectorizedepisode`:

```diff
-    while not donemask.all():
+    while not donemask.all():
@@
-        for i in range(trainenv.num_envs):
-            agent.increment_step()
-
-        loss = agent.learn()
-        if loss is not None:
-            eplosses.append(loss)
-
-        # Final learn calls if buffer full...
-        for _ in range(trainenv.num_envs):
-            agent.learn()
+        for _ in range(trainenv.num_envs):
+            agent.increment_step()
+
+        # Один вызов обучения на batched шаг.
+        loss = agent.learn()
+        if loss is not None:
+            eplosses.append(loss)
```

(Если тебе нужен дополнительный финальный прогон learn после эпизода — лучше сделать это явно с лимитом количества итераций.)

***

## 6) Улучшить расчёт MaxDD в evaluate_agent (без обязательной замены, но очень желательно)

Цель: сделать расчёт MaxDD более устойчивым и немного эффективнее.[1]

В `evaluateagent`:

```diff
-    if tradepnls:
-        denormpnls = np.array(tradepnls, dtype=np.float64)
-        equitycurve = np.cumsum(denormpnls) + initialbalance
-        peak = np.maximum.accumulate(equitycurve)
-        drawdowns = (equitycurve - peak) / peak
-        maxdd = float(np.min(drawdowns))
-    else:
-        maxdd = 0.0
+    if tradepnls:
+        denormpnls = np.array(tradepnls, dtype=np.float64)
+        equity = float(initialbalance)
+        peak = float(initialbalance)
+        maxdd = 0.0
+        for pnl in denormpnls:
+            equity += pnl
+            if equity > peak:
+                peak = equity
+            if peak > 0.0:
+                dd = (equity - peak) / peak
+                if dd < maxdd:
+                    maxdd = dd
+    else:
+        maxdd = 0.0
```

***