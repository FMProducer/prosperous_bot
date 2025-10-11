from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple, Any, Sequence

class DummyVecEnv:
    """
    Минимальная синхронная векторизация без зависимостей (аналог gym Sync/DummyVecEnv):
    - хранит N независимых копий среды в одном процессе;
    - reset()/step() работают с батчами наблюдений/действий;
    - при done/terminated/ truncated — авто-reset соответствующей под-среды.
    """
    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        assert len(env_fns) >= 1, "Need at least one env_fn"
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self, seed=None, options=None):
        obs_batch, infos = [], []
        for i, env in enumerate(self.envs):
            s = None if seed is None else (seed + i if isinstance(seed, int) else None)
            obs, info = env.reset(seed=s, options=options)
            obs_batch.append(obs)
            infos.append(info)
        return np.stack(obs_batch), infos

    def step(self, actions: Sequence[Any]):
        assert len(actions) == self.num_envs, "actions must match num_envs"
        obs_b, rew_b, done_b, trunc_b, infos = [], [], [], [], []
        for env, act in zip(self.envs, actions):
            next_obs, reward, done, truncated, info = env.step(act)
            # autoreset для закончившихся эпизодов
            if done or truncated:
                info = dict(info or {})
                info["terminal_observation"] = next_obs
                next_obs, info_reset = env.reset(seed=None, options=None)
                info["reset_info"] = info_reset
            obs_b.append(next_obs)
            rew_b.append(float(reward))
            done_b.append(bool(done))
            trunc_b.append(bool(truncated))
            infos.append(info)
        return (
            np.stack(obs_b),
            np.asarray(rew_b, dtype=float),
            np.asarray(done_b, dtype=bool),
            np.asarray(trunc_b, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
