from __future__ import annotations
import multiprocessing as mp
from typing import Callable, Sequence, Any, Tuple, List
import numpy as np

_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"

import torch

def _worker(remote, env_fn):
    # Configure torch threads for this worker process
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "_get_spaces":
                remote.send((env.action_space, env.observation_space))
            elif cmd == _CMD_RESET:
                seed, options = data
                obs, info = env.reset(seed=seed, options=options)
                remote.send((obs, info))
            elif cmd == _CMD_STEP:
                action = data
                obs, rew, terminated, truncated, info = env.step(action)
                # autoreset семантика: если эпизод кончился — сразу reset
                # ⚠️ Но НЕ пересылаем финальный кадр при terminated (он не нужен DQN-таргету:
                #     next_Q будет занулён). Передаём финальный кадр ТОЛЬКО при truncated,
                #     где нужен бутстрап корректного s'.
                if terminated or truncated:
                    info = dict(info or {})
                    if truncated:
                        info["terminal_observation"] = obs  # финальный кадр нужен для bootstrap
                    obs, info_reset = env.reset(seed=None, options=None)
                    info["reset_info"] = info_reset
                remote.send((obs, float(rew), bool(terminated), bool(truncated), info))
            elif cmd == _CMD_CLOSE:
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

class SubprocVecEnv:
    """
    Лёгкая мультипроцессная векторизация среды:
    - каждый воркер держит собственную копию env;
    - общение через Pipe (минимальный IPC);
    - интерфейс совместим с DummyVecEnv (reset -> stack, step -> stack).
    Подходит для CPU-тяжёлых окружений, где DummyVecEnv/потоки упираются в GIL. 
    """
    def __init__(self, env_fns: Sequence[Callable[[], Any]], start_method: str = "spawn"):
        assert len(env_fns) >= 1, "Need at least one env_fn"
        self.num_envs = len(env_fns)
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.procs: List[mp.Process] = []
        for work_remote, fn in zip(self.work_remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, fn), daemon=True)
            p.start()
            work_remote.close()
            self.procs.append(p)

    def reset(self, seed=None, options=None):
        for r in self.remotes:
            r.send((_CMD_RESET, (seed, options)))
        results = [r.recv() for r in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def step(self, actions: Sequence[Any]):
        assert len(actions) == self.num_envs, "actions must match num_envs"
        for r, a in zip(self.remotes, actions):
            r.send((_CMD_STEP, a))
        results = [r.recv() for r in self.remotes]
        obs, rews, terms, truncs, infos = zip(*results)
        return (np.stack(obs),
                np.asarray(rews, dtype=float),
                np.asarray(terms, dtype=bool),
                np.asarray(truncs, dtype=bool),
                list(infos))

    def close(self):
        for r in self.remotes:
            try:
                r.send((_CMD_CLOSE, None))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=0.2)
            except Exception:
                pass
