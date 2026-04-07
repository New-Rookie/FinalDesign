"""Block F — convergence comparison for RL algorithms (RC1).

3-way comparison: ME-IPPO vs Improved IPPO vs IPPO.
Uses best LR pair from Block C.

V14: N_EPISODES 800->1500, N_SEEDS 1->5.
"""
from __future__ import annotations
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P1.protocols.indp import INDPProtocol
from P1.algorithms.me_ippo import MemoryEnhancedIPPO
from P1.algorithms.improved_ippo import ImprovedIPPO
from P1.algorithms.ippo import IPPO
from P1.experiments.block_c import BEST_LR_PAIR_P1
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

N_SEEDS = 5
N_EPISODES = 1500
N_WINDOWS = 50
ALGO_NAMES = ["ME_IPPO", "Improved_IPPO", "IPPO"]


def _worker(args):
    import torch
    torch.set_num_threads(1)
    algo, seed, n_episodes, n_windows, device, counter = args
    cfg = EnvConfig(N_total=50, eta_N=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode='discovery', max_steps=n_windows * cfg.N_slot)
    protocol = INDPProtocol(cfg)
    rng = np.random.default_rng(seed)
    a_lr, c_lr = BEST_LR_PAIR_P1
    if algo == "ME_IPPO":
        agent = MemoryEnhancedIPPO(cfg.N_total, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device)
    elif algo == "Improved_IPPO":
        agent = ImprovedIPPO(cfg.N_total, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device)
    else:
        agent = IPPO(cfg.N_total, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device)
    rows = []
    for ep in range(n_episodes):
        info = agent.train_episode(env, protocol, n_windows=n_windows, rng=rng)
        rows.append({"experiment": "F", "algorithm": algo, "seed": seed,
                      "episode": ep, "mean_reward": info["mean_reward"]})
        if counter is not None:
            counter.value += 1
    env.close()
    return rows


def run_block_f(log_dir='P1/logs', n_seeds=N_SEEDS, n_episodes=N_EPISODES,
                n_windows=N_WINDOWS, n_workers=None, device="cpu"):
    os.makedirs(log_dir, exist_ok=True)
    n_workers = min(os.cpu_count() or 1, 48) if n_workers is None else n_workers

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    units = [(a, s, n_episodes, n_windows, device, counter)
             for a in ALGO_NAMES for s in range(n_seeds)]

    n_configs = len(units)
    total_steps = n_configs * n_episodes

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, u) for u in units]
        rows = poll_progress(
            futures, counter, total_steps,
            f"Block F ({n_configs} algos × {n_episodes} ep)",
            unit="ep")

    df = pd.DataFrame(rows)
    summary = df.groupby(['algorithm', 'episode'])['mean_reward'].agg(['mean', 'std']).reset_index()

    save_block_results("P1", "F", raw_df=df, summary_df=summary, log_dir=log_dir)
    return summary
