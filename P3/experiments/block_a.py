"""
Experiment Block A — Improved MATD3 learning-rate sweep (parallelised).

Fix N_total=15, M_tot=60 Mbit, eta_B=eta_F=eta_S=1.0.
Train Improved MATD3 at 5 (actor_lr, critic_lr) pairs.
Log per-episode mean reward, mean T_total, mean E_total, mean Gamma.

Output: block_a_raw.csv, block_a_summary.csv
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P3.algorithms.improved_matd3 import ImprovedMATD3
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

LR_PAIRS = [
    (1e-4, 3e-4),
    (3e-4, 1e-3),
    (5e-4, 1e-3),
    (1e-4, 1e-4),
    (3e-4, 3e-4),
]
BEST_LR_P3 = (3e-4, 1e-3)  # (actor_lr, critic_lr)
N_SEEDS = 1
N_EPISODES = 500
N_WINDOWS = 5


def _worker_block_a(
    args: Tuple[Tuple[float, float], int, int, int, str, Any],
) -> List[Dict[str, Any]]:
    lr_pair, seed, n_episodes, n_windows, device, counter = args
    actor_lr, critic_lr = lr_pair
    import torch
    torch.set_num_threads(1)
    cfg = EnvConfig(N_total=15, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="resource_mgmt",
                       max_steps=n_windows * 20 + 50)
    rng = np.random.default_rng(seed)

    lr_label = f"a{actor_lr:.0e}_c{critic_lr:.0e}"
    agent = ImprovedMATD3(min(cfg.N_src, cfg.node_counts["buoy"]), cfg,
                          actor_lr=actor_lr, critic_lr=critic_lr,
                          n_episodes=n_episodes, device=device)
    records: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        info = agent.train_episode(env, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "A",
            "lr_label": lr_label,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "seed": seed,
            "episode": ep,
            "mean_reward": info["mean_reward"],
            "mean_T_total": info["mean_T_total"],
            "mean_E_total": info["mean_E_total"],
            "mean_Gamma": info["mean_Gamma"],
            "policy_loss": info.get("policy_loss", 0),
            "value_loss": info.get("value_loss", 0),
        })
        if counter is not None:
            counter.value += 1
    env.close()
    return records


def run_block_a(
    log_dir: str = "P3/logs",
    n_seeds: int = N_SEEDS,
    n_episodes: int = N_EPISODES,
    n_windows: int = N_WINDOWS,
    n_workers: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    work_units = [
        (lr_pair, seed, n_episodes, n_windows, device, counter)
        for lr_pair in LR_PAIRS
        for seed in range(n_seeds)
    ]

    n_configs = len(work_units)
    total_steps = n_configs * n_episodes

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker_block_a, wu) for wu in work_units]
        records = poll_progress(futures, counter, total_steps,
                                f"Block A ({n_configs} LR pairs × {n_episodes} ep)",
                                unit="ep")

    df = pd.DataFrame(records)

    summary = df.groupby(["lr_label", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()
    wide = summary.pivot_table(index="episode", columns="lr_label",
                               values="mean").reset_index()

    save_block_results("P3", "A", raw_df=df, summary_df=wide, log_dir=log_dir)
    return wide


if __name__ == "__main__":
    t0 = time.time()
    s = run_block_a()
    print(f"\nBlock A completed in {time.time() - t0:.1f}s")
