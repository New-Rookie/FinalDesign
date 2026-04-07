"""
Experiment Block A — GMAPPO learning-rate sweep  (parallelised).

Fix N_total = 20, eta_ch = 1.0.
Train GMAPPO at 5 (actor_lr, critic_lr) pairs.
Log per-episode mean reward and mean LA_pi.

Parallelisation granularity: one work-unit per (lr, seed) pair.
Each worker independently creates EnvConfig / env / rng / estimator / agent.

Output: block_a_raw.csv, block_a_summary.csv
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P2.link_quality.rf_estimator import LinkQualityEstimator
from P2.algorithms.gmappo import GMAPPO
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
BEST_LR_P2 = (3e-4, 1e-3)
N_SEEDS = 1
N_EPISODES = 1000
N_WINDOWS_PER_EP = 50


def _worker_block_a(args):
    """Train one (actor_lr, critic_lr, seed) configuration and return per-episode records."""
    import torch
    torch.set_num_threads(1)
    actor_lr, critic_lr, seed, estimator_path, n_episodes, n_windows, device, counter = args
    lr_label = f"a{actor_lr}_c{critic_lr}"

    cfg = EnvConfig(N_total=20, eta_ch=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=n_windows * 20 + 50)
    rng = np.random.default_rng(seed)

    estimator = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        estimator.load(estimator_path)

    env.reset()
    n_actual = len(env.nodes)
    agent = GMAPPO(n_actual, cfg, estimator, actor_lr=actor_lr, critic_lr=critic_lr,
                   lr_t_max=n_episodes, device=device)

    records: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        ep_info = agent.train_episode(env, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "A",
            "lr_label": lr_label,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "seed": seed,
            "episode": ep,
            "mean_reward": ep_info["mean_reward"],
            "mean_LA": ep_info["mean_LA"],
            "policy_loss": ep_info.get("policy_loss", 0),
            "value_loss": ep_info.get("value_loss", 0),
        })
        if counter is not None:
            counter.value += 1

    env.close()
    return records


def run_block_a(
    log_dir: str = "P2/logs",
    estimator: LinkQualityEstimator | None = None,
    estimator_path: str | None = None,
    n_seeds: int = N_SEEDS,
    n_episodes: int = N_EPISODES,
    n_windows: int = N_WINDOWS_PER_EP,
    n_workers: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    if estimator_path is None:
        default_path = os.path.join(log_dir, "rf_estimator.pkl")
        if estimator is not None and hasattr(estimator, "save"):
            estimator_path = os.path.join(log_dir, "_estimator_shared.pkl")
            estimator.save(estimator_path)
        elif os.path.exists(default_path):
            estimator_path = default_path

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    work_units = [
        (actor_lr, critic_lr, seed, estimator_path, n_episodes, n_windows, device, counter)
        for actor_lr, critic_lr in LR_PAIRS
        for seed in range(n_seeds)
    ]

    n_configs = len(work_units)
    total_episodes = n_configs * n_episodes

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker_block_a, wu) for wu in work_units]
        records = poll_progress(futures, counter, total_episodes,
                                f"Block A ({n_configs} cfgs × {n_episodes} ep)",
                                unit="ep")

    df = pd.DataFrame(records)

    summary = df.groupby(["lr_label", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()

    save_block_results("P2", "A", raw_df=df, summary_df=summary, log_dir=log_dir)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_a()
    print(f"\nBlock A completed in {time.time() - t0:.1f}s")
