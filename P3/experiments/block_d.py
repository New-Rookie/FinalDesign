"""
Experiment Block D — Closed-loop total-data-volume variation (parallelised).

Fix eta_B=eta_F=eta_S=1.0, N_total=15.
Sweep M_tot over {20, 40, 60, 80, 100} Mbit.
Compare: Improved MATD3, MATD3, Greedy, GA, ACO.
Primary metrics: average T_total, average E_total, and throughput mean_Gamma.

Output: block_d_raw.csv, block_d_summary.csv
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
from P3.algorithms.matd3 import MATD3
from P3.algorithms.greedy import GreedyAllocator
from P3.algorithms.aco import ACOAllocator
from P3.algorithms.ga import GAAllocator
from P3.experiments.block_a import BEST_LR_P3
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

M_TOT_VALUES = [20e6, 40e6, 60e6, 80e6, 100e6]
N_SEEDS = 1
N_TRAIN = 500
N_EVAL = 50
ALGO_NAMES = ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]
RL_ALGOS = {"Improved_MATD3", "MATD3"}
SOURCE_ACTIVATION_RATIO = 0.6


def _train_and_eval_rl(agent, env, n_train, n_eval, rng,
                       n_windows_train=10, counter=None):
    for _ in range(n_train):
        agent.train_episode(env, n_windows=n_windows_train, rng=rng)
        if counter is not None:
            counter.value += 1
    env.reset()
    T_vals, E_vals, G_vals = [], [], []
    for _ in range(n_eval):
        m = agent.eval_window(env, rng=rng)
        T_vals.append(m["mean_T_total"])
        E_vals.append(m["mean_E_total"])
        G_vals.append(m["mean_Gamma"])
        if counter is not None:
            counter.value += 1
    return float(np.mean(T_vals)), float(np.mean(E_vals)), float(np.mean(G_vals))


def _worker_block_d(args):
    m_tot, seed, algo_name, n_train, n_eval, n_windows_train, device, counter = args
    import torch
    torch.set_num_threads(1)
    cfg = EnvConfig(
        N_total=15, M_tot=m_tot,
        source_activation_ratio=SOURCE_ACTIVATION_RATIO,
        print_diagnostics=False,
    )
    env = MarineIoTEnv(cfg, mode="resource_mgmt",
                       max_steps=n_eval * 20 + 100)
    rng = np.random.default_rng(seed)
    n = min(cfg.N_src, cfg.node_counts["buoy"])

    a_lr, c_lr = BEST_LR_P3
    if algo_name == "Improved_MATD3":
        agent = ImprovedMATD3(n, cfg, actor_lr=a_lr, critic_lr=c_lr,
                              n_episodes=n_train, device=device)
        mean_T, mean_E, mean_G = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "MATD3":
        agent = MATD3(n, cfg, actor_lr=a_lr, critic_lr=c_lr, device=device)
        mean_T, mean_E, mean_G = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "Greedy":
        agent = GreedyAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
        if counter is not None:
            counter.value += 1
    elif algo_name == "ACO":
        agent = ACOAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
        if counter is not None:
            counter.value += 1
    else:
        agent = GAAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
        if counter is not None:
            counter.value += 1

    env.close()
    return [{
        "experiment": "D",
        "M_tot_Mbit": m_tot / 1e6,
        "seed": seed,
        "algorithm": algo_name,
        "mean_T_total": mean_T,
        "mean_E_total": mean_E,
        "mean_Gamma": mean_G,
        "source_activation_ratio": SOURCE_ACTIVATION_RATIO,
    }]


def run_block_d(
    log_dir: str = "P3/logs",
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN,
    n_eval: int = N_EVAL,
    n_windows_train: int = 10,
    n_workers: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    work_units = [
        (m_tot, seed, algo, n_train, n_eval, n_windows_train, device, counter)
        for m_tot in M_TOT_VALUES
        for seed in range(n_seeds)
        for algo in ALGO_NAMES
    ]

    n_rl = sum(1 for wu in work_units if wu[2] in RL_ALGOS)
    n_nonrl = len(work_units) - n_rl
    total_steps = n_rl * (n_train + n_eval) + n_nonrl

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker_block_d, wu) for wu in work_units]
        records = poll_progress(
            futures, counter, total_steps,
            f"Block D ({len(work_units)} cfgs, {total_steps} steps)",
            unit="step")

    df = pd.DataFrame(records)

    agg = df.groupby(["M_tot_Mbit", "algorithm"]).agg(
        mean_T=("mean_T_total", "mean"), std_T=("mean_T_total", "std"),
        mean_E=("mean_E_total", "mean"), std_E=("mean_E_total", "std"),
        mean_Gamma=("mean_Gamma", "mean"), std_Gamma=("mean_Gamma", "std"),
    ).reset_index()
    wide = agg.pivot_table(index="M_tot_Mbit", columns="algorithm",
                           values=["mean_T", "mean_E", "mean_Gamma"])
    wide.columns = [f"{v}_{a}" for v, a in wide.columns]
    wide = wide.reset_index()

    save_block_results("P3", "D", raw_df=df, summary_df=wide, log_dir=log_dir)
    return wide


if __name__ == "__main__":
    t0 = time.time()
    s = run_block_d()
    print(s.to_string(index=False))
    print(f"\nBlock D completed in {time.time() - t0:.1f}s")
