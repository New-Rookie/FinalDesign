"""
Experiment Block B — Average total delay under resource variation (parallelised).

Three separate sweeps: eta_B, eta_F, eta_S.
Only one scaling factor varies at a time; the other two remain at 1.0.
Compare: Improved MATD3, MATD3, Greedy, GA, ACO.

Block C (energy) is computed simultaneously.

Output: block_b_raw.csv, block_b_summary.csv
        block_c_raw.csv, block_c_summary.csv
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

ETA_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]
ETA_TYPES = ["eta_B", "eta_F", "eta_S"]
N_SEEDS = 1
N_TRAIN = 500
N_EVAL = 50
ALGO_NAMES = ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]
RL_ALGOS = {"Improved_MATD3", "MATD3"}


def _make_cfg(eta_type: str, eta_val: float) -> EnvConfig:
    kw = {"N_total": 15, "print_diagnostics": False}
    if eta_type == "eta_B":
        kw["eta_B"] = eta_val
    elif eta_type == "eta_F":
        kw["eta_F"] = eta_val
    elif eta_type == "eta_S":
        kw["eta_S"] = eta_val
    return EnvConfig(**kw)


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


def _worker_block_bc(args):
    eta_type, eta_val, seed, algo_name, n_train, n_eval, n_windows_train, device, counter = args
    import torch
    torch.set_num_threads(1)
    cfg = _make_cfg(eta_type, eta_val)
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
    else:  # GA
        agent = GAAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
        if counter is not None:
            counter.value += 1

    env.close()
    return [{
        "experiment": "BC",
        "eta_type": eta_type,
        "eta_value": eta_val,
        "seed": seed,
        "algorithm": algo_name,
        "mean_T_total": mean_T,
        "mean_E_total": mean_E,
        "mean_Gamma": mean_G,
    }]


def run_block_bc(
    log_dir: str = "P3/logs",
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN,
    n_eval: int = N_EVAL,
    n_windows_train: int = 10,
    n_workers: int | None = None,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    work_units = [
        (eta_type, eta_val, seed, algo, n_train, n_eval,
         n_windows_train, device, counter)
        for eta_type in ETA_TYPES
        for eta_val in ETA_VALUES
        for seed in range(n_seeds)
        for algo in ALGO_NAMES
    ]

    n_rl = sum(1 for wu in work_units if wu[3] in RL_ALGOS)
    n_nonrl = len(work_units) - n_rl
    total_steps = n_rl * (n_train + n_eval) + n_nonrl

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker_block_bc, wu) for wu in work_units]
        records = poll_progress(
            futures, counter, total_steps,
            f"Block B+C ({len(work_units)} cfgs, {total_steps} steps)",
            unit="step")

    df = pd.DataFrame(records)

    # Block B: delay
    df_b = df[["eta_type", "eta_value", "seed", "algorithm", "mean_T_total"]].copy()
    df_b.insert(0, "experiment", "B")
    sum_b = df_b.groupby(["eta_type", "eta_value", "algorithm"])["mean_T_total"].agg(
        ["mean", "std"]).reset_index()
    wide_b = sum_b.pivot_table(index=["eta_type", "eta_value"],
                               columns="algorithm", values="mean").reset_index()

    save_block_results("P3", "B", raw_df=df_b, summary_df=wide_b, log_dir=log_dir)

    # Block C: energy
    df_c = df[["eta_type", "eta_value", "seed", "algorithm", "mean_E_total"]].copy()
    df_c.insert(0, "experiment", "C")
    sum_c = df_c.groupby(["eta_type", "eta_value", "algorithm"])["mean_E_total"].agg(
        ["mean", "std"]).reset_index()
    wide_c = sum_c.pivot_table(index=["eta_type", "eta_value"],
                               columns="algorithm", values="mean").reset_index()

    save_block_results("P3", "C", raw_df=df_c, summary_df=wide_c, log_dir=log_dir)

    return wide_b, wide_c


if __name__ == "__main__":
    t0 = time.time()
    sb, sc = run_block_bc()
    print(f"\nBlock B+C completed in {time.time() - t0:.1f}s")
