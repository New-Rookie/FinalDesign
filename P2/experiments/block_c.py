"""
Experiment Block C — Algorithm comparison under channel-condition variation (parallelised).

Fix N_total = 20, sweep eta_ch in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}.
Compare GMAPPO / MAPPO / Greedy / ACO / GA.
Primary metric: mean LA_pi (also reports mean Q_pi, mean S_pi).

Output: block_c_raw.csv, block_c_summary.csv
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
from P2.algorithms.mappo import MAPPO
from P2.algorithms.greedy import GreedySelector
from P2.algorithms.aco import ACOSelector
from P2.algorithms.ga import GASelector
from P2.experiments.block_a import BEST_LR_P2
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ETA_CH_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_SEEDS = 1
N_TRAIN_EPISODES = 1000
N_EVAL_WINDOWS = 50
N_WINDOWS_TRAIN = 50
ALGO_NAMES = ["GMAPPO", "MAPPO", "Greedy", "ACO", "GA"]
RL_ALGOS = {"GMAPPO", "MAPPO"}


def _train_and_eval_rl(agent, env, n_train, n_eval, rng, n_windows_train=50,
                       counter=None):
    for _ in range(n_train):
        agent.train_episode(env, n_windows=n_windows_train, rng=rng)
        if counter is not None:
            counter.value += 1

    agent._prev_actions.clear()
    agent._sinr_histories.clear()
    env.reset()
    source_ids = agent.path_mgr.select_source_buoys(
        env.nodes, agent.cfg.N_src, rng)

    las, qs, ss = [], [], []
    for _ in range(n_eval):
        result = agent.run_window(env, source_ids, n_steps=10, rng=rng)
        las.append(result["mean_LA"])
        qs.append(result["mean_Q"])
        ss.append(result["mean_S"])
        if counter is not None:
            counter.value += 1
    return (float(np.mean(las)), float(np.mean(qs)), float(np.mean(ss)))


def _worker_block_c(args):
    """Run one (eta_ch, seed, algorithm) configuration."""
    import torch
    torch.set_num_threads(1)
    eta_ch, seed, algo_name, estimator_path, n_train, n_eval, n_windows_train, device, counter = args

    cfg = EnvConfig(N_total=20, eta_ch=eta_ch, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=max(n_windows_train, n_eval) * 20 + 200)
    rng = np.random.default_rng(seed)

    estimator = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        estimator.load(estimator_path)

    env.reset()
    n = len(env.nodes)
    a_lr, c_lr = BEST_LR_P2

    if algo_name == "GMAPPO":
        agent = GMAPPO(n, cfg, estimator, actor_lr=a_lr, critic_lr=c_lr,
                       lr_t_max=n_train, device=device)
        mean_la, mean_q, mean_s = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "MAPPO":
        agent = MAPPO(n, cfg, estimator, actor_lr=a_lr, critic_lr=c_lr, device=device)
        mean_la, mean_q, mean_s = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "Greedy":
        agent = GreedySelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1
    elif algo_name == "ACO":
        agent = ACOSelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1
    else:  # GA
        agent = GASelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1

    env.close()

    return [{
        "experiment": "C",
        "eta_ch": eta_ch,
        "seed": seed,
        "algorithm": algo_name,
        "mean_LA": mean_la,
        "mean_Q": mean_q,
        "mean_S": mean_s,
    }]


def run_block_c(
    log_dir: str = "P2/logs",
    estimator: LinkQualityEstimator | None = None,
    estimator_path: str | None = None,
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN_EPISODES,
    n_eval: int = N_EVAL_WINDOWS,
    n_windows_train: int = N_WINDOWS_TRAIN,
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
        (eta_ch, seed, algo_name, estimator_path, n_train, n_eval,
         n_windows_train, device, counter)
        for eta_ch in ETA_CH_VALUES
        for seed in range(n_seeds)
        for algo_name in ALGO_NAMES
    ]

    n_rl = sum(1 for wu in work_units if wu[2] in RL_ALGOS)
    n_nonrl = len(work_units) - n_rl
    total_steps = n_rl * (n_train + n_eval) + n_nonrl

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker_block_c, wu) for wu in work_units]
        records = poll_progress(futures, counter, total_steps,
                                f"Block C ({len(work_units)} cfgs, {total_steps} steps)",
                                unit="step")

    df = pd.DataFrame(records)

    pivot = df.pivot_table(index="eta_ch", columns="algorithm",
                           values=["mean_LA", "mean_Q", "mean_S"],
                           aggfunc=["mean", "std"])
    pivot.columns = [f"{algo}_{metric}_{stat}"
                     for stat, metric, algo in pivot.columns]
    pivot = pivot.reset_index()

    save_block_results("P2", "C", raw_df=df, summary_df=pivot, log_dir=log_dir)
    return pivot


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_c()
    print(summary.to_string(index=False))
    print(f"\nBlock C completed in {time.time() - t0:.1f}s")
