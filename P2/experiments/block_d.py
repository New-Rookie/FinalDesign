"""
Block D — Weight-ratio sweep: compare all algorithms under different w_Q : w_S.

Fix N_total=20, eta_ch=1.0.
Sweep w_Q : w_S over {0.2:0.8, 0.3:0.7, 0.5:0.5, 0.7:0.3, 0.8:0.2}.

For each weight configuration, train GMAPPO / MAPPO and run Greedy / ACO / GA,
then report mean Q_pi, mean S_pi, and mean LA separately.

This allows three comparison plots:
  1. Q_pi  vs w_Q  for all algorithms
  2. S_pi  vs w_Q  for all algorithms
  3. LA_pi vs w_Q  for all algorithms

Output: block_d_raw.csv, block_d_summary.csv
"""
from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

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

WQ_VALUES = [0.2, 0.3, 0.5, 0.7, 0.8]
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


def _worker(args):
    import torch
    torch.set_num_threads(1)
    w_q, seed, algo_name, estimator_path, n_train, n_eval, n_windows_train, device, counter = args
    w_s = round(1.0 - w_q, 2)

    cfg = EnvConfig(N_total=20, eta_ch=1.0, w_Q=w_q, w_S=w_s,
                    print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode='link_selection',
                       max_steps=max(n_windows_train, n_eval) * 20 + 200)
    est = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        est.load(estimator_path)
    rng = np.random.default_rng(seed)
    env.reset()
    n_actual = len(env.nodes)
    a_lr, c_lr = BEST_LR_P2

    if algo_name == "GMAPPO":
        agent = GMAPPO(n_actual, cfg, est, actor_lr=a_lr, critic_lr=c_lr,
                       lr_t_max=n_train, device=device)
        mean_la, mean_q, mean_s = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "MAPPO":
        agent = MAPPO(n_actual, cfg, est, actor_lr=a_lr, critic_lr=c_lr, device=device)
        mean_la, mean_q, mean_s = _train_and_eval_rl(
            agent, env, n_train, n_eval, rng, n_windows_train, counter)
    elif algo_name == "Greedy":
        agent = GreedySelector(n_actual, cfg, est)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1
    elif algo_name == "ACO":
        agent = ACOSelector(n_actual, cfg, est)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1
    else:  # GA
        agent = GASelector(n_actual, cfg, est)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
        mean_q = result["mean_Q"]
        mean_s = result["mean_S"]
        if counter is not None:
            counter.value += 1

    env.close()
    return [{
        'experiment': 'D',
        'w_Q': w_q,
        'w_S': w_s,
        'seed': seed,
        'algorithm': algo_name,
        'mean_LA': mean_la,
        'mean_Q': mean_q,
        'mean_S': mean_s,
    }]


def run_block_d(log_dir='P2/logs', estimator_path=None,
                n_seeds=N_SEEDS, n_train=N_TRAIN_EPISODES,
                n_eval=N_EVAL_WINDOWS, n_windows_train=N_WINDOWS_TRAIN,
                n_workers=None, device="cpu"):
    os.makedirs(log_dir, exist_ok=True)
    n_workers = min(os.cpu_count() or 1, 48) if n_workers is None else n_workers

    if estimator_path is None:
        default_path = os.path.join(log_dir, "rf_estimator.pkl")
        if os.path.exists(default_path):
            estimator_path = default_path

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    units = [(w_q, s, algo_name, estimator_path, n_train, n_eval,
              n_windows_train, device, counter)
             for w_q in WQ_VALUES
             for s in range(n_seeds)
             for algo_name in ALGO_NAMES]

    n_rl = sum(1 for u in units if u[2] in RL_ALGOS)
    n_nonrl = len(units) - n_rl
    total_steps = n_rl * (n_train + n_eval) + n_nonrl

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, u) for u in units]
        rows = poll_progress(futures, counter, total_steps,
                             f"Block D ({len(units)} cfgs, {total_steps} steps)",
                             unit="step")

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(index=['w_Q', 'w_S'], columns='algorithm',
                           values=['mean_LA', 'mean_Q', 'mean_S'],
                           aggfunc=['mean', 'std'])
    pivot.columns = [f"{algo}_{metric}_{stat}"
                     for stat, metric, algo in pivot.columns]
    pivot = pivot.reset_index()

    save_block_results("P2", "D", raw_df=df, summary_df=pivot, log_dir=log_dir)
    return pivot


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_d()
    print(summary.to_string(index=False))
    print(f"\nBlock D completed in {time.time() - t0:.1f}s")
