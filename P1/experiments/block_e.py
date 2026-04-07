"""
Experiment Block E — Algorithm comparison under node-count variation.

Fix eta_N = 1.0, sweep N_total over {20, 35, 50, 65, 80}.
Compare ME-IPPO / Improved IPPO / IPPO / Greedy / ACO / GA.
Primary metric: total E_ND (sum over all nodes).

V14: N_TRAIN_EPISODES 800->1500, n_windows_train 5->10, N_SEEDS 1->5.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P1.protocols.indp import INDPProtocol
from P1.algorithms.me_ippo import MemoryEnhancedIPPO
from P1.algorithms.improved_ippo import ImprovedIPPO
from P1.algorithms.ippo import IPPO
from P1.algorithms.greedy import GreedyOptimizer
from P1.algorithms.aco import ACOOptimizer
from P1.algorithms.ga import GAOptimizer
from P1.experiments.block_c import BEST_LR_PAIR_P1
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

N_TOTAL_VALUES = [20, 35, 50, 65, 80]
N_SEEDS = 1
N_TRAIN_EPISODES = 1500
N_EVAL_WINDOWS = 50

ALGO_NAMES = ["ME_IPPO", "Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]
RL_ALGOS = {"ME_IPPO", "Improved_IPPO", "IPPO"}


def _train_and_eval_rl(agent, env, protocol, cfg, rng, n_train, n_eval,
                       n_windows_train=10, counter=None):
    for ep in range(n_train):
        agent.train_episode(env, protocol, n_windows=n_windows_train, rng=rng)
        if counter is not None:
            counter.value += 1

    obs, _ = env.reset()
    nodes = env.nodes
    n = len(nodes)
    energies = []
    for w in range(n_eval):
        if hasattr(agent, 'build_global_state'):
            if hasattr(agent, '_augment_obs'):
                aug_obs = agent._augment_obs(obs)
                gs = agent.build_global_state(aug_obs)
                actions, _ = agent.select_actions(aug_obs, gs)
            else:
                gs = agent.build_global_state(obs)
                actions, _ = agent.select_actions(obs, gs)
        else:
            actions, _ = agent.select_actions(obs)
        result = protocol.run_window(nodes, cfg, rng, [actions] * cfg.N_slot)
        energies.append(result["mean_energy"] * n)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            obs, _ = env.reset()
        if counter is not None:
            counter.value += 1
    return float(np.mean(energies))


def _run_single_config_e(args):
    import torch
    torch.set_num_threads(1)
    n_total, seed, algo_name, n_train, n_eval, n_windows_train, device, counter = args
    cfg = EnvConfig(N_total=n_total, eta_N=1.0, print_diagnostics=False)
    rng = np.random.default_rng(seed)
    n = n_total
    a_lr, c_lr = BEST_LR_PAIR_P1

    algo_factories = {
        "ME_IPPO": lambda: MemoryEnhancedIPPO(n, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device),
        "Improved_IPPO": lambda: ImprovedIPPO(n, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device),
        "IPPO": lambda: IPPO(n, cfg=cfg, actor_lr=a_lr, critic_lr=c_lr, device=device),
        "Greedy": lambda: GreedyOptimizer(n, cfg=cfg),
        "ACO": lambda: ACOOptimizer(n, cfg=cfg),
        "GA": lambda: GAOptimizer(n, cfg=cfg),
    }

    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_eval * cfg.N_slot)
    protocol = INDPProtocol(cfg)
    agent = algo_factories[algo_name]()

    if algo_name in RL_ALGOS:
        total_e = _train_and_eval_rl(agent, env, protocol, cfg, rng,
                                     n_train, n_eval, n_windows_train, counter)
    else:
        result = agent.run_episode(env, protocol, n_eval, rng)
        total_e = result["mean_energy"] * n
        if counter is not None:
            counter.value += 1

    env.close()
    return [{
        "experiment": "E",
        "N_total": n_total,
        "seed": seed,
        "algorithm": algo_name,
        "total_E_ND": total_e,
    }]


def run_block_e(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_train: int = N_TRAIN_EPISODES,
                n_eval: int = N_EVAL_WINDOWS,
                n_windows_train: int = 10,
                n_workers: int = None,
                device: str = "cpu") -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_list = [(n_total, seed, algo_name, n_train, n_eval, n_windows_train,
                  device, counter)
                 for n_total in N_TOTAL_VALUES
                 for seed in range(n_seeds)
                 for algo_name in ALGO_NAMES]

    n_rl = sum(1 for _, _, a, *_ in args_list if a in RL_ALGOS)
    n_nonrl = len(args_list) - n_rl
    total_steps = n_rl * (n_train + n_eval) + n_nonrl

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_run_single_config_e, a) for a in args_list]
        all_records = poll_progress(
            futures, counter, total_steps,
            f"Block E ({len(args_list)} algo-cfgs, {total_steps} steps)",
            unit="step")

    df = pd.DataFrame(all_records)

    pivot = df.pivot_table(index="N_total", columns="algorithm",
                           values="total_E_ND", aggfunc=["mean", "std"])
    pivot.columns = [f"{algo}_{stat}" for stat, algo in pivot.columns]
    pivot = pivot.reset_index()

    save_block_results("P1", "E", raw_df=df, summary_df=pivot, log_dir=log_dir)
    return pivot


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_e()
    print(summary.to_string(index=False))
    print(f"\nBlock E completed in {time.time() - t0:.1f}s")
