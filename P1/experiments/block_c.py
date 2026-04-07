"""
Experiment Block C — Actor/Critic LR pair sweep for ME-IPPO.

Fix eta_N = 1.0, N_total = 50.
Train ME-IPPO at 5 (actor_lr, critic_lr) pairs.
Log per-episode mean reward.

V14: N_EPISODES 1000->1500, N_SEEDS 1->5.
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
BEST_LR_PAIR_P1 = (3e-4, 1e-3)
N_SEEDS = 1
N_EPISODES = 1500
N_WINDOWS_PER_EP = 50


def _run_single_config_c(args):
    import torch
    torch.set_num_threads(1)
    actor_lr, critic_lr, seed, n_episodes, n_windows, device, counter = args
    cfg = EnvConfig(N_total=50, eta_N=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_windows * cfg.N_slot)
    rng = np.random.default_rng(seed)
    protocol = INDPProtocol(cfg)

    agent = MemoryEnhancedIPPO(
        n_agents=cfg.N_total, obs_dim=16, act_dim=2,
        actor_lr=actor_lr, critic_lr=critic_lr, cfg=cfg, device=device)

    records = []
    for ep in range(n_episodes):
        ep_info = agent.train_episode(env, protocol, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "C",
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "lr_label": f"({actor_lr:.0e},{critic_lr:.0e})",
            "seed": seed,
            "episode": ep,
            "mean_reward": ep_info["mean_reward"],
            "mean_f1": ep_info["mean_f1"],
            "mean_energy": ep_info["mean_energy"],
            "policy_loss": ep_info.get("policy_loss", 0),
            "value_loss": ep_info.get("value_loss", 0),
        })
        if counter is not None:
            counter.value += 1

    env.close()
    return records


def run_block_c(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_episodes: int = N_EPISODES,
                n_windows: int = N_WINDOWS_PER_EP,
                n_workers: int = None,
                device: str = "cpu") -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_list = [(a_lr, c_lr, seed, n_episodes, n_windows, device, counter)
                 for a_lr, c_lr in LR_PAIRS
                 for seed in range(n_seeds)]

    n_configs = len(args_list)
    total_steps = n_configs * n_episodes

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_run_single_config_c, a) for a in args_list]
        all_records = poll_progress(
            futures, counter, total_steps,
            f"Block C ({n_configs} LR pairs × {n_episodes} ep)",
            unit="ep")

    df = pd.DataFrame(all_records)

    summary = df.groupby(["lr_label", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()

    save_block_results("P1", "C", raw_df=df, summary_df=summary, log_dir=log_dir)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_c()
    print(f"\nBlock C completed in {time.time() - t0:.1f}s")
