"""
Experiment Block B — Mechanism comparison under node-count variation.

Fix eta_N = 1.0, sweep N_total over {20, 35, 50, 65, 80}.
Compare INDP / Disco / ALOHA.  Primary metric: F1_topo.
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
from P1.protocols.disco import DiscoProtocol
from P1.protocols.aloha import ALOHAProtocol
from utils.progress import poll_progress
from utils.result_saver import save_block_results

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

N_TOTAL_VALUES = [20, 35, 50, 65, 80]
N_SEEDS = 1
N_WINDOWS = 10
PROTOCOL_NAMES = ["INDP", "Disco", "ALOHA"]


def _run_single_config_b(args):
    import torch
    torch.set_num_threads(1)
    n_total, seed, n_windows, counter = args
    cfg = EnvConfig(N_total=n_total, eta_N=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_windows * cfg.N_slot)
    rng = np.random.default_rng(seed)

    protocols = {
        "INDP": INDPProtocol(cfg),
        "Disco": DiscoProtocol(cfg),
        "ALOHA": ALOHAProtocol(cfg),
    }

    records = []
    for name, proto in protocols.items():
        obs, info = env.reset(seed=seed)
        nodes = env.nodes
        n = len(nodes)
        f1_values = []

        for w in range(n_windows):
            env.recompute_ground_truth()
            result = proto.run_window(nodes, cfg, rng)
            env.set_discovered_topology(result["disc_adj"])
            gt = env.get_ground_truth_topology()
            f1, *_ = proto.compute_f1(gt, n)
            f1_values.append(f1)
            actions = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions)
            if term or trunc:
                obs, _ = env.reset(seed=seed)
            if counter is not None:
                counter.value += 1

        records.append({
            "experiment": "B",
            "N_total": n_total,
            "seed": seed,
            "mechanism": name,
            "mean_f1_topo": float(np.mean(f1_values)),
        })
    env.close()
    return records


def run_block_b(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_windows: int = N_WINDOWS, n_workers: int = None) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_list = [(n_total, seed, n_windows, counter)
                 for n_total in N_TOTAL_VALUES
                 for seed in range(n_seeds)]

    n_configs = len(args_list)
    total_steps = n_configs * len(PROTOCOL_NAMES) * n_windows

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_run_single_config_b, a) for a in args_list]
        all_records = poll_progress(
            futures, counter, total_steps,
            f"Block B ({n_configs} cfgs × {len(PROTOCOL_NAMES)} protos × {n_windows} wins)",
            unit="win")

    df = pd.DataFrame(all_records)

    pivot = df.pivot_table(index="N_total", columns="mechanism",
                           values="mean_f1_topo", aggfunc=["mean", "std"])
    pivot.columns = [f"{mech}_{stat}" for stat, mech in pivot.columns]
    pivot = pivot.reset_index()

    save_block_results("P1", "B", raw_df=df, summary_df=pivot, log_dir=log_dir)
    return pivot


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_b()
    print(summary.to_string(index=False))
    print(f"\nBlock B completed in {time.time() - t0:.1f}s")
