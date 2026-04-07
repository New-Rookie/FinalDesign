"""
Greedy baseline for MEC resource management.

For each source buoy, selects the local node with the highest SNR,
decides offloading ratio based on edge-MEC availability, and allocates
bandwidth/compute proportionally.  No learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P3.resource_mgmt.task_offloader import (
    QueueState, simulate_offloading, select_source_buoys,
    find_local_candidates, find_edge_candidates,
)
from P3.resource_mgmt.metrics import aggregate_results


class GreedyAllocator:
    """Greedy one-shot resource allocator."""

    def __init__(self, n_agents: int, cfg: EnvConfig):
        self.n_agents = n_agents
        self.cfg = cfg

    def run_episode(
        self, env: MarineIoTEnv, n_windows: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng, cfg.source_activation_ratio)
        queue = QueueState()

        ep_T, ep_E, ep_G, ep_suc = [], [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            actions = self._greedy_actions(env, source_ids)
            results = simulate_offloading(env, cfg, source_ids, actions, queue)
            m = aggregate_results(results, cfg.Gamma_max)
            ep_T.append(m["mean_T_total"])
            ep_E.append(m["mean_E_total"])
            ep_G.append(m["mean_Gamma"])
            ep_suc.append(m["success_rate"])

            actions_env = np.ones((len(env.nodes), 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()

        return {
            "mean_T_total": float(np.mean(ep_T)),
            "mean_E_total": float(np.mean(ep_E)),
            "mean_Gamma": float(np.mean(ep_G)),
            "success_rate": float(np.mean(ep_suc)),
        }

    def _greedy_actions(self, env, source_ids):
        cfg = self.cfg
        actions: Dict[int, Dict] = {}
        for bid in source_ids:
            cands = find_local_candidates(env, bid, cfg)
            if not cands:
                continue
            lid = cands[0][0]
            e_cands = find_edge_candidates(env, lid, cfg)
            eid = e_cands[0][0] if e_cands else -1
            alpha = 0.5 if eid >= 0 else 0.0
            actions[bid] = {
                "local_id": lid,
                "edge_id": eid,
                "alpha_off": alpha,
                "bw_frac": 0.8,
                "f_frac": 0.8,
                "omega_comm": 0.4,
                "omega_comp": 0.4,
                "omega_rem": 0.2,
            }
        return actions
