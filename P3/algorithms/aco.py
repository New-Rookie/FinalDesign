"""
ACO baseline for MEC resource management.

Ant Colony Optimisation over the joint (local-node, alpha, bw, compute)
decision space.  Each ant builds a complete resource assignment for all
source buoys, evaluated via the closed-loop offloading simulator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P3.resource_mgmt.task_offloader import (
    QueueState, simulate_offloading, select_source_buoys,
    find_local_candidates, find_edge_candidates,
)
from P3.resource_mgmt.metrics import aggregate_results, compute_reward


ALPHA_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
BW_CHOICES = [0.3, 0.5, 0.7, 0.9]
F_CHOICES = [0.3, 0.5, 0.7, 0.9]


class ACOAllocator:

    def __init__(
        self, n_agents: int, cfg: EnvConfig,
        n_ants: int = 12, alpha_ph: float = 1.0, beta_ph: float = 2.0,
        rho: float = 0.1, q: float = 1.0,
    ):
        self.n_agents = n_agents
        self.cfg = cfg
        self.n_ants = n_ants
        self.alpha_ph = alpha_ph
        self.beta_ph = beta_ph
        self.rho = rho
        self.q = q
        self._pheromone: Dict[Tuple, float] = defaultdict(lambda: 1.0)

    def run_episode(
        self, env: MarineIoTEnv, n_windows: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng, cfg.source_activation_ratio)
        self._pheromone.clear()

        ep_T, ep_E, ep_G, ep_suc = [], [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            cand_map = self._build_candidates(env, source_ids)
            best_cost = -1e9
            best_actions: Dict = {}

            for ant in range(self.n_ants):
                actions = self._ant_solution(source_ids, cand_map, rng)
                queue = QueueState()
                results = simulate_offloading(env, cfg, source_ids, actions, queue)
                m = aggregate_results(results, cfg.Gamma_max)
                cost = compute_reward(m, cfg.T_max, cfg.E_max, cfg.Gamma_max)

                if cost > best_cost:
                    best_cost = cost
                    best_actions = dict(actions)

                for bid, act in actions.items():
                    key = (bid, act["local_id"], round(act["alpha_off"], 2))
                    self._pheromone[key] += self.q * max(cost + 2.0, 0.01)

            for key in list(self._pheromone.keys()):
                self._pheromone[key] *= (1.0 - self.rho)
                self._pheromone[key] = max(0.01, min(50.0, self._pheromone[key]))

            queue = QueueState()
            results = simulate_offloading(env, cfg, source_ids, best_actions, queue)
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

    def _build_candidates(self, env, source_ids):
        cfg = self.cfg
        cand_map: Dict[int, List] = {}
        for bid in source_ids:
            cands = find_local_candidates(env, bid, cfg)
            edge_map: Dict[int, List] = {}
            for lid, _ in cands:
                edge_map[lid] = find_edge_candidates(env, lid, cfg)
            cand_map[bid] = (cands, edge_map)
        return cand_map

    def _ant_solution(self, source_ids, cand_map, rng):
        actions: Dict[int, Dict] = {}
        for bid in source_ids:
            cands, edge_map = cand_map[bid]
            if not cands:
                continue
            probs = np.zeros(len(cands))
            for ci, (lid, snr_val) in enumerate(cands):
                tau = self._pheromone.get((bid, lid, 0.5), 1.0) ** self.alpha_ph
                h_val = max(snr_val, 0.01) ** self.beta_ph
                probs[ci] = tau * h_val
            s = probs.sum()
            probs = probs / s if s > 0 else np.ones(len(cands)) / len(cands)
            idx = rng.choice(len(cands), p=probs)
            lid = cands[idx][0]

            e_cands = edge_map.get(lid, [])
            eid = e_cands[0][0] if e_cands else -1
            alpha = float(rng.choice(ALPHA_CHOICES))
            if eid < 0:
                alpha = 0.0
            bw = float(rng.choice(BW_CHOICES))
            f = float(rng.choice(F_CHOICES))

            actions[bid] = {
                "local_id": lid,
                "edge_id": eid,
                "alpha_off": alpha,
                "bw_frac": bw,
                "f_frac": f,
                "omega_comm": 0.4,
                "omega_comp": 0.4,
                "omega_rem": 0.2,
            }
        return actions
