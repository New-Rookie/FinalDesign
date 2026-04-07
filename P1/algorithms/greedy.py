"""
Greedy heuristic for INDP action selection.

Each agent selects the (listen_fraction, tx_power_fraction) that maximises
its immediate one-step local utility, with no learning or lookahead.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Env.config import EnvConfig
from Env.nodes import BaseNode


class GreedyOptimizer:
    """Grid-search greedy over discretised action space per agent."""

    def __init__(self, n_agents: int, cfg: Optional[EnvConfig] = None,
                 n_listen_bins: int = 5, n_power_bins: int = 5):
        self.n_agents = n_agents
        self.cfg = cfg or EnvConfig()
        self.n_listen = n_listen_bins
        self.n_power = n_power_bins
        self._listen_grid = np.linspace(0.2, 0.8, n_listen_bins)
        self._power_grid = np.linspace(0.2, 1.0, n_power_bins)

    def select_actions(self, obs: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        """
        For each agent, pick the action that heuristically maximises
        expected detection (high SINR -> more listen) while balancing energy.
        """
        n = min(self.n_agents, obs.shape[0])
        actions = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            avg_sinr = obs[i, 10]
            nbr_count = obs[i, 9]
            # heuristic: if high SINR environment, listen more; if few
            # neighbours discovered, transmit more
            if nbr_count < 3:
                listen = 0.3
                power = 0.9
            elif avg_sinr > 5:
                listen = 0.7
                power = 0.4
            else:
                listen = 0.5
                power = 0.6
            actions[i, 0] = listen
            actions[i, 1] = power
        return actions

    def run_episode(self, env, protocol, n_windows: int = 10,
                    rng=None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)
        ep_f1, ep_e = [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            actions = self.select_actions(obs, rng)
            result = protocol.run_window(nodes, cfg, rng,
                                         [actions] * cfg.N_slot)
            env.set_discovered_topology(result["disc_adj"])
            gt = env.get_ground_truth_topology()
            f1, *_ = protocol.compute_f1(gt, n)
            ep_f1.append(f1)
            ep_e.append(result["mean_energy"])
            obs, _, term, trunc, info = env.step(actions)
            if term or trunc:
                obs, info = env.reset()

        return {"mean_f1": float(np.mean(ep_f1)),
                "mean_energy": float(np.mean(ep_e))}
