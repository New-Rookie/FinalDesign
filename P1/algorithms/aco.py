"""
Ant Colony Optimisation (ACO) for INDP action selection.

The continuous (tau, P_tx) space is quantised into a grid.  Pheromone
trails are maintained on the grid and updated based on the F1_topo
achieved.  Standard AS/MMAS update rules.

V14 fix: energy is now reported for the best ant's actions (the ones
actually applied), not the last random ant's actions.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from Env.config import EnvConfig


class ACOOptimizer:

    def __init__(self, n_agents: int, cfg: Optional[EnvConfig] = None,
                 n_ants: int = 20, n_listen_bins: int = 5,
                 n_power_bins: int = 5, alpha: float = 1.0,
                 beta: float = 2.0, rho: float = 0.1,
                 q: float = 1.0):
        self.n_agents = n_agents
        self.cfg = cfg or EnvConfig()
        self.n_ants = n_ants
        self.alpha = alpha   # pheromone importance
        self.beta = beta     # heuristic importance
        self.rho = rho       # evaporation rate
        self.q = q           # pheromone deposit factor

        self.listen_grid = np.linspace(0.1, 0.9, n_listen_bins)
        self.power_grid = np.linspace(0.1, 1.0, n_power_bins)
        n_choices = n_listen_bins * n_power_bins

        # pheromone per agent, per grid cell
        self.pheromone = np.ones((n_agents, n_choices), dtype=np.float64)
        self._n_choices = n_choices
        self._n_l = n_listen_bins
        self._n_p = n_power_bins

    def _decode(self, idx: int):
        li = idx // self._n_p
        pi = idx % self._n_p
        return self.listen_grid[li], self.power_grid[pi]

    def _select(self, agent_idx: int, rng: np.random.Generator) -> int:
        tau = self.pheromone[agent_idx]
        prob = tau ** self.alpha
        prob /= prob.sum()
        return int(rng.choice(self._n_choices, p=prob))

    def select_actions(self, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self._last_choices = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            c = self._select(i, rng)
            self._last_choices[i] = c
            actions[i, 0], actions[i, 1] = self._decode(c)
        return actions

    def update_pheromone(self, fitness: float):
        self.pheromone *= (1.0 - self.rho)
        deposit = self.q * fitness
        for i in range(self.n_agents):
            self.pheromone[i, self._last_choices[i]] += deposit
        self.pheromone = np.clip(self.pheromone, 0.01, 50.0)

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
            best_f1 = -1
            best_actions = None
            for ant in range(self.n_ants):
                actions = self.select_actions(rng)
                result = protocol.run_window(nodes, cfg, rng,
                                             [actions] * cfg.N_slot)
                gt = env.get_ground_truth_topology()
                f1, *_ = protocol.compute_f1(gt, n)
                if f1 > best_f1:
                    best_f1 = f1
                    best_actions = actions.copy()
                self.update_pheromone(f1)

            result_best = protocol.run_window(nodes, cfg, rng,
                                              [best_actions] * cfg.N_slot)
            env.set_discovered_topology(result_best["disc_adj"])
            ep_f1.append(best_f1)
            ep_e.append(result_best["mean_energy"])
            obs, _, term, trunc, info = env.step(best_actions)
            if term or trunc:
                obs, info = env.reset()

        return {"mean_f1": float(np.mean(ep_f1)),
                "mean_energy": float(np.mean(ep_e))}
