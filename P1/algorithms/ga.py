"""
Genetic Algorithm (GA) for INDP action optimisation.

Maintains a population of candidate joint-action vectors.
Fitness = w1 * F1_topo - w2 * mean_E_ND.
Tournament selection, BLX-alpha crossover, Gaussian mutation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from Env.config import EnvConfig


class GAOptimizer:

    def __init__(self, n_agents: int, cfg: Optional[EnvConfig] = None,
                 pop_size: int = 30, n_generations: int = 10,
                 mutation_std: float = 0.1, crossover_alpha: float = 0.3,
                 tournament_k: int = 3, w1: float = 1.0, w2: float = 0.1):
        self.n_agents = n_agents
        self.cfg = cfg or EnvConfig()
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.mut_std = mutation_std
        self.cx_alpha = crossover_alpha
        self.tourn_k = tournament_k
        self.w1 = w1
        self.w2 = w2

    def _init_population(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.05, 0.95, size=(self.pop_size, self.n_agents, 2)).astype(np.float32)

    def _tournament(self, fitness: np.ndarray, rng: np.random.Generator) -> int:
        candidates = rng.choice(len(fitness), size=self.tourn_k, replace=False)
        return int(candidates[np.argmax(fitness[candidates])])

    def _crossover(self, p1: np.ndarray, p2: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        alpha = self.cx_alpha
        lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
        hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
        child = rng.uniform(lo, hi).astype(np.float32)
        return np.clip(child, 0.0, 1.0)

    def _mutate(self, ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0, self.mut_std, size=ind.shape).astype(np.float32)
        return np.clip(ind + noise, 0.0, 1.0)

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
            pop = self._init_population(rng)
            fitness = np.zeros(self.pop_size, dtype=np.float32)

            for gen in range(self.n_gen):
                for p_idx in range(self.pop_size):
                    actions = pop[p_idx]
                    result = protocol.run_window(nodes, cfg, rng,
                                                 [actions] * cfg.N_slot)
                    gt = env.get_ground_truth_topology()
                    f1, *_ = protocol.compute_f1(gt, n)
                    energy = protocol.mean_energy(cfg)
                    fitness[p_idx] = self.w1 * f1 - self.w2 * energy

                # breed next generation
                new_pop = np.zeros_like(pop)
                # elitism: keep best
                elite_idx = np.argmax(fitness)
                new_pop[0] = pop[elite_idx]
                for c in range(1, self.pop_size):
                    i1 = self._tournament(fitness, rng)
                    i2 = self._tournament(fitness, rng)
                    child = self._crossover(pop[i1], pop[i2], rng)
                    child = self._mutate(child, rng)
                    new_pop[c] = child
                pop = new_pop

            # evaluate the best individual
            best = pop[0]
            result = protocol.run_window(nodes, cfg, rng,
                                         [best] * cfg.N_slot)
            env.set_discovered_topology(result["disc_adj"])
            gt = env.get_ground_truth_topology()
            f1, *_ = protocol.compute_f1(gt, n)
            ep_f1.append(f1)
            ep_e.append(result["mean_energy"])
            obs, _, term, trunc, info = env.step(best)
            if term or trunc:
                obs, info = env.reset()

        return {"mean_f1": float(np.mean(ep_f1)),
                "mean_energy": float(np.mean(ep_e))}
