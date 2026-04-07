"""
GA baseline for MEC resource management.

Population of joint continuous-valued resource-allocation vectors.
Fitness = negative weighted delay+energy cost via the closed-loop simulator.
Tournament selection, uniform crossover, Gaussian mutation.
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
from P3.resource_mgmt.metrics import aggregate_results, compute_reward


class GAAllocator:

    def __init__(
        self, n_agents: int, cfg: EnvConfig,
        pop_size: int = 16, n_generations: int = 6,
        mutation_std: float = 0.15, tournament_k: int = 3,
    ):
        self.n_agents = n_agents
        self.cfg = cfg
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.mut_std = mutation_std
        self.tourn_k = tournament_k

    def run_episode(
        self, env: MarineIoTEnv, n_windows: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng, cfg.source_activation_ratio)

        ep_T, ep_E, ep_G, ep_suc = [], [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            struct = self._build_structure(env, source_ids)
            n_src = len(source_ids)
            gene_len = n_src * 3  # alpha, bw, f per source

            pop = rng.random((self.pop_size, gene_len)).astype(np.float32)
            fitness = np.zeros(self.pop_size, dtype=np.float32)

            for gen in range(self.n_gen):
                for pi in range(self.pop_size):
                    fitness[pi] = self._evaluate(
                        env, cfg, source_ids, struct, pop[pi])

                new_pop = np.empty_like(pop)
                elite = int(np.argmax(fitness))
                new_pop[0] = pop[elite].copy()
                for ci in range(1, self.pop_size):
                    i1 = self._tournament(fitness, rng)
                    i2 = self._tournament(fitness, rng)
                    child = self._crossover(pop[i1], pop[i2], rng)
                    child = self._mutate(child, rng)
                    new_pop[ci] = child
                pop = new_pop

            for pi in range(self.pop_size):
                fitness[pi] = self._evaluate(
                    env, cfg, source_ids, struct, pop[pi])
            best = pop[int(np.argmax(fitness))]
            actions = self._decode(source_ids, struct, best)
            queue = QueueState()
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

    def _build_structure(self, env, source_ids):
        cfg = self.cfg
        struct = {}
        for bid in source_ids:
            cands = find_local_candidates(env, bid, cfg)
            if not cands:
                struct[bid] = (None, None)
                continue
            lid = cands[0][0]
            e_cands = find_edge_candidates(env, lid, cfg)
            eid = e_cands[0][0] if e_cands else -1
            struct[bid] = (lid, eid)
        return struct

    def _decode(self, source_ids, struct, genes):
        actions = {}
        for i, bid in enumerate(source_ids):
            lid, eid = struct.get(bid, (None, None))
            if lid is None:
                continue
            alpha = float(genes[i * 3])
            bw = float(np.clip(genes[i * 3 + 1], 0.05, 1.0))
            f = float(np.clip(genes[i * 3 + 2], 0.05, 1.0))
            if eid < 0:
                alpha = 0.0
            actions[bid] = {
                "local_id": lid, "edge_id": eid,
                "alpha_off": alpha, "bw_frac": bw, "f_frac": f,
                "omega_comm": 0.4, "omega_comp": 0.4, "omega_rem": 0.2,
            }
        return actions

    def _evaluate(self, env, cfg, source_ids, struct, genes):
        actions = self._decode(source_ids, struct, genes)
        queue = QueueState()
        results = simulate_offloading(env, cfg, source_ids, actions, queue)
        m = aggregate_results(results, cfg.Gamma_max)
        return compute_reward(m, cfg.T_max, cfg.E_max, cfg.Gamma_max)

    def _tournament(self, fitness, rng):
        idxs = rng.choice(len(fitness), size=self.tourn_k, replace=False)
        return int(idxs[np.argmax(fitness[idxs])])

    def _crossover(self, p1, p2, rng):
        mask = rng.random(len(p1)) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, genes, rng):
        noise = rng.normal(0, self.mut_std, size=len(genes)).astype(np.float32)
        return np.clip(genes + noise, 0.0, 1.0)
