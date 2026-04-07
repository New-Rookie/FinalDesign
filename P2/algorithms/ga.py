"""
GA baseline for MEC-aware link-selection.

Population of joint first-hop assignments (one per source buoy).
Fitness = mean LA_pi across all source paths.
Tournament selection, uniform crossover, random-swap mutation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class as get_link_class
from Env.phy import communication_range_estimate

from P2.link_quality.metrics import (
    compute_let, compute_p_surv, compute_s_ho,
    path_quality, path_stability, link_advantage, compute_lqi,
)
from P2.link_quality.path_manager import PathManager
from P2.link_quality.rf_estimator import LinkQualityEstimator


class GASelector:

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator,
                 pop_size: int = 20, n_generations: int = 8,
                 mutation_rate: float = 0.15, tournament_k: int = 3):
        self.n_agents = n_agents
        self.cfg = cfg
        self.estimator = estimator
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.mut_rate = mutation_rate
        self.tourn_k = tournament_k
        self.path_mgr = PathManager(cfg)
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self._q_cache: Dict[Tuple[int, int], float] = {}
        self._comm_range_cache: Dict[Tuple[str, str], float] = {}

    def run_episode(self, env: MarineIoTEnv, n_windows: int = 10,
                    rng: Optional[np.random.Generator] = None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        nodes = env.nodes
        n = len(nodes)
        source_ids = PathManager.select_source_buoys(nodes, cfg.N_src, rng)
        self._sinr_histories.clear()

        ep_la, ep_q, ep_s = [], [], []
        total_switches = 0
        prev_actions: Dict[int, int] = {}

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr(env)
            self._q_cache = {}

            gamma_lin = cfg.gamma_link_linear
            candidate_map: Dict[int, List[int]] = {}
            ships_uavs = [nd.node_id for nd in nodes
                          if nd.node_type in ("ship", "uav")]
            for bid in source_ids:
                cands = []
                for cid in ships_uavs:
                    lp = env.link_phy.get((bid, cid))
                    if lp and lp.snr >= gamma_lin:
                        cands.append(cid)
                candidate_map[bid] = cands

            pop = self._init_population(source_ids, candidate_map, rng)
            fitness = np.zeros(self.pop_size, dtype=np.float32)

            for gen in range(self.n_gen):
                for pi in range(self.pop_size):
                    fitness[pi] = self._evaluate_individual(
                        env, source_ids, pop[pi])

                new_pop = [None] * self.pop_size
                elite = int(np.argmax(fitness))
                new_pop[0] = pop[elite].copy()
                for ci in range(1, self.pop_size):
                    i1 = self._tournament(fitness, rng)
                    i2 = self._tournament(fitness, rng)
                    child = self._crossover(pop[i1], pop[i2],
                                            source_ids, rng)
                    child = self._mutate(child, source_ids,
                                         candidate_map, rng)
                    new_pop[ci] = child
                pop = new_pop

            best = pop[int(np.argmax(fitness))]
            best_la = self._evaluate_individual(env, source_ids, best)
            ep_la.append(best_la)

            w_qs, w_ss = [], []
            for bid in source_ids:
                fh = best.get(bid)
                if fh is not None:
                    hq, hs = self._evaluate_hops(env, bid, fh)
                    if hq:
                        w_qs.append(path_quality(hq))
                        w_ss.append(path_stability(hs))
            ep_q.append(float(np.mean(w_qs)) if w_qs else 0.0)
            ep_s.append(float(np.mean(w_ss)) if w_ss else 0.0)

            total_switches += sum(
                1 for bid in source_ids
                if prev_actions.get(bid) is not None and
                prev_actions[bid] != best.get(bid))
            prev_actions.update(best)

            actions_env = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()
                nodes = env.nodes

        return {"mean_LA": float(np.mean(ep_la)),
                "mean_Q": float(np.mean(ep_q)) if ep_q else 0.0,
                "mean_S": float(np.mean(ep_s)) if ep_s else 0.0,
                "n_switch": total_switches}

    def _init_population(self, source_ids, candidate_map, rng):
        pop = []
        for _ in range(self.pop_size):
            ind: Dict[int, int] = {}
            for bid in source_ids:
                cands = candidate_map.get(bid, [])
                if cands:
                    ind[bid] = int(rng.choice(cands))
            pop.append(ind)
        return pop

    def _tournament(self, fitness, rng):
        candidates = rng.choice(len(fitness), size=self.tourn_k, replace=False)
        return int(candidates[np.argmax(fitness[candidates])])

    def _crossover(self, p1, p2, source_ids, rng):
        child: Dict[int, int] = {}
        for bid in source_ids:
            if rng.random() < 0.5:
                child[bid] = p1.get(bid, p2.get(bid, bid))
            else:
                child[bid] = p2.get(bid, p1.get(bid, bid))
        return child

    def _mutate(self, ind, source_ids, candidate_map, rng):
        for bid in source_ids:
            if rng.random() < self.mut_rate:
                cands = candidate_map.get(bid, [])
                if cands:
                    ind[bid] = int(rng.choice(cands))
        return ind

    def _evaluate_individual(self, env, source_ids, ind):
        las = []
        for bid in source_ids:
            fh = ind.get(bid)
            if fh is None:
                continue
            hq, hs = self._evaluate_hops(env, bid, fh)
            if hq:
                las.append(link_advantage(
                    path_quality(hq), path_stability(hs),
                    self.cfg.w_Q, self.cfg.w_S))
        return float(np.mean(las)) if las else 0.0

    # ─── hop evaluation ───────────────────────────────────────────────

    def _evaluate_hops(self, env, bid, first_hop):
        nodes = env.nodes
        hq, hs = [], []
        lp = env.link_phy.get((bid, first_hop))
        if not lp:
            return [], []
        lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
        hq.append(self._predict_q(lp, lc, bid, first_hop))
        hs.append(self._compute_s_ho(env, bid, first_hop))
        best_sat, best_sig = None, -1.0
        for nd in nodes:
            if nd.node_type != "satellite":
                continue
            lps = env.link_phy.get((first_hop, nd.node_id))
            if lps and lps.snr >= self.cfg.gamma_link_linear and lps.p_sig > best_sig:
                best_sat, best_sig = nd, lps.p_sig
        if not best_sat:
            return [], []
        lps = env.link_phy[(first_hop, best_sat.node_id)]
        lcs = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
        hq.append(self._predict_q(lps, lcs, first_hop, best_sat.node_id))
        hs.append(self._compute_s_ho(env, first_hop, best_sat.node_id))
        best_land, best_sl = None, -1.0
        for nd in nodes:
            if nd.node_type != "land":
                continue
            lpl = env.link_phy.get((best_sat.node_id, nd.node_id))
            if lpl and lpl.p_sig > best_sl:
                best_land, best_sl = nd, lpl.p_sig
        if not best_land:
            return [], []
        lpl = env.link_phy[(best_sat.node_id, best_land.node_id)]
        lcl = get_link_class(best_sat.node_type, best_land.node_type)
        hq.append(self._predict_q(lpl, lcl, best_sat.node_id, best_land.node_id))
        hs.append(self._compute_s_ho(env, best_sat.node_id, best_land.node_id))
        return hq, hs

    def _predict_q(self, lp, lc, tx, rx):
        cached = self._q_cache.get((tx, rx))
        if cached is not None:
            return cached
        if not self.estimator.is_trained:
            sig = max(lp.snr, 1e-30)
            ber = 0.5 * math.erfc(math.sqrt(sig))
            exp = min(int(self.cfg.L_pkt), 64)
            val = max(1e-6, min(1.0, (1.0 - ber) ** exp))
        else:
            hist = self._sinr_histories.get((tx, rx), [lp.snr])
            sa = np.array(hist[-10:])
            val = self.estimator.predict_single(
                lc, float(lp.rssi), float(lp.snr), float(lp.sinr),
                compute_lqi(lp.sinr), float(np.mean(sa)), float(np.std(sa)),
                float(lp.rssi), 0.0, float(lp.doppler), 0, len(hist))
        self._q_cache[(tx, rx)] = val
        return val

    def _get_comm_range(self, type_i: str, type_j: str) -> float:
        key = (type_i, type_j)
        if key not in self._comm_range_cache:
            self._comm_range_cache[key] = communication_range_estimate(
                type_i, type_j, self.cfg)
        return self._comm_range_cache[key]

    def _compute_s_ho(self, env, tx_id, rx_id):
        tx_n, rx_n = env.nodes[tx_id], env.nodes[rx_id]
        dp = rx_n.position - tx_n.position
        dv = rx_n.velocity - tx_n.velocity
        r = self._get_comm_range(tx_n.node_type, rx_n.node_type)
        let = compute_let(dp, dv, r)
        hist = self._sinr_histories.get((tx_id, rx_id), [])
        if len(hist) < 2:
            lp = env.link_phy.get((tx_id, rx_id))
            v = lp.snr if lp else 1.0
            hist = [v, v]
        p = compute_p_surv(np.array(hist[-10:]), self.cfg.gamma_ho_linear,
                           self.cfg.N_p, self.cfg.delta_t_sim * 1e-3)
        return compute_s_ho(let, p, self.cfg.tau_req * 1e-3)

    def _update_sinr(self, env):
        m = getattr(env, '_phy', None)
        if m is None:
            return
        N = m.SNR.shape[0]
        tx_indices = np.where(m.tx_active)[0]
        node_ids = m.node_ids
        snr_matrix = m.SNR
        for ti in tx_indices:
            tx_id = int(node_ids[ti])
            row = snr_matrix[ti]
            for rj in range(N):
                if ti == rj:
                    continue
                rx_id = int(node_ids[rj])
                key = (tx_id, rx_id)
                hist = self._sinr_histories[key]
                hist.append(float(row[rj]))
                if len(hist) > 20:
                    del hist[:-20]
