"""
ACO baseline for MEC-aware link-selection.

Pheromone trails on first-hop choices per source buoy.  Each ant builds a
path by selecting hops proportional to pheromone * heuristic (hop-level LA).
Best path's LA is used for pheromone deposit.
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


class ACOSelector:

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator,
                 n_ants: int = 15, alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.1, q: float = 1.0):
        self.n_agents = n_agents
        self.cfg = cfg
        self.estimator = estimator
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.path_mgr = PathManager(cfg)
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self._pheromone: Dict[Tuple[int, int], float] = defaultdict(lambda: 1.0)
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
        self._pheromone.clear()

        ep_la, ep_q, ep_s = [], [], []
        total_switches = 0
        prev_actions: Dict[int, int] = {}

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr(env)
            self._q_cache = {}

            gamma_lin = cfg.gamma_link_linear
            ships_uavs = [nd.node_id for nd in nodes
                          if nd.node_type in ("ship", "uav")]

            best_la_total = -1.0
            best_actions: Dict[int, int] = {}

            for ant in range(self.n_ants):
                action_dict: Dict[int, int] = {}
                ant_la = []
                for bid in source_ids:
                    candidates = []
                    for cid in ships_uavs:
                        lp = env.link_phy.get((bid, cid))
                        if lp and lp.snr >= gamma_lin:
                            candidates.append(cid)
                    if not candidates:
                        continue

                    probs = np.zeros(len(candidates))
                    for ci, cid in enumerate(candidates):
                        tau = self._pheromone[(bid, cid)] ** self.alpha
                        hq, hs = self._evaluate_hops(env, bid, cid)
                        if hq:
                            h_val = link_advantage(
                                path_quality(hq), path_stability(hs),
                                cfg.w_Q, cfg.w_S) ** self.beta
                        else:
                            h_val = 0.01
                        probs[ci] = tau * h_val
                    s = probs.sum()
                    if s > 0:
                        probs /= s
                    else:
                        probs = np.ones(len(candidates)) / len(candidates)
                    chosen_idx = rng.choice(len(candidates), p=probs)
                    chosen = candidates[chosen_idx]
                    action_dict[bid] = chosen

                    hq, hs = self._evaluate_hops(env, bid, chosen)
                    if hq:
                        ant_la.append(link_advantage(
                            path_quality(hq), path_stability(hs),
                            cfg.w_Q, cfg.w_S))

                mean_la = float(np.mean(ant_la)) if ant_la else 0.0
                if mean_la > best_la_total:
                    best_la_total = mean_la
                    best_actions = dict(action_dict)

                # deposit pheromone
                for bid, cid in action_dict.items():
                    self._pheromone[(bid, cid)] += self.q * mean_la

            # evaporate
            for key in list(self._pheromone.keys()):
                self._pheromone[key] *= (1.0 - self.rho)
                self._pheromone[key] = max(0.01, min(50.0, self._pheromone[key]))

            total_switches += sum(
                1 for b, c in best_actions.items()
                if prev_actions.get(b) is not None and prev_actions[b] != c)
            prev_actions.update(best_actions)
            ep_la.append(best_la_total)

            w_qs, w_ss = [], []
            for bid in source_ids:
                fh = best_actions.get(bid)
                if fh is not None:
                    hq, hs = self._evaluate_hops(env, bid, fh)
                    if hq:
                        w_qs.append(path_quality(hq))
                        w_ss.append(path_stability(hs))
            ep_q.append(float(np.mean(w_qs)) if w_qs else 0.0)
            ep_s.append(float(np.mean(w_ss)) if w_ss else 0.0)

            actions_env = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()
                nodes = env.nodes

        return {"mean_LA": float(np.mean(ep_la)),
                "mean_Q": float(np.mean(ep_q)) if ep_q else 0.0,
                "mean_S": float(np.mean(ep_s)) if ep_s else 0.0,
                "n_switch": total_switches}

    # ─── shared hop evaluation ────────────────────────────────────────

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
            self._q_cache[(tx, rx)] = val
            return val
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
