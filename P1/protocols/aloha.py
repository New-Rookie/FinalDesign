"""
Slotted ALOHA — simplest baseline for neighbour discovery.

Each node transmits a beacon in each slot with probability p_aloha.
No SIC, no CFAR, no memory — purely probabilistic access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.nodes import BaseNode
from Env.phy import received_signal_power, environmental_noise, compute_phy_matrices


@dataclass
class ALOHANodeState:
    node_id: int
    lnt: Set[int] = field(default_factory=set)
    tx_slots: int = 0
    rx_slots: int = 0
    collisions: int = 0


class ALOHAProtocol:
    """Pure slotted ALOHA neighbour discovery."""

    def __init__(self, cfg: EnvConfig, p_aloha: float = 0.1):
        self.cfg = cfg
        self.p_aloha = p_aloha
        self.states: Dict[int, ALOHANodeState] = {}

    def init_window(self, nodes: List[BaseNode]):
        self.states = {}
        for node in nodes:
            self.states[node.node_id] = ALOHANodeState(node_id=node.node_id)

    def run_slot(self, nodes: List[BaseNode], cfg: EnvConfig,
                 rng: np.random.Generator,
                 actions: Optional[np.ndarray] = None) -> Dict:
        tx_set: List[int] = []
        rx_set: List[int] = []

        for node in nodes:
            st = self.states[node.node_id]
            if rng.random() < self.p_aloha:
                tx_set.append(node.node_id)
                st.tx_slots += 1
                node.tx_power = cfg.tx_power_w(node.node_type)
            else:
                rx_set.append(node.node_id)
                st.rx_slots += 1
                node.tx_power = 0.0

        detections: Dict[int, List[int]] = {nid: [] for nid in rx_set}

        if tx_set and rx_set:
            m = compute_phy_matrices(nodes, cfg, rng)
            id_to_idx = {int(nid): idx for idx, nid in enumerate(m.node_ids)}

            for rx_id in rx_set:
                ri = id_to_idx.get(rx_id)
                if ri is None:
                    continue
                n_env = float(m.N_env[ri])

                powers = []
                for tx_id in tx_set:
                    if tx_id == rx_id:
                        continue
                    ti = id_to_idx.get(tx_id)
                    if ti is None:
                        continue
                    powers.append((tx_id, float(m.P_sig[ti, ri])))

                if not powers:
                    continue

                powers.sort(key=lambda x: x[1], reverse=True)
                strongest_id, strongest_p = powers[0]
                i_rest = sum(p for _, p in powers[1:]) + n_env
                sinr = strongest_p / max(i_rest, 1e-30)

                if sinr >= cfg.gamma_link_linear:
                    detections[rx_id].append(strongest_id)
                else:
                    self.states[rx_id].collisions += 1

        for rx_id, dets in detections.items():
            for tx_id in dets:
                self.states[rx_id].lnt.add(tx_id)

        return {"detections": detections, "tx_set": tx_set, "rx_set": rx_set}

    def build_discovered_topology(self, n_nodes: int) -> np.ndarray:
        adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        for nid, st in self.states.items():
            for nbr in st.lnt:
                if 0 <= nid < n_nodes and 0 <= nbr < n_nodes:
                    adj[nid, nbr] = True
                    adj[nbr, nid] = True
        return adj

    def compute_f1(self, gt_adj: np.ndarray, n_nodes: int) -> Tuple[float, int, int, int]:
        disc = self.build_discovered_topology(n_nodes)
        gt_upper = np.triu(gt_adj, k=1)
        disc_upper = np.triu(disc, k=1)
        tp = int((gt_upper & disc_upper).sum())
        fp = int((~gt_upper & disc_upper).sum())
        fn = int((gt_upper & ~disc_upper).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        return f1, tp, fp, fn

    def compute_energy(self, node_id: int, cfg: EnvConfig) -> float:
        st = self.states.get(node_id)
        if st is None:
            return 0.0
        dt = cfg.delta_t_slot * 1e-3
        e_tx = st.tx_slots * cfg.tx_power_w("ship") * dt
        e_rx = st.rx_slots * cfg.P_listen * dt
        return e_tx + e_rx

    def mean_energy(self, cfg: EnvConfig) -> float:
        if not self.states:
            return 0.0
        return float(np.mean([self.compute_energy(nid, cfg) for nid in self.states]))

    def run_window(self, nodes: List[BaseNode], cfg: EnvConfig,
                   rng: np.random.Generator,
                   actions_per_slot=None) -> Dict:
        self.init_window(nodes)
        for s in range(cfg.N_slot):
            self.run_slot(nodes, cfg, rng)
        return {
            "disc_adj": self.build_discovered_topology(len(nodes)),
            "mean_energy": self.mean_energy(cfg),
        }
