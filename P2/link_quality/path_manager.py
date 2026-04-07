"""
Service-path enumeration for MEC-aware link selection.

A service path is:  b -> l [-> e] -> s -> g
  b = source buoy
  l = local compute node (ship/UAV within R_local)
  e = optional MEC offload node (second ship/UAV reachable from l)
  s = satellite relay
  g = ground base station (land)

Paths are pruned by K_nbr and K_sat to avoid combinatorial explosion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.nodes import BaseNode
from Env.phy import communication_range_estimate, LinkPHY


@dataclass
class ServicePath:
    """One end-to-end service path from buoy to ground station."""
    buoy_id: int
    local_id: int
    mec_id: Optional[int]   # None when no offload hop
    sat_id: int
    ground_id: int

    @property
    def hops(self) -> List[Tuple[int, int]]:
        path = [(self.buoy_id, self.local_id)]
        if self.mec_id is not None:
            path.append((self.local_id, self.mec_id))
            path.append((self.mec_id, self.sat_id))
        else:
            path.append((self.local_id, self.sat_id))
        path.append((self.sat_id, self.ground_id))
        return path

    @property
    def hop_count(self) -> int:
        return 4 if self.mec_id is not None else 3

    @property
    def node_sequence(self) -> List[int]:
        if self.mec_id is not None:
            return [self.buoy_id, self.local_id, self.mec_id,
                    self.sat_id, self.ground_id]
        return [self.buoy_id, self.local_id, self.sat_id, self.ground_id]

    def __hash__(self):
        return hash(tuple(self.node_sequence))

    def __eq__(self, other):
        if not isinstance(other, ServicePath):
            return False
        return self.node_sequence == other.node_sequence


class PathManager:
    """Enumerate and manage feasible service paths for all source buoys."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def enumerate_paths(self, nodes: List[BaseNode],
                        link_phy: Dict[Tuple[int, int], LinkPHY],
                        source_ids: List[int]) -> Dict[int, List[ServicePath]]:
        """Return {buoy_id: [feasible ServicePaths]} for every source buoy."""
        node_map = {n.node_id: n for n in nodes}

        type_sets = self._classify_nodes(nodes)
        ships_uavs = type_sets["ship"] | type_sets["uav"]
        satellites = type_sets["satellite"]
        grounds = type_sets["land"]

        gamma_lin = self.cfg.gamma_link_linear

        result: Dict[int, List[ServicePath]] = {}
        for bid in source_ids:
            if bid not in node_map:
                result[bid] = []
                continue
            b_node = node_map[bid]

            local_candidates = self._find_local_candidates(
                b_node, ships_uavs, node_map, link_phy, gamma_lin)

            paths: List[ServicePath] = []
            for lid, _ in local_candidates:
                l_node = node_map[lid]

                sat_candidates = self._find_sat_candidates(
                    l_node, satellites, node_map, link_phy, gamma_lin)

                for sid, _ in sat_candidates:
                    gid = self._nearest_ground(
                        node_map[sid], grounds, node_map, link_phy)
                    if gid is not None:
                        paths.append(ServicePath(bid, lid, None, sid, gid))

                mec_candidates = self._find_mec_candidates(
                    l_node, ships_uavs - {lid}, node_map, link_phy, gamma_lin)
                for eid, _ in mec_candidates[:3]:
                    e_node = node_map[eid]
                    sat_from_e = self._find_sat_candidates(
                        e_node, satellites, node_map, link_phy, gamma_lin)
                    for sid, _ in sat_from_e:
                        gid = self._nearest_ground(
                            node_map[sid], grounds, node_map, link_phy)
                        if gid is not None:
                            paths.append(ServicePath(bid, lid, eid, sid, gid))

            result[bid] = paths
        return result

    # ─── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify_nodes(nodes: List[BaseNode]) -> Dict[str, Set[int]]:
        types: Dict[str, Set[int]] = {
            "satellite": set(), "uav": set(), "ship": set(),
            "buoy": set(), "land": set(),
        }
        for n in nodes:
            types.setdefault(n.node_type, set()).add(n.node_id)
        return types

    def _find_local_candidates(
            self, buoy: BaseNode, candidates: Set[int],
            node_map: Dict[int, BaseNode],
            link_phy: Dict[Tuple[int, int], LinkPHY],
            gamma_lin: float) -> List[Tuple[int, float]]:
        """Top K_nbr ships/UAVs reachable from buoy, ranked by P_sig."""
        scored: List[Tuple[int, float]] = []
        for cid in candidates:
            lp = link_phy.get((buoy.node_id, cid))
            if lp is None:
                continue
            if lp.snr < gamma_lin:
                continue
            scored.append((cid, lp.p_sig))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.cfg.K_nbr]

    def _find_mec_candidates(
            self, local_node: BaseNode, candidates: Set[int],
            node_map: Dict[int, BaseNode],
            link_phy: Dict[Tuple[int, int], LinkPHY],
            gamma_lin: float) -> List[Tuple[int, float]]:
        scored: List[Tuple[int, float]] = []
        for cid in candidates:
            lp = link_phy.get((local_node.node_id, cid))
            if lp is None:
                continue
            if lp.snr < gamma_lin:
                continue
            scored.append((cid, lp.p_sig))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.cfg.K_nbr]

    def _find_sat_candidates(
            self, node: BaseNode, satellites: Set[int],
            node_map: Dict[int, BaseNode],
            link_phy: Dict[Tuple[int, int], LinkPHY],
            gamma_lin: float) -> List[Tuple[int, float]]:
        scored: List[Tuple[int, float]] = []
        for sid in satellites:
            lp = link_phy.get((node.node_id, sid))
            if lp is None:
                continue
            if lp.snr < gamma_lin:
                continue
            scored.append((sid, lp.p_sig))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.cfg.K_sat]

    def _nearest_ground(
            self, sat_node: BaseNode, grounds: Set[int],
            node_map: Dict[int, BaseNode],
            link_phy: Dict[Tuple[int, int], LinkPHY]) -> Optional[int]:
        best_id, best_sig = None, -1.0
        for gid in grounds:
            lp = link_phy.get((sat_node.node_id, gid))
            if lp is not None and lp.p_sig > best_sig:
                best_id = gid
                best_sig = lp.p_sig
        return best_id

    # ─── source buoy selection ────────────────────────────────────────

    @staticmethod
    def select_source_buoys(nodes: List[BaseNode], n_src: int,
                            rng: np.random.Generator) -> List[int]:
        buoy_ids = [n.node_id for n in nodes if n.node_type == "buoy"]
        if len(buoy_ids) <= n_src:
            return buoy_ids
        return list(rng.choice(buoy_ids, size=n_src, replace=False))
