"""
Closed-loop task-offloading simulation for Research Content 3.

Implements the deterministic fluid-queue delay and energy models from
Manuscript III.  Each source buoy's observation data traverses:

    buoy b  ──►  local ship/UAV l  ──►  (optional MEC e)  ──►  satellite s  ──►  base station g

All formulas reference the final manuscript:
  R_ij   = b_ij · log₂(1 + SNR_ij)   [orthogonal channel access]
  T_edge = T_upload + max(T_local_path, T_edge_path) + T_backhaul
  E_total = Σ E_tx + E_rx + E_comp + E_queue
  Success iff T_total ≤ T_max AND E_total ≤ E_max
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.phy import shannon_rate, LinkPHY


# ═══════════════════════════════════════════════════════════════════════════
# Per-task result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OffloadResult:
    buoy_id: int
    local_id: int
    edge_id: int            # -1 if no edge offload
    alpha_off: float        # offloading ratio
    T_total: float          # end-to-end delay (s)
    E_total: float          # end-to-end energy (J)
    Gamma: float            # throughput (bit/s)
    success: bool           # T ≤ T_max and E ≤ E_max
    T_upload: float = 0.0
    T_local_path: float = 0.0
    T_edge_path: float = 0.0
    T_backhaul: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Queue state tracker
# ═══════════════════════════════════════════════════════════════════════════

class QueueState:
    """Tracks fluid-queue backlogs per node for delay computation.

    Supports per-node storage capacity partitioning via omega weights
    (Manuscript III eq 2-36):
        omega_tx + omega_c + omega_r = 1
    where each fraction determines the buffer capacity for communication,
    computation, and result queues respectively.
    """

    def __init__(self):
        self.Q_comm: Dict[Tuple[int, int], float] = {}
        self.Q_comp: Dict[int, float] = {}
        self.Q_rem: Dict[int, float] = {}
        self.cap_comm: Dict[int, float] = {}
        self.cap_comp: Dict[int, float] = {}

    def reset(self):
        self.Q_comm.clear()
        self.Q_comp.clear()
        self.Q_rem.clear()
        self.cap_comm.clear()
        self.cap_comp.clear()

    def set_capacity(self, node_id: int, S_max: float,
                     omega_tx: float, omega_c: float):
        """Set buffer capacities from storage partition weights."""
        self.cap_comm[node_id] = S_max * omega_tx
        self.cap_comp[node_id] = S_max * omega_c

    def add_comm(self, src: int, dst: int, bits: float):
        key = (src, dst)
        self.Q_comm[key] = self.Q_comm.get(key, 0.0) + bits

    def drain_comm(self, src: int, dst: int, rate: float, dt: float):
        key = (src, dst)
        drained = min(self.Q_comm.get(key, 0.0), rate * dt)
        self.Q_comm[key] = max(0.0, self.Q_comm.get(key, 0.0) - drained)

    def add_comp(self, node_id: int, bits: float):
        self.Q_comp[node_id] = self.Q_comp.get(node_id, 0.0) + bits

    def get_comm_delay(self, src: int, dst: int, rate: float) -> float:
        if rate <= 0:
            return 1e6
        return self.Q_comm.get((src, dst), 0.0) / rate

    def get_comp_delay(self, node_id: int, mu: float) -> float:
        if mu <= 0:
            return 1e6
        return self.Q_comp.get(node_id, 0.0) / mu

    def overflow_penalty(self, node_id: int) -> float:
        """Return a normalised [0,1] penalty if queue backlogs exceed capacity."""
        cap_c = self.cap_comm.get(node_id, float("inf"))
        cap_p = self.cap_comp.get(node_id, float("inf"))
        total_comm = sum(v for (_, d), v in self.Q_comm.items() if d == node_id)
        total_comp = self.Q_comp.get(node_id, 0.0)
        ov_c = max(0.0, total_comm - cap_c) / max(cap_c, 1.0)
        ov_p = max(0.0, total_comp - cap_p) / max(cap_p, 1.0)
        return min(ov_c + ov_p, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: find candidate MEC nodes for a buoy
# ═══════════════════════════════════════════════════════════════════════════

def find_local_candidates(
    env: MarineIoTEnv,
    buoy_id: int,
    cfg: EnvConfig,
) -> List[Tuple[int, float]]:
    """Return (node_id, snr) for ship/UAV within R_local_buoy of the buoy."""
    buoy = env.nodes[buoy_id]
    gamma_lin = cfg.gamma_link_linear
    results = []
    for nd in env.nodes:
        if nd.node_type not in ("ship", "uav"):
            continue
        lp = env.link_phy.get((buoy_id, nd.node_id))
        if lp is None or lp.snr < gamma_lin:
            continue
        if lp.distance > cfg.R_local_buoy:
            continue
        results.append((nd.node_id, lp.snr))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_edge_candidates(
    env: MarineIoTEnv,
    local_id: int,
    cfg: EnvConfig,
) -> List[Tuple[int, float]]:
    """Return (node_id, snr) for ship/UAV reachable from the local node."""
    gamma_lin = cfg.gamma_link_linear
    results = []
    for nd in env.nodes:
        if nd.node_type not in ("ship", "uav"):
            continue
        if nd.node_id == local_id:
            continue
        lp = env.link_phy.get((local_id, nd.node_id))
        if lp is None or lp.snr < gamma_lin:
            continue
        results.append((nd.node_id, lp.snr))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:4]


def find_best_satellite(
    env: MarineIoTEnv,
    from_id: int,
    cfg: EnvConfig,
) -> Optional[int]:
    """Return the satellite with best signal from from_id, or None.

    Satellite backhaul is pre-established infrastructure, so only a
    very low SNR floor (0.1 linear ≈ -10 dB) is required — the full
    gamma_link threshold applies only to ad-hoc local links.
    """
    sat_snr_floor = 0.1
    best_id, best_sig = None, -1.0
    for nd in env.nodes:
        if nd.node_type != "satellite":
            continue
        lp = env.link_phy.get((from_id, nd.node_id))
        if lp and lp.snr >= sat_snr_floor and lp.p_sig > best_sig:
            best_id = nd.node_id
            best_sig = lp.p_sig
    return best_id


def find_best_ground(
    env: MarineIoTEnv,
    sat_id: int,
    cfg: EnvConfig,
) -> Optional[int]:
    """Return the land station with best signal from satellite."""
    best_id, best_sig = None, -1.0
    for nd in env.nodes:
        if nd.node_type != "land":
            continue
        lp = env.link_phy.get((sat_id, nd.node_id))
        if lp and lp.p_sig > best_sig:
            best_id = nd.node_id
            best_sig = lp.p_sig
    return best_id


# ═══════════════════════════════════════════════════════════════════════════
# Link rate helper
# ═══════════════════════════════════════════════════════════════════════════

def _link_rate(env: MarineIoTEnv, tx: int, rx: int, bw: float) -> float:
    """Shannon rate over link (tx→rx) with given allocated bandwidth.

    Uses SNR (not SINR) because *bw_frac* already models orthogonal
    channel access (FDMA / OFDMA): each link occupies a dedicated
    sub-band, so co-channel interference is avoided by design.
    """
    lp = env.link_phy.get((tx, rx))
    if lp is None:
        return 0.0
    return shannon_rate(bw, lp.snr)


# ═══════════════════════════════════════════════════════════════════════════
# Core offloading simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_offloading(
    env: MarineIoTEnv,
    cfg: EnvConfig,
    source_ids: List[int],
    actions: Dict[int, Dict[str, float]],
    queue: QueueState,
) -> List[OffloadResult]:
    """
    Simulate one round of closed-loop task offloading.

    Parameters
    ----------
    actions : dict  buoy_id -> {
        "local_id": int,   # assigned local node
        "edge_id":  int,   # assigned edge MEC (-1 = none)
        "alpha_off": float, # offloading ratio [0,1]
        "bw_frac":  float, # bandwidth fraction [0,1]
        "f_frac":   float, # compute fraction [0,1]
        "omega_comm": float, # storage partition: comm buffer fraction
        "omega_comp": float, # storage partition: comp buffer fraction
        "omega_rem":  float, # storage partition: result buffer fraction
    }
    """
    ceilings = cfg.scaled_ceilings()
    M_b = cfg.M_b
    M_res = cfg.gamma_cmp * M_b
    results: List[OffloadResult] = []

    for bid in source_ids:
        FAIL_T = 10.0 * cfg.T_max
        FAIL_E = 10.0 * cfg.E_max

        act = actions.get(bid)
        if act is None:
            results.append(OffloadResult(
                buoy_id=bid, local_id=-1, edge_id=-1,
                alpha_off=0.0, T_total=FAIL_T, E_total=FAIL_E,
                Gamma=0.0, success=False))
            continue

        lid = act["local_id"]
        eid = act["edge_id"]
        alpha = float(np.clip(act["alpha_off"], 0.0, 1.0))
        bw_frac = float(np.clip(act["bw_frac"], 0.01, 1.0))
        f_frac = float(np.clip(act["f_frac"], 0.01, 1.0))

        raw_oc = max(act.get("omega_comm", 0.33), 0.01)
        raw_op = max(act.get("omega_comp", 0.33), 0.01)
        raw_or = max(act.get("omega_rem", 0.34), 0.01)
        omega_sum = raw_oc + raw_op + raw_or
        omega_comm = raw_oc / omega_sum
        omega_comp = raw_op / omega_sum

        local_type = env.nodes[lid].node_type if lid >= 0 else "ship"
        ceil_l = ceilings.get(local_type, ceilings["ship"])
        B_l = bw_frac * ceil_l["B_max"]
        F_l = f_frac * ceil_l["F_max"]
        S_l = ceil_l["S_max"]

        if lid >= 0:
            queue.set_capacity(lid, S_l, omega_comm, omega_comp)

        # ── Upload: buoy → local ────────────────────────────────────
        bw_buoy = bw_frac * cfg.eta_B * cfg.B_0_buoy
        R_b_l = _link_rate(env, bid, lid, bw_buoy)
        if R_b_l <= 0:
            results.append(OffloadResult(
                buoy_id=bid, local_id=lid, edge_id=eid,
                alpha_off=alpha, T_total=FAIL_T, E_total=FAIL_E,
                Gamma=0.0, success=False))
            continue

        T_q_bl = queue.get_comm_delay(bid, lid, R_b_l)
        T_tx_bl = M_b / R_b_l
        T_upload = T_q_bl + T_tx_bl

        P_tx_b = cfg.tx_power_w("buoy")
        E_tx_bl = P_tx_b * T_tx_bl
        E_rx_lb = cfg.P_rx * T_tx_bl

        # ── Validate edge path feasibility first ──────────────────
        T_edge_path = 0.0
        E_edge_branch = 0.0
        edge_ok = False

        if alpha > 0 and eid >= 0:
            edge_type = env.nodes[eid].node_type
            ceil_e = ceilings.get(edge_type, ceilings["ship"])
            B_e = bw_frac * ceil_e["B_max"]
            F_e = f_frac * ceil_e["F_max"]
            R_l_e = _link_rate(env, lid, eid, B_l)
            sat_e = find_best_satellite(env, eid, cfg)

            if R_l_e > 0 and sat_e is not None:
                R_e_s = _link_rate(env, eid, sat_e, cfg.B_0_sat)
                if R_e_s > 0:
                    edge_ok = True

            if not edge_ok:
                alpha = 0.0
                eid = -1

        # ── Local compute branch ────────────────────────────────────
        local_data = (1.0 - alpha) * M_b
        T_q_comp_l = queue.get_comp_delay(lid, F_l / cfg.c_v)
        T_comp_l = local_data * cfg.c_v / max(F_l, 1.0) if local_data > 0 else 0.0
        E_comp_l = cfg.kappa * (F_l ** 2) * local_data * cfg.c_v

        # Local → satellite
        sat_id = find_best_satellite(env, lid, cfg)
        gnd_id = find_best_ground(env, sat_id, cfg) if sat_id is not None else None

        if sat_id is None or gnd_id is None:
            results.append(OffloadResult(
                buoy_id=bid, local_id=lid, edge_id=eid,
                alpha_off=alpha, T_total=FAIL_T, E_total=FAIL_E,
                Gamma=0.0, success=False))
            continue

        R_l_s = _link_rate(env, lid, sat_id, cfg.B_0_sat)
        local_M_res = (1.0 - alpha) * M_res
        T_tx_ls = local_M_res / max(R_l_s, 1.0) if local_M_res > 0 else 0.0

        T_local_path = T_q_comp_l + T_comp_l + T_tx_ls

        E_tx_ls = cfg.tx_power_w(local_type) * T_tx_ls
        E_rx_sl_local = cfg.P_rx * T_tx_ls

        # ── Edge compute branch (validated above) ────────────────────
        if edge_ok:
            edge_data = alpha * M_b
            edge_M_res = alpha * M_res
            T_tx_le = edge_data / R_l_e
            T_comp_e = edge_data * cfg.c_v / max(F_e, 1.0)
            T_tx_es = edge_M_res / R_e_s
            T_edge_path = T_tx_le + T_comp_e + T_tx_es

            E_tx_le = cfg.tx_power_w(local_type) * T_tx_le
            E_rx_el = cfg.P_rx * T_tx_le
            E_comp_e = cfg.kappa * (F_e ** 2) * edge_data * cfg.c_v
            E_tx_es = cfg.tx_power_w(edge_type) * T_tx_es
            E_rx_se_edge = cfg.P_rx * T_tx_es
            E_edge_branch = E_tx_le + E_rx_el + E_comp_e + E_tx_es + E_rx_se_edge

        elif alpha > 0 and eid < 0:
            T_local_path += alpha * M_b * cfg.c_v / max(F_l, 1.0)
            T_tx_ls_extra = alpha * M_res / max(R_l_s, 1.0)
            T_local_path += T_tx_ls_extra
            E_comp_l += cfg.kappa * (F_l ** 2) * alpha * M_b * cfg.c_v
            E_tx_ls += cfg.tx_power_w(local_type) * T_tx_ls_extra

        # ── Backhaul: satellite → base station ──────────────────────
        R_s_g = _link_rate(env, sat_id, gnd_id, cfg.B_0_sat)
        T_tx_sg = M_res / max(R_s_g, 1.0)
        T_q_sg = queue.get_comm_delay(sat_id, gnd_id, R_s_g)
        T_backhaul = T_q_sg + T_tx_sg

        E_tx_sg = cfg.tx_power_w("satellite") * T_tx_sg
        E_rx_gs = cfg.P_rx * T_tx_sg

        # ── Totals ──────────────────────────────────────────────────
        T_total = T_upload + max(T_local_path, T_edge_path) + T_backhaul

        ov_local = queue.overflow_penalty(lid) if lid >= 0 else 0.0
        ov_edge = queue.overflow_penalty(eid) if eid >= 0 and alpha > 0 else 0.0
        T_overflow = (ov_local + ov_edge) * cfg.T_max * 0.5
        T_total += T_overflow

        E_queue = cfg.P_mem * T_total
        E_total = (E_tx_bl + E_rx_lb + E_comp_l + E_tx_ls + E_rx_sl_local +
                   E_edge_branch + E_tx_sg + E_rx_gs + E_queue)

        Gamma = M_b / max(T_total, 1e-9)
        success = (T_total <= cfg.T_max) and (E_total <= cfg.E_max)

        # Update queues
        queue.add_comm(bid, lid, M_b)
        queue.add_comp(lid, (1.0 - alpha) * M_b)
        if eid >= 0 and alpha > 0:
            queue.add_comm(lid, eid, alpha * M_b)
            queue.add_comp(eid, alpha * M_b)

        results.append(OffloadResult(
            buoy_id=bid, local_id=lid, edge_id=eid,
            alpha_off=alpha, T_total=T_total, E_total=E_total,
            Gamma=Gamma, success=success,
            T_upload=T_upload, T_local_path=T_local_path,
            T_edge_path=T_edge_path, T_backhaul=T_backhaul))

    return results


def select_source_buoys(
    nodes, n_src: int, rng: np.random.Generator, source_activation_ratio: float = 1.0,
) -> List[int]:
    """Pick active source buoy IDs for the current window."""
    buoy_ids = [n.node_id for n in nodes if n.node_type == "buoy"]
    if not buoy_ids:
        return []
    target = max(1, int(round(n_src * float(np.clip(source_activation_ratio, 0.1, 1.0)))))
    target = min(target, len(buoy_ids))
    if len(buoy_ids) <= target:
        return buoy_ids
    return list(rng.choice(buoy_ids, size=target, replace=False))
