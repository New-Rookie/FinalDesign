"""
Physical-layer computations — unified across all three research chapters.

Implements the thesis-wide locked formulas for:
  P_sig, RSSI, SNR, SINR, aggregate interference,
  communication range, and Shannon service rate.

Includes both scalar (per-link) and vectorized (batch matrix) implementations.
The vectorized compute_all_links_vectorized() produces the same
Dict[Tuple[int,int], LinkPHY] output as the original compute_all_links().
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .config import EnvConfig, _db2lin, _dbm2w
from .channel import (
    compute_path_loss,
    fading_gain,
    environmental_noise,
    doppler_shift,
    C_LIGHT,
    K_BOLTZMANN,
    T0,
    TYPE_ID,
    build_link_class_masks,
    vectorized_path_loss,
    vectorized_fading,
    vectorized_environmental_noise,
    vectorized_doppler,
)
from .nodes import BaseNode


# ═══════════════════════════════════════════════════════════════════════════
# Per-link signal power (scalar — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def received_signal_power(tx_node: BaseNode, rx_node: BaseNode,
                          cfg: EnvConfig,
                          rng: np.random.Generator,
                          pl_cache: Optional[Dict] = None) -> float:
    """
    P_sig,ij(t) = P_tx,j * G_tx,j * G_rx,i * 10^(-PL_ij/10) * |g_ij|^2
    Returns power in Watts.
    """
    p_tx = tx_node.tx_power if tx_node.tx_power > 0 else cfg.tx_power_w(tx_node.node_type)
    g_tx, _ = cfg.antenna_gains(tx_node.node_type)
    _, g_rx = cfg.antenna_gains(rx_node.node_type)

    key = (tx_node.node_id, rx_node.node_id)
    if pl_cache is not None and key in pl_cache:
        pl_db = pl_cache[key]
    else:
        pl_db = compute_path_loss(tx_node.node_type, rx_node.node_type,
                                  tx_node.position, rx_node.position, cfg, rng)
        if pl_cache is not None:
            pl_cache[key] = pl_db

    g_fading = fading_gain(tx_node.node_type, rx_node.node_type, cfg, rng)
    p_sig = p_tx * g_tx * g_rx * (10.0 ** (-pl_db / 10.0)) * g_fading
    return max(p_sig, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate interference at a receiver (scalar — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_interference(rx_node: BaseNode,
                           all_nodes: List[BaseNode],
                           cfg: EnvConfig,
                           rng: np.random.Generator,
                           exclude_ids: Optional[set] = None,
                           pl_cache: Optional[Dict] = None) -> float:
    """
    I_i(t) = sum_{j != i, j active} P_sig,ji(t)
    """
    exclude = exclude_ids or set()
    interference = 0.0
    for node in all_nodes:
        if node.node_id == rx_node.node_id:
            continue
        if node.node_id in exclude:
            continue
        if not node.is_active or node.tx_power <= 0:
            continue
        interference += received_signal_power(node, rx_node, cfg, rng, pl_cache)
    return interference


# ═══════════════════════════════════════════════════════════════════════════
# RSSI / SNR / SINR (scalar)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rssi(p_sig: float, interference: float, n_env: float) -> float:
    return p_sig + interference + n_env


def compute_snr(p_sig: float, n_env: float) -> float:
    return p_sig / max(n_env, 1e-30)


def compute_sinr(p_sig: float, interference: float, n_env: float) -> float:
    return p_sig / max(interference + n_env, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════
# Full per-link PHY snapshot
# ═══════════════════════════════════════════════════════════════════════════

class LinkPHY:
    __slots__ = ("tx_id", "rx_id", "p_sig", "interference", "n_env",
                 "rssi", "snr", "sinr", "pl_db", "doppler", "distance")

    def __init__(self, tx_id: int, rx_id: int, p_sig: float,
                 interference: float, n_env: float,
                 pl_db: float, doppler: float, distance: float):
        self.tx_id = tx_id
        self.rx_id = rx_id
        self.p_sig = p_sig
        self.interference = interference
        self.n_env = n_env
        self.rssi = compute_rssi(p_sig, interference, n_env)
        self.snr = compute_snr(p_sig, n_env)
        self.sinr = compute_sinr(p_sig, interference, n_env)
        self.pl_db = pl_db
        self.doppler = doppler
        self.distance = distance


class PHYMatrices(NamedTuple):
    """Raw N×N numpy arrays from vectorized PHY computation."""
    P_sig: np.ndarray
    I_matrix: np.ndarray
    SNR: np.ndarray
    SINR: np.ndarray
    N_env: np.ndarray       # (N,)
    PL: np.ndarray
    doppler: np.ndarray
    dist_3d: np.ndarray
    node_ids: np.ndarray    # (N,) int32
    tx_active: np.ndarray   # (N,) bool


class MatrixBackedLinkDict:
    """Dict-like wrapper over PHYMatrices for backward-compatible access.

    Supports ``get((tx_id, rx_id))``, ``__getitem__``, ``items()``,
    and ``values()`` by constructing :class:`LinkPHY` objects on demand.
    """

    __slots__ = ("_m", "_id_to_idx")

    def __init__(self, m: PHYMatrices):
        self._m = m
        self._id_to_idx = {int(nid): idx for idx, nid in enumerate(m.node_ids)}

    def get(self, key, default=None):
        tx_id, rx_id = key
        ti = self._id_to_idx.get(tx_id)
        ri = self._id_to_idx.get(rx_id)
        if ti is None or ri is None or ti == ri:
            return default
        if not self._m.tx_active[ti]:
            return default
        return LinkPHY(
            tx_id, rx_id,
            float(self._m.P_sig[ti, ri]),
            float(self._m.I_matrix[ti, ri]),
            float(self._m.N_env[ri]),
            float(self._m.PL[ti, ri]),
            float(self._m.doppler[ti, ri]),
            float(self._m.dist_3d[ti, ri]),
        )

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def items(self):
        m = self._m
        tx_indices = np.where(m.tx_active)[0]
        N = len(m.node_ids)
        for ti in tx_indices:
            tx_id = int(m.node_ids[ti])
            for rj in range(N):
                if ti == rj:
                    continue
                rx_id = int(m.node_ids[rj])
                yield (tx_id, rx_id), LinkPHY(
                    tx_id, rx_id,
                    float(m.P_sig[ti, rj]),
                    float(m.I_matrix[ti, rj]),
                    float(m.N_env[rj]),
                    float(m.PL[ti, rj]),
                    float(m.doppler[ti, rj]),
                    float(m.dist_3d[ti, rj]),
                )

    def values(self):
        for _, lp in self.items():
            yield lp

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        n_tx = int(self._m.tx_active.sum())
        N = len(self._m.node_ids)
        return n_tx * (N - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Original scalar compute_all_links (preserved for reference / fallback)
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_links(nodes: List[BaseNode], cfg: EnvConfig,
                      rng: np.random.Generator) -> Dict[Tuple[int, int], LinkPHY]:
    """Compute PHY quantities for all ordered pairs where the transmitter is active."""
    pl_cache: Dict = {}
    noise_cache: Dict[int, float] = {}
    links: Dict[Tuple[int, int], LinkPHY] = {}

    for rx in nodes:
        if rx.node_id not in noise_cache:
            noise_cache[rx.node_id] = environmental_noise(cfg, rx.node_type, rng)

    for tx in nodes:
        if not tx.is_active:
            continue
        for rx in nodes:
            if tx.node_id == rx.node_id:
                continue
            p_sig = received_signal_power(tx, rx, cfg, rng, pl_cache)
            i_agg = aggregate_interference(rx, nodes, cfg, rng,
                                           exclude_ids={tx.node_id}, pl_cache=pl_cache)
            n_env = noise_cache[rx.node_id]
            dist = float(np.linalg.norm(tx.position - rx.position))
            f_c = cfg.carrier_freq(tx.node_type, rx.node_type)
            fd = doppler_shift(tx.position, rx.position,
                               tx.velocity, rx.velocity, f_c)
            key_pl = (tx.node_id, rx.node_id)
            pl_db = pl_cache.get(key_pl, 0.0)
            links[(tx.node_id, rx.node_id)] = LinkPHY(
                tx.node_id, rx.node_id, p_sig, i_agg, n_env,
                pl_db, fd, dist)
    return links


# ═══════════════════════════════════════════════════════════════════════════
# Antenna gain lookup tables (vectorized helpers)
# ═══════════════════════════════════════════════════════════════════════════

def _build_gain_tables(cfg: EnvConfig):
    """Pre-build linear gain arrays indexed by TYPE_ID for vectorized ops."""
    type_names = ["satellite", "uav", "ship", "buoy", "land"]
    g_tx = np.zeros(5, dtype=np.float64)
    g_rx = np.zeros(5, dtype=np.float64)
    for name in type_names:
        tid = TYPE_ID[name]
        gt, gr = cfg.antenna_gains(name)
        g_tx[tid] = gt
        g_rx[tid] = gr
    return g_tx, g_rx


def _build_txpower_default(cfg: EnvConfig):
    """Default max Tx power per type (used when node.tx_power == 0)."""
    type_names = ["satellite", "uav", "ship", "buoy", "land"]
    p = np.zeros(5, dtype=np.float64)
    for name in type_names:
        p[TYPE_ID[name]] = cfg.tx_power_w(name)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized compute_all_links — O(N^2) replacement
# ═══════════════════════════════════════════════════════════════════════════

def compute_phy_matrices(
    nodes: List[BaseNode],
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> PHYMatrices:
    """Vectorized O(N^2) PHY computation returning raw numpy matrices."""
    N = len(nodes)
    if N == 0:
        _z2 = np.zeros((0, 0), dtype=np.float64)
        _z1 = np.zeros(0, dtype=np.float64)
        _zi = np.zeros(0, dtype=np.int32)
        _zb = np.zeros(0, dtype=bool)
        return PHYMatrices(_z2, _z2, _z2, _z2, _z1, _z2, _z2, _z2, _zi, _zb)

    positions = np.stack([n.position for n in nodes])
    velocities = np.stack([n.velocity for n in nodes])
    type_ids = np.array([TYPE_ID[n.node_type] for n in nodes], dtype=np.int32)
    node_ids = np.array([n.node_id for n in nodes], dtype=np.int32)

    p_default = _build_txpower_default(cfg)
    tx_powers = np.array([
        n.tx_power if n.tx_power > 0 else p_default[TYPE_ID[n.node_type]]
        for n in nodes
    ], dtype=np.float64)

    active = np.array([n.is_active for n in nodes], dtype=bool)
    tx_active = active & (tx_powers > 0)

    diff = positions[:, None, :] - positions[None, :, :]
    dist_3d = np.linalg.norm(diff, axis=-1)
    dist_2d = np.linalg.norm(diff[:, :, :2], axis=-1)

    mask_sat, mask_uav, mask_sea, mask_terr = build_link_class_masks(type_ids)

    PL = vectorized_path_loss(
        dist_3d, dist_2d, positions, type_ids,
        mask_sat, mask_uav, mask_sea, mask_terr, cfg, rng)

    mask_rician = mask_sat | mask_uav
    mask_rayleigh = mask_sea | mask_terr
    fading_matrix = vectorized_fading(N, mask_rician, mask_rayleigh, cfg, rng)

    g_tx_table, g_rx_table = _build_gain_tables(cfg)
    G_tx = g_tx_table[type_ids]
    G_rx = g_rx_table[type_ids]

    PL_linear = np.power(10.0, -PL / 10.0)
    P_sig = (tx_powers[:, None] * G_tx[:, None] * G_rx[None, :] *
             PL_linear * fading_matrix)
    P_sig = np.maximum(P_sig, 1e-30)
    np.fill_diagonal(P_sig, 0.0)
    P_sig *= tx_active[:, None]

    total_at_rx = P_sig.sum(axis=0)
    I_matrix = total_at_rx[None, :] - P_sig

    N_env = vectorized_environmental_noise(cfg, type_ids, rng)

    N_env_row = N_env[None, :]
    SNR = P_sig / np.maximum(N_env_row, 1e-30)
    SINR = P_sig / np.maximum(I_matrix + N_env_row, 1e-30)

    is_sat = (type_ids == TYPE_ID["satellite"])
    f_c_matrix = np.where(
        is_sat[:, None] | is_sat[None, :],
        cfg.f_c_sat, cfg.f_c_local)
    doppler_matrix = vectorized_doppler(positions, velocities, f_c_matrix)

    return PHYMatrices(P_sig, I_matrix, SNR, SINR, N_env,
                       PL, doppler_matrix, dist_3d, node_ids, tx_active)


def compute_all_links_vectorized(
    nodes: List[BaseNode],
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> "MatrixBackedLinkDict":
    """Vectorized O(N^2) replacement for compute_all_links.

    Returns a :class:`MatrixBackedLinkDict` that is API-compatible with
    ``Dict[Tuple[int,int], LinkPHY]`` but backed by numpy matrices.
    """
    m = compute_phy_matrices(nodes, cfg, rng)
    return MatrixBackedLinkDict(m)


# ═══════════════════════════════════════════════════════════════════════════
# GT topology helper — vectorized expected SNR matrix
# ═══════════════════════════════════════════════════════════════════════════

def compute_gt_snr_matrix_vectorized(
    nodes: List[BaseNode],
    cfg: EnvConfig,
    step_count: int,
    n_samples: int = 8,
) -> np.ndarray:
    """
    Compute E[SNR_ij] for all directed pairs (i,j) in vectorized form.

    Mathematical consistency with scalar GT logic:
      - same path-loss / fading / environmental-noise formulas
      - same tx power and antenna gains
      - same fixed GT noise scale (cfg.gt_eta_N)
      - same Monte Carlo sample count (default 8)

    Returns
    -------
    snr_mean : np.ndarray
        (N, N) matrix where entry [i, j] is expected SNR from tx i to rx j.
    """
    N = len(nodes)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)

    positions = np.stack([n.position for n in nodes])
    type_ids = np.array([TYPE_ID[n.node_type] for n in nodes], dtype=np.int32)

    # Max-power GT convention (independent of node run-time activity)
    p_default = _build_txpower_default(cfg)
    tx_powers = p_default[type_ids]  # (N,)

    # Distance matrices
    diff = positions[:, None, :] - positions[None, :, :]
    dist_3d = np.linalg.norm(diff, axis=-1)
    dist_2d = np.linalg.norm(diff[:, :, :2], axis=-1)

    mask_sat, mask_uav, mask_sea, mask_terr = build_link_class_masks(type_ids)
    mask_rician = mask_sat | mask_uav
    mask_rayleigh = mask_sea | mask_terr

    g_tx_table, g_rx_table = _build_gain_tables(cfg)
    G_tx = g_tx_table[type_ids]
    G_rx = g_rx_table[type_ids]

    # Deterministic RNG for GT stability at each step
    seed_base = (int(step_count) * 2654435761 + 1013904223) & 0xFFFFFFFF

    saved_eta_N = cfg.eta_N
    cfg.eta_N = cfg.gt_eta_N

    snr_sum = np.zeros((N, N), dtype=np.float64)
    try:
        for k in range(max(1, int(n_samples))):
            rng = np.random.default_rng((seed_base + k * 7919) & 0xFFFFFFFF)

            PL = vectorized_path_loss(
                dist_3d, dist_2d, positions, type_ids,
                mask_sat, mask_uav, mask_sea, mask_terr, cfg, rng,
            )
            fad = vectorized_fading(N, mask_rician, mask_rayleigh, cfg, rng)

            PL_linear = np.power(10.0, -PL / 10.0)
            P_sig = tx_powers[:, None] * G_tx[:, None] * G_rx[None, :] * PL_linear * fad
            P_sig = np.maximum(P_sig, 1e-30)
            np.fill_diagonal(P_sig, 0.0)

            N_env = vectorized_environmental_noise(cfg, type_ids, rng)  # per-rx
            snr_sum += P_sig / np.maximum(N_env[None, :], 1e-30)
    finally:
        cfg.eta_N = saved_eta_N

    return snr_sum / max(1, int(n_samples))


# ═══════════════════════════════════════════════════════════════════════════
# Communication range
# ═══════════════════════════════════════════════════════════════════════════

def communication_range_estimate(type_i: str, type_j: str,
                                 cfg: EnvConfig) -> float:
    """
    Rough analytic estimate of R_comm,ij from the unified threshold:
      R_comm = sup{d : E[SINR(d)] >= gamma_link}
    Uses free-space + average fading to give a ballpark range.
    """
    p_tx = cfg.tx_power_w(type_j)
    g_tx, _ = cfg.antenna_gains(type_j)
    _, g_rx = cfg.antenna_gains(type_i)
    f_c = cfg.carrier_freq(type_i, type_j)
    n_env = K_BOLTZMANN * cfg.B_meas * (T0 * cfg.F_rx + 290.0) * cfg.eta_N
    gamma_lin = cfg.gamma_link_linear

    eirp = p_tx * g_tx * g_rx
    wavelength = C_LIGHT / f_c
    d_max = (wavelength / (4 * math.pi)) * math.sqrt(eirp / (gamma_lin * n_env))
    return max(d_max, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Shannon service rate
# ═══════════════════════════════════════════════════════════════════════════

def shannon_rate(bandwidth: float, sinr: float) -> float:
    """R_ij(t) = B * log2(1 + SINR)"""
    if sinr <= 0:
        return 0.0
    return bandwidth * math.log2(1.0 + sinr)


K_BOLTZMANN_REF = K_BOLTZMANN
