"""
MarineIoTEnv — Gymnasium-compatible simulation of an integrated
air-land-sea-space MEC-enabled marine IoT network.

Modes
  discovery        — Chapter 1 (INDP neighbour-discovery)
  link_selection   — Chapter 2 (GMAPPO service-path selection)
  resource_mgmt    — Chapter 3 (improved MATD3 resource optimisation)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np

from .config import EnvConfig
from .nodes import BaseNode, create_node
from .channel import environmental_noise, doppler_shift, C_LIGHT
from .phy import (
    received_signal_power,
    aggregate_interference,
    compute_sinr,
    compute_snr,
    compute_rssi,
    compute_all_links,
    compute_all_links_vectorized,
    compute_phy_matrices,
    compute_gt_snr_matrix_vectorized,
    communication_range_estimate,
    shannon_rate,
    LinkPHY,
    PHYMatrices,
    MatrixBackedLinkDict,
)
from .diagnostics import print_env_config


class MarineIoTEnv(gym.Env):
    """
    Multi-agent environment for the marine IoT MEC architecture.

    Observation / action spaces are set up per *mode* so that the same
    core simulation serves all three research chapters.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None, *,
                 mode: str = "discovery",
                 render_mode: Optional[str] = None,
                 max_steps: int = 500):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.mode = mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0

        # will be populated at reset()
        self.nodes: List[BaseNode] = []
        self._rng: np.random.Generator = np.random.default_rng()
        self.link_phy: Any = {}
        self._phy: Optional[PHYMatrices] = None

        # topology matrices (N x N boolean) — ground truth & discovered
        self._gt_adj: Optional[np.ndarray] = None
        self._disc_adj: Optional[np.ndarray] = None

        # per-window tracking
        self._window_slot = 0
        self._window_contacts: Optional[np.ndarray] = None   # (N, N) int: slot count
        self._window_decodable: Optional[np.ndarray] = None   # (N, N) int

        # F1 cache (invalidated on topology change)
        self._f1_cache: Optional[Tuple[float, int, int, int]] = None

        # discovery protocol handle (pluggable from P1)
        self.discovery_protocol: Any = None

        # renderer (lazy import to avoid pygame dep when training headless)
        self._renderer: Any = None

        # action / observation spaces (set generically; protocols can override)
        n = self.cfg.N_total
        self.n_agents = n
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n, 16), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(n, 2), dtype=np.float32)

        if self.cfg.print_diagnostics:
            print_env_config(self.cfg, enabled=True)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # spawn nodes according to class fractions
        counts = self.cfg.node_counts
        self.nodes = []
        nid = 0
        for cls in ("satellite", "uav", "ship", "buoy", "land"):
            for _ in range(counts.get(cls, 0)):
                node = create_node(nid, cls, self._rng,
                                   self.cfg.area_width, self.cfg.area_height,
                                   self.cfg.sat_altitude)
                node.tx_power = self.cfg.tx_power_w(cls)
                self.nodes.append(node)
                nid += 1

        self.n_agents = len(self.nodes)
        n = self.n_agents

        # topology bookkeeping
        self._gt_adj = np.zeros((n, n), dtype=bool)
        self._disc_adj = np.zeros((n, n), dtype=bool)
        self._f1_cache = None
        self._window_slot = 0
        self._window_contacts = np.zeros((n, n), dtype=np.int32)
        self._window_decodable = np.zeros((n, n), dtype=np.int32)

        self._recompute_phy()
        self._update_ground_truth_topology()

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        dt = self.cfg.delta_t_sim * 1e-3  # convert ms -> s

        # apply actions (continuous: col0 = listen fraction, col1 = tx power fraction)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.n_agents, -1)
        for i, node in enumerate(self.nodes):
            if i < actions.shape[0] and actions.shape[1] >= 2:
                p_max = self.cfg.tx_power_w(node.node_type)
                node.tx_power = float(np.clip(actions[i, 1], 0, 1)) * p_max

        # advance mobility
        for node in self.nodes:
            node.update(dt, self._rng)

        # recompute PHY
        self._recompute_phy()
        self._update_slot_tracking()
        self._step_count += 1

        # check window boundary
        terminated = False
        if self._window_slot >= self.cfg.N_slot:
            self._update_ground_truth_topology()
            self._window_slot = 0
            n = self.n_agents
            self._window_contacts = np.zeros((n, n), dtype=np.int32)
            self._window_decodable = np.zeros((n, n), dtype=np.int32)

        truncated = self._step_count >= self.max_steps

        self._f1_cache = None
        obs = self._build_obs()
        rewards = self._compute_rewards()
        info = self._build_info(rewards)
        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # PHY recomputation
    # ------------------------------------------------------------------

    def _recompute_phy(self):
        self._phy = compute_phy_matrices(self.nodes, self.cfg, self._rng)
        self.link_phy = MatrixBackedLinkDict(self._phy)

    def _update_slot_tracking(self):
        self._window_slot += 1
        m = self._phy
        if m is None:
            return
        gamma_lin = self.cfg.gamma_link_linear
        snr_t = m.SNR.T  # SNR[tx, rx] -> transposed so [rx, tx]
        psig_t = m.P_sig.T
        self._window_decodable += (snr_t >= gamma_lin).astype(np.int32)
        self._window_contacts += (psig_t > 0).astype(np.int32)
        np.fill_diagonal(self._window_decodable, 0)
        np.fill_diagonal(self._window_contacts, 0)

    # ------------------------------------------------------------------
    # Ground-truth topology (channel-state-based)
    # ------------------------------------------------------------------

    def _update_ground_truth_topology(self):
        """
        Compute A_gt using vectorized expected-SNR matrix + contact-duration.

        Per Manuscript I:
          A_gt,ij(w) = 1{ E[SNR_ij(d)] >= gamma_link }

        GT uses fixed reference noise level (gt_eta_N). The same physical
        formulas are preserved; only computation is batched.
        """
        n = self.n_agents
        gt = np.zeros((n, n), dtype=bool)
        if n <= 1:
            self._gt_adj = gt
            return

        gamma_lin = self.cfg.gamma_link_linear
        t_min_s = self.cfg.T_min * 1e-3  # ms -> s

        # Directed expected SNR matrix at GT noise scale, then symmetrize by max.
        snr_dir = compute_gt_snr_matrix_vectorized(
            self.nodes, self.cfg, step_count=self._step_count, n_samples=4)
        snr_best = np.maximum(snr_dir, snr_dir.T)

        # Candidate undirected edges by channel quality.
        cand = snr_best >= gamma_lin
        np.fill_diagonal(cand, False)

        # Contact-duration filter (kept mathematically consistent with scalar logic).
        positions = np.stack([nd.position for nd in self.nodes])
        velocities = np.stack([nd.velocity for nd in self.nodes])
        rel_v = velocities[:, None, :] - velocities[None, :, :]
        rel_speed = np.linalg.norm(rel_v, axis=-1)
        d_now = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

        idx_i, idx_j = np.where(np.triu(cand, k=1))
        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            if rel_speed[i, j] < 0.1:
                gt[i, j] = True
                gt[j, i] = True
                continue

            r_comm = communication_range_estimate(
                self.nodes[i].node_type, self.nodes[j].node_type, self.cfg)
            if d_now[i, j] > r_comm:
                continue

            margin = max(r_comm - float(d_now[i, j]), 0.0)
            t_est = margin / float(rel_speed[i, j]) if rel_speed[i, j] > 0 else float("inf")
            if t_est >= t_min_s:
                gt[i, j] = True
                gt[j, i] = True

        self._gt_adj = gt

    def _channel_snr(self, tx_node: "BaseNode", rx_node: "BaseNode") -> float:
        """Compute the expected SNR from tx to rx at max power.

        Uses a deterministic sub-RNG seeded from the node-pair IDs so that
        repeated calls for the same pair within a topology evaluation
        produce the same result.
        """
        from .channel import compute_path_loss, fading_gain, environmental_noise
        p_tx = self.cfg.tx_power_w(tx_node.node_type)
        g_tx, _ = self.cfg.antenna_gains(tx_node.node_type)
        _, g_rx = self.cfg.antenna_gains(rx_node.node_type)

        # Deterministic sub-RNG for GT stability
        pair_seed = (tx_node.node_id * 10007 + rx_node.node_id + self._step_count)
        gt_rng = np.random.default_rng(pair_seed & 0xFFFFFFFF)

        n_samples = 8
        snr_sum = 0.0
        for _ in range(n_samples):
            pl_db = compute_path_loss(
                tx_node.node_type, rx_node.node_type,
                tx_node.position, rx_node.position,
                self.cfg, gt_rng)
            g_fad = fading_gain(tx_node.node_type, rx_node.node_type,
                                self.cfg, gt_rng)
            p_sig = p_tx * g_tx * g_rx * (10.0 ** (-pl_db / 10.0)) * g_fad
            n_env = environmental_noise(self.cfg, rx_node.node_type, gt_rng)
            snr_sum += p_sig / max(n_env, 1e-30)
        return snr_sum / n_samples

    def _contact_duration_ok(self, ni: "BaseNode", nj: "BaseNode",
                             t_min_s: float) -> bool:
        """Check whether the pair remains in range for at least t_min."""
        rel_v = ni.velocity - nj.velocity
        rel_speed = float(np.linalg.norm(rel_v))
        if rel_speed < 0.1:
            return True  # quasi-static pair — contact persists

        d_now = float(np.linalg.norm(ni.position - nj.position))
        r_comm = communication_range_estimate(
            ni.node_type, nj.node_type, self.cfg)

        if d_now > r_comm:
            return False

        # conservative: how long until they move apart by (r_comm - d_now)?
        margin = max(r_comm - d_now, 0.0)
        t_est = margin / rel_speed if rel_speed > 0 else float("inf")
        return t_est >= t_min_s

    def recompute_ground_truth(self):
        """Public method: recompute GT topology from current node positions."""
        self._update_ground_truth_topology()

    def get_ground_truth_topology(self) -> np.ndarray:
        return self._gt_adj.copy()

    def set_discovered_topology(self, adj: np.ndarray):
        self._disc_adj = adj.copy()
        self._f1_cache = None

    def get_discovered_topology(self) -> np.ndarray:
        return self._disc_adj.copy()

    # ------------------------------------------------------------------
    # Topology metrics (F1)
    # ------------------------------------------------------------------

    def compute_f1_topo(self) -> Tuple[float, int, int, int]:
        if self._f1_cache is not None:
            return self._f1_cache
        gt_upper = np.triu(self._gt_adj, k=1)
        disc_sym = self._disc_adj | self._disc_adj.T
        disc_upper = np.triu(disc_sym, k=1)
        tp = int((gt_upper & disc_upper).sum())
        fp = int((~gt_upper & disc_upper).sum())
        fn = int((gt_upper & ~disc_upper).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        self._f1_cache = (f1, tp, fp, fn)
        return f1, tp, fp, fn

    # ------------------------------------------------------------------
    # Discovery energy
    # ------------------------------------------------------------------

    def compute_discovery_energy(self, node_idx: int,
                                 tx_slots: int, rx_slots: int,
                                 sleep_slots: int, sic_iters: int) -> float:
        dt = self.cfg.delta_t_slot * 1e-3
        node = self.nodes[node_idx]
        p_tx = node.tx_power
        e_tx = tx_slots * p_tx * dt
        e_rx = rx_slots * (self.cfg.P_listen + self.cfg.P_dec) * dt
        e_sleep = sleep_slots * self.cfg.P_sleep * dt
        e_sic = sic_iters * self.cfg.C_ops * self.cfg.kappa * (self.cfg.f_cpu_ND ** 2)
        return e_tx + e_rx + e_sleep + e_sic

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        n = self.n_agents
        obs = np.zeros((n, 16), dtype=np.float32)

        _type_enc = {"satellite": 0, "uav": 1, "ship": 2, "buoy": 3, "land": 4}
        positions = np.stack([nd.position for nd in self.nodes])
        velocities = np.stack([nd.velocity for nd in self.nodes])
        e_res = np.array([nd.energy_residual for nd in self.nodes], dtype=np.float32)
        tx_pow = np.array([nd.tx_power for nd in self.nodes], dtype=np.float32)
        type_ids = np.array([_type_enc.get(nd.node_type, -1) for nd in self.nodes],
                            dtype=np.float32)

        obs[:, 0] = type_ids
        obs[:, 1:4] = positions / 100_000.0
        obs[:, 4:7] = velocities / 100.0
        obs[:, 7] = e_res / 100.0
        obs[:, 8] = tx_pow * 1e3

        if self._disc_adj is not None:
            obs[:, 9] = self._disc_adj.sum(axis=1).astype(np.float32)

        m = self._phy
        if m is not None and m.SINR.shape[0] == n:
            sinr_incoming = m.SINR.copy()
            np.fill_diagonal(sinr_incoming, np.nan)
            obs[:, 10] = np.nanmean(sinr_incoming, axis=0).astype(np.float32)
            obs[:, 11] = np.nanmax(sinr_incoming, axis=0).astype(np.float32)
            i_per_rx = m.I_matrix.sum(axis=0)
            obs[:, 12] = np.minimum(i_per_rx * 1e10, 100.0).astype(np.float32)

        obs[:, 13] = float(self._window_slot) / max(self.cfg.N_slot, 1)
        return obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> np.ndarray:
        if self.mode == "discovery":
            f1, tp, fp, fn = self.compute_f1_topo()
            return np.full(self.n_agents, f1, dtype=np.float32)
        return np.zeros(self.n_agents, dtype=np.float32)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _build_info(self, rewards: Optional[np.ndarray] = None) -> Dict:
        f1, tp, fp, fn = self.compute_f1_topo()
        return {
            "step": self._step_count,
            "window_slot": self._window_slot,
            "f1_topo": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "n_nodes": self.n_agents,
            "node_counts": self.cfg.node_counts,
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return
        if self._renderer is None:
            from .renderer import GameRenderer
            self._renderer = GameRenderer(self.cfg)
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
