"""
Idle-run probe collection for the RF link-quality estimator.

Runs the environment with no policy optimisation, collecting labelled
(feature-vector, PRR) samples for every active link at each step.
Samples are collected per link class and saved to a pandas DataFrame.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class
from Env.phy import LinkPHY, shannon_rate

from P2.link_quality.metrics import compute_lqi

# Number of recent SINR / RSSI values kept for rolling statistics
_ROLLING_WINDOW = 10


def collect_probes(cfg: EnvConfig, n_steps: int = 400,
                   n_probe_per_class: int = 20_000,
                   seed: int = 0,
                   verbose: bool = True) -> pd.DataFrame:
    """Collect labelled probe samples across all link classes.

    Returns a DataFrame with columns:
        link_class, RSSI, SNR, SINR, LQI, mu_SINR, sigma_SINR,
        mu_RSSI, sigma_RSSI, doppler, type_pair, dwell,
        prr_emp  (label)
    """
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=n_steps + 10)
    obs, _ = env.reset(seed=seed)
    nodes = env.nodes
    n = len(nodes)
    rng = np.random.default_rng(seed)

    sinr_history: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    rssi_history: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    dwell_counter: Dict[Tuple[int, int], int] = defaultdict(int)

    class_counts: Dict[str, int] = defaultdict(int)
    target = n_probe_per_class
    records: List[Dict] = []

    pbar = tqdm(range(n_steps), desc="Probe collection",
                unit="step", leave=True, dynamic_ncols=True,
                disable=not verbose)

    for step in pbar:
        actions = np.ones((n, 2), dtype=np.float32)
        actions[:, 0] = 0.5
        actions[:, 1] = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
        obs, _, term, trunc, _ = env.step(actions)

        for (tx_id, rx_id), lp in env.link_phy.items():
            tx_node = env.nodes[tx_id] if tx_id < n else None
            rx_node = env.nodes[rx_id] if rx_id < n else None
            if tx_node is None or rx_node is None:
                continue

            lc = link_class(tx_node.node_type, rx_node.node_type)
            if class_counts[lc] >= target:
                continue

            key = (tx_id, rx_id)
            sinr_history[key].append(lp.sinr)
            rssi_history[key].append(lp.rssi)
            if len(sinr_history[key]) > _ROLLING_WINDOW:
                sinr_history[key] = sinr_history[key][-_ROLLING_WINDOW:]
                rssi_history[key] = rssi_history[key][-_ROLLING_WINDOW:]

            if lp.snr >= cfg.gamma_link_linear:
                dwell_counter[key] += 1
            else:
                dwell_counter[key] = 0

            if len(sinr_history[key]) < 3:
                continue

            sinr_arr = np.array(sinr_history[key])
            rssi_arr = np.array(rssi_history[key])

            prr_emp = _simulate_prr(lp, cfg, rng, lc=lc)

            type_pair = _encode_type_pair(tx_node.node_type,
                                          rx_node.node_type)

            records.append({
                "link_class": lc,
                "RSSI": float(lp.rssi),
                "SNR": float(lp.snr),
                "SINR": float(lp.sinr),
                "LQI": compute_lqi(lp.sinr),
                "mu_SINR": float(np.mean(sinr_arr)),
                "sigma_SINR": float(np.std(sinr_arr)),
                "mu_RSSI": float(np.mean(rssi_arr)),
                "sigma_RSSI": float(np.std(rssi_arr)),
                "doppler": float(lp.doppler),
                "type_pair": type_pair,
                "dwell": dwell_counter[key],
                "prr_emp": prr_emp,
            })
            class_counts[lc] += 1

        if term or trunc:
            obs, _ = env.reset(seed=seed + step + 1)
            nodes = env.nodes

        all_done = all(class_counts.get(c, 0) >= target
                       for c in ("satellite", "uav_terrestrial",
                                 "sea_surface", "terrestrial"))
        pbar.set_postfix({c[:6]: class_counts.get(c, 0) for c in
                          ("satellite", "uav_terrestrial",
                           "sea_surface", "terrestrial")})
        if all_done:
            break

    pbar.close()
    env.close()
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _effective_lpkt(cfg: EnvConfig) -> int:
    """Effective packet length for PRR computation.

    Real maritime systems use FEC coding (turbo / LDPC) which provides
    significant coding gain.  Using the raw L_pkt=1024 with uncoded BER
    gives (1-BER)^1024 ≈ 0 for any moderate BER, collapsing all PRR
    labels to zero.  Capping at 64 models the coded-packet regime and
    keeps PRR labels consistent with the algorithm-side fallback.
    """
    return min(int(cfg.L_pkt), 64)


def _simulate_prr(lp: LinkPHY, cfg: EnvConfig,
                  rng: np.random.Generator,
                  lc: str = "",
                  n_packets: int = 20) -> float:
    """Simulate PRR by sending n_packets probe packets.

    All link types use SNR (not SINR) because the MEC MAC layer
    (OFDMA / TDMA / NOMA) coordinates access so that the selected
    link operates on a dedicated time-frequency resource with
    negligible co-channel interference.
    """
    sig_quality = max(lp.snr, 1e-30)
    ber = 0.5 * math.erfc(math.sqrt(sig_quality))
    L_eff = _effective_lpkt(cfg)
    pkt_success = (1.0 - ber) ** L_eff
    successes = rng.binomial(n_packets, min(pkt_success, 1.0))
    return successes / n_packets


_TYPE_PAIR_MAP = {
    ("satellite", "satellite"): 0,
    ("satellite", "uav"): 1, ("uav", "satellite"): 1,
    ("satellite", "ship"): 2, ("ship", "satellite"): 2,
    ("satellite", "buoy"): 3, ("buoy", "satellite"): 3,
    ("satellite", "land"): 4, ("land", "satellite"): 4,
    ("uav", "uav"): 5,
    ("uav", "ship"): 6, ("ship", "uav"): 6,
    ("uav", "buoy"): 7, ("buoy", "uav"): 7,
    ("uav", "land"): 8, ("land", "uav"): 8,
    ("ship", "ship"): 9,
    ("ship", "buoy"): 10, ("buoy", "ship"): 10,
    ("ship", "land"): 11, ("land", "ship"): 11,
    ("buoy", "buoy"): 12,
    ("buoy", "land"): 13, ("land", "buoy"): 13,
    ("land", "land"): 14,
}


def _encode_type_pair(type_a: str, type_b: str) -> int:
    return _TYPE_PAIR_MAP.get((type_a, type_b), 15)
