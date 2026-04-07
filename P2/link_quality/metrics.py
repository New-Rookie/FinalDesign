"""
Locked metric formulas for Research Content 2 — MEC-aware link selection.

Hop-level:  LET, P_surv, S_HO, Q_ij (from RF estimator)
Path-level: Q_pi, S_pi, LA_pi
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as _gauss

from Env.config import EnvConfig
from Env.phy import communication_range_estimate


# ═══════════════════════════════════════════════════════════════════════════
# Hop-level metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_let(dp: np.ndarray, dv: np.ndarray, r_comm: float) -> float:
    """Link Expiration Time.

    Solves  ||dp + tau * dv||^2 = R_comm^2  for the smallest positive tau.
    Returns float('inf') when the pair is quasi-static inside range.
    """
    speed_sq = float(np.dot(dv, dv))
    if speed_sq < 1e-6:
        d_now = float(np.linalg.norm(dp))
        return float("inf") if d_now < r_comm else 0.0

    a = speed_sq
    b = 2.0 * float(np.dot(dp, dv))
    c = float(np.dot(dp, dp)) - r_comm * r_comm

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return 0.0

    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)

    positives = [t for t in (t1, t2) if t > 0]
    if not positives:
        return 0.0
    return min(positives)


def compute_p_surv(sinr_seq: np.ndarray, gamma_ho_lin: float,
                   n_p: int, delta_t: float) -> float:
    """Discrete-horizon survival probability.

    mu_gamma(t+n) = gamma(t) + n * slope
    P_surv = prod_{n=1}^{N_p} Phi( (mu - gamma_ho) / sigma )

    *sinr_seq* is a recent history of linear SINR values (newest last).
    *delta_t* is the step interval in seconds.
    """
    if len(sinr_seq) < 2:
        return 0.5

    sinr_db = 10.0 * np.log10(np.clip(sinr_seq, 1e-30, None))
    gamma_ho_db = 10.0 * math.log10(max(gamma_ho_lin, 1e-30))

    diffs = np.diff(sinr_db)
    slope = float(np.mean(diffs)) / delta_t if delta_t > 0 else 0.0
    sigma = max(float(np.std(diffs)), 0.5)

    current_db = float(sinr_db[-1])
    product = 1.0
    for n in range(1, n_p + 1):
        mu_n = current_db + n * slope * delta_t
        z = (mu_n - gamma_ho_db) / sigma
        product *= float(_gauss.cdf(z))
    return product


def compute_s_ho(let: float, p_surv: float, tau_req_s: float) -> float:
    """Handover stability.

    S_HO,ij = (1 - exp(-LET / tau_req)) * P_surv
    """
    if let == float("inf"):
        exp_term = 0.0
    elif tau_req_s <= 0:
        exp_term = 0.0
    else:
        exp_term = math.exp(-let / tau_req_s)
    return (1.0 - exp_term) * p_surv


# ═══════════════════════════════════════════════════════════════════════════
# Path-level metrics
# ═══════════════════════════════════════════════════════════════════════════

def path_quality(q_hops: List[float]) -> float:
    """Q_pi = geometric mean of per-hop Q values."""
    if not q_hops:
        return 0.0
    product = 1.0
    for q in q_hops:
        product *= max(q, 1e-12)
    return product ** (1.0 / len(q_hops))


def path_stability(s_hops: List[float]) -> float:
    """S_pi = geometric mean of per-hop S_HO values."""
    if not s_hops:
        return 0.0
    product = 1.0
    for s in s_hops:
        product *= max(s, 1e-12)
    return product ** (1.0 / len(s_hops))


def link_advantage(q_pi: float, s_pi: float, w_q: float, w_s: float) -> float:
    """LA_pi = w_Q * Q_pi + w_S * S_pi"""
    return w_q * q_pi + w_s * s_pi


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: compute full hop metrics for a single link
# ═══════════════════════════════════════════════════════════════════════════

def hop_metrics(q_ij: float, sinr_seq: np.ndarray,
                dp: np.ndarray, dv: np.ndarray,
                r_comm: float, cfg: EnvConfig) -> Dict[str, float]:
    """Return Q, LET, P_surv, S_HO for one hop."""
    let = compute_let(dp, dv, r_comm)
    tau_req_s = cfg.tau_req * 1e-3
    p_surv = compute_p_surv(sinr_seq, cfg.gamma_ho_linear,
                            cfg.N_p, cfg.delta_t_sim * 1e-3)
    s_ho = compute_s_ho(let, p_surv, tau_req_s)
    return {"Q": q_ij, "LET": let, "P_surv": p_surv, "S_HO": s_ho}


def evaluate_path(hop_q: List[float], hop_s_ho: List[float],
                  cfg: EnvConfig) -> Dict[str, float]:
    """Evaluate full path-level metrics."""
    q_pi = path_quality(hop_q)
    s_pi = path_stability(hop_s_ho)
    la = link_advantage(q_pi, s_pi, cfg.w_Q, cfg.w_S)
    return {"Q_pi": q_pi, "S_pi": s_pi, "LA_pi": la}


def compute_lqi(sinr_linear: float, xi_low: float = 0.0,
                xi_high: float = 30.0) -> int:
    """LQI_ij = clip(round(255 * (xi - xi_low)/(xi_high - xi_low)), 0, 255)

    xi is SINR in dB used as the quality statistic.
    """
    xi = 10.0 * math.log10(max(sinr_linear, 1e-30))
    if xi_high <= xi_low:
        return 0
    lqi = round(255.0 * (xi - xi_low) / (xi_high - xi_low))
    return max(0, min(255, lqi))
