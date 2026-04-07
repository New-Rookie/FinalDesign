"""
Metrics module for Research Content 3 resource management experiments.

Provides aggregate statistics computed from a batch of OffloadResult objects.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .task_offloader import OffloadResult


def aggregate_results(results: List[OffloadResult], Gamma_max: float = 1e8) -> Dict[str, float]:
    """Compute per-window aggregate metrics from offloading results.

    Tasks with a valid link path are used for T/E/Gamma means so that
    uncontrollable topology failures (no local candidate in range) do not
    swamp the metrics the agent *can* influence.  The success_rate and
    routable_ratio still reflect full-set statistics.
    """
    if not results:
        return {
            "mean_T_total": 1e6,
            "mean_E_total": 1e6,
            "mean_Gamma": 0.0,
            "success_rate": 0.0,
            "mean_alpha": 0.0,
            "throughput_normalised": 0.0,
            "routable_ratio": 0.0,
        }

    routable = [r for r in results if r.local_id >= 0]
    successes = np.array([r.success for r in results], dtype=np.float64)
    alphas = np.array([r.alpha_off for r in results], dtype=np.float64)

    if routable:
        T_vals = np.array([r.T_total for r in routable], dtype=np.float64)
        E_vals = np.array([r.E_total for r in routable], dtype=np.float64)
        G_vals = np.array([r.Gamma for r in routable], dtype=np.float64)
    else:
        T_vals = np.array([r.T_total for r in results], dtype=np.float64)
        E_vals = np.array([r.E_total for r in results], dtype=np.float64)
        G_vals = np.array([r.Gamma for r in results], dtype=np.float64)

    return {
        "mean_T_total": float(np.mean(T_vals)),
        "mean_E_total": float(np.mean(E_vals)),
        "mean_Gamma": float(np.mean(G_vals)),
        "success_rate": float(np.mean(successes)),
        "mean_alpha": float(np.mean(alphas)),
        "throughput_normalised": float(np.mean(G_vals) / max(Gamma_max, 1.0)),
        "routable_ratio": float(len(routable) / len(results)),
    }


def compute_reward(
    metrics: Dict[str, float],
    T_max: float,
    E_max: float,
    Gamma_max: float,
    lambda_G: float = 2.0,
    lambda_T: float = 0.5,
    lambda_E: float = 0.5,
    lambda_V: float = 0.5,
) -> float:
    """
    Scalar reward following Manuscript III with improved shaping:
      r = λ_Γ (Γ/Γ_max) − λ_T (T/T_max) − λ_E (E/E_max) − λ_V 1{violation}
          + bonus_success + bonus_progress + route_bonus

    Design choices for stable RL training:
      * Clamp at 5.0 preserves gradient for modest vs catastrophic failures.
      * routable_ratio bonus rewards policies that keep buoys assignable
        (even though topology is partially uncontrollable, the agent can
        learn to avoid wasting effort on unreachable buoys).
    """
    norm_G = metrics["throughput_normalised"]
    norm_T = min(metrics["mean_T_total"] / max(T_max, 1e-9), 5.0)
    norm_E = min(metrics["mean_E_total"] / max(E_max, 1e-9), 5.0)
    violation = 1.0 if (metrics["mean_T_total"] > T_max or
                        metrics["mean_E_total"] > E_max) else 0.0

    base = (lambda_G * norm_G
            - lambda_T * norm_T
            - lambda_E * norm_E
            - lambda_V * violation)

    success_bonus = 0.5 * metrics["success_rate"]

    progress_T = max(0.0, 1.0 - norm_T) * 0.3
    progress_E = max(0.0, 1.0 - norm_E) * 0.3

    route_bonus = 0.2 * metrics.get("routable_ratio", 1.0)

    return base + success_bonus + progress_T + progress_E + route_bonus
