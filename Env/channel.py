"""
Unified channel model for the integrated air-land-sea-space IoT simulation.

Path-loss families
  * 3GPP TR 38.811  — satellite-involving links
  * 3GPP TR 36.777 / 38.901  — UAV-to-ground, land/ship/buoy terrestrial
  * Two-ray sea-surface correction — short-range maritime RF

Small-scale fading
  * Rician  — LOS-dominant aerial / NTN links
  * Rayleigh / log-normal — NLOS or cluttered surface / land

Environmental noise
  * ITU-R P.372 external temperature + receiver noise factor

Includes both scalar (per-link) and vectorized (batch matrix) implementations.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import rice, rayleigh

from .config import EnvConfig

# Physical constants
C_LIGHT = 2.998e8        # m/s
K_BOLTZMANN = 1.380649e-23  # J/K
T0 = 290.0               # K reference temperature

# ═══════════════════════════════════════════════════════════════════════════
# Link-class determination
# ═══════════════════════════════════════════════════════════════════════════

_SAT_TYPES = {"satellite"}
_SURFACE_TYPES = {"ship", "buoy"}
_AERIAL_TYPES = {"uav"}

# Integer type encoding shared across scalar and vectorized code
TYPE_ID = {"satellite": 0, "uav": 1, "ship": 2, "buoy": 3, "land": 4}

def link_class(type_i: str, type_j: str) -> str:
    types = {type_i, type_j}
    if types & _SAT_TYPES:
        return "satellite"
    if types & _AERIAL_TYPES:
        return "uav_terrestrial"
    if types <= _SURFACE_TYPES:
        return "sea_surface"
    return "terrestrial"


# ═══════════════════════════════════════════════════════════════════════════
# Scalar path-loss models (unchanged — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def _fspl(d: float, f: float) -> float:
    """Free-space path loss (dB).  ITU-R P.525."""
    if d <= 0:
        return 0.0
    return 20.0 * math.log10(d) + 20.0 * math.log10(f) - 147.55


def path_loss_satellite(d_3d: float, f_c: float, elevation_deg: float,
                        cfg: EnvConfig, rng: np.random.Generator) -> float:
    """3GPP TR 38.811 large-scale model for satellite links."""
    pl_fs = _fspl(d_3d, f_c)
    cl = max(0.0, 20.0 - elevation_deg * 0.2)
    sf_std = max(2.0, 8.0 - elevation_deg * 0.06)
    sf = rng.normal(0, sf_std * cfg.eta_ch)
    atm = 0.2 * (90.0 - elevation_deg) / 90.0
    return pl_fs + cl + sf + atm


def path_loss_uav_terrestrial(d_3d: float, h_uav: float, f_c: float,
                              cfg: EnvConfig, rng: np.random.Generator) -> float:
    """3GPP TR 36.777 / 38.901 for UAV-to-ground / UAV-to-ship links."""
    d_2d = max(1.0, math.sqrt(max(0, d_3d ** 2 - h_uav ** 2)))
    p_los = min(1.0, 1.0 / (1.0 + 0.1 * math.exp(-0.01 * (h_uav - 20.0))))

    pl_los = _fspl(d_3d, f_c) + rng.normal(0, 4.0 * cfg.eta_ch)
    pl_nlos = _fspl(d_3d, f_c) + 20.0 + rng.normal(0, 8.0 * cfg.eta_ch)

    if rng.random() < p_los:
        return pl_los
    return pl_nlos


def path_loss_sea_surface(d_2d: float, h_tx: float, h_rx: float,
                          f_c: float, cfg: EnvConfig,
                          rng: np.random.Generator) -> float:
    """Two-ray sea-surface correction for short-range maritime RF."""
    if d_2d < 1.0:
        d_2d = 1.0
    wavelength = C_LIGHT / f_c
    d_break = 4.0 * max(h_tx, 0.5) * max(h_rx, 0.5) / wavelength

    if d_2d <= d_break:
        pl = _fspl(d_2d, f_c) + rng.normal(0, 3.0 * cfg.eta_ch)
    else:
        pl = (40.0 * math.log10(d_2d)
              - 20.0 * math.log10(max(h_tx, 0.5))
              - 20.0 * math.log10(max(h_rx, 0.5))
              + rng.normal(0, 6.0 * cfg.eta_ch))
    return pl


def path_loss_terrestrial(d_3d: float, f_c: float, cfg: EnvConfig,
                          rng: np.random.Generator) -> float:
    """3GPP TR 38.901 UMa for land-to-ship / land-to-buoy / land-to-land."""
    pl = _fspl(d_3d, f_c) + 10.0 + rng.normal(0, 6.0 * cfg.eta_ch)
    return pl


def compute_path_loss(type_i: str, type_j: str,
                      pos_i: np.ndarray, pos_j: np.ndarray,
                      cfg: EnvConfig, rng: np.random.Generator) -> float:
    """Dispatch to the appropriate path-loss model based on link class."""
    d_3d = float(np.linalg.norm(pos_i - pos_j))
    if d_3d < 1.0:
        d_3d = 1.0
    f_c = cfg.carrier_freq(type_i, type_j)
    lc = link_class(type_i, type_j)

    if lc == "satellite":
        h_ground = min(pos_i[2], pos_j[2])
        h_sat = max(pos_i[2], pos_j[2])
        d_horiz = float(np.linalg.norm(pos_i[:2] - pos_j[:2]))
        elevation = math.degrees(math.atan2(h_sat - h_ground, max(d_horiz, 1.0)))
        elevation = max(5.0, min(90.0, elevation))
        return path_loss_satellite(d_3d, f_c, elevation, cfg, rng)

    if lc == "uav_terrestrial":
        h_uav = max(pos_i[2], pos_j[2])
        return path_loss_uav_terrestrial(d_3d, h_uav, f_c, cfg, rng)

    if lc == "sea_surface":
        h_tx = max(pos_i[2], 0.5)
        h_rx = max(pos_j[2], 0.5)
        d_2d = float(np.linalg.norm(pos_i[:2] - pos_j[:2]))
        return path_loss_sea_surface(d_2d, h_tx, h_rx, f_c, cfg, rng)

    return path_loss_terrestrial(d_3d, f_c, cfg, rng)


# ═══════════════════════════════════════════════════════════════════════════
# Scalar small-scale fading (unchanged — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def fading_gain(type_i: str, type_j: str, cfg: EnvConfig,
                rng: np.random.Generator) -> float:
    """
    |g_ij(t)|^2  — instantaneous fading power gain.
    Rician for LOS-dominant (aerial / NTN), Rayleigh for NLOS / surface.
    """
    lc = link_class(type_i, type_j)
    if lc in ("satellite", "uav_terrestrial"):
        K_rice = 10.0 * cfg.eta_ch
        nu = math.sqrt(K_rice / (1.0 + K_rice))
        sigma = math.sqrt(0.5 / (1.0 + K_rice))
        x = rng.normal(nu, sigma)
        y = rng.normal(0, sigma)
        return x * x + y * y
    else:
        return rng.exponential(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Scalar environmental noise (unchanged — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def environmental_noise(cfg: EnvConfig, node_type: str,
                        rng: np.random.Generator) -> float:
    """
    N_env,i(t) = k_B * B_meas * (T0 * F_rx + T_ext)  * eta_N
    T_ext loosely follows ITU-R P.372 ambient categories.
    """
    t_ext_map = {
        "satellite": 50.0,
        "uav": 200.0,
        "ship": 300.0,
        "buoy": 350.0,
        "land": 250.0,
    }
    t_ext = t_ext_map.get(node_type, 290.0) + rng.normal(0, 20.0)
    t_ext = max(10.0, t_ext)
    n_env = K_BOLTZMANN * cfg.B_meas * (T0 * cfg.F_rx + t_ext) * cfg.eta_N
    return n_env


# ═══════════════════════════════════════════════════════════════════════════
# Scalar Doppler shift (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def doppler_shift(pos_i: np.ndarray, pos_j: np.ndarray,
                  vel_i: np.ndarray, vel_j: np.ndarray,
                  f_c: float) -> float:
    """f_D,ij = (f_c / c) * (v_i - v_j) · (p_j - p_i) / ||p_j - p_i||"""
    dp = pos_j - pos_i
    dist = np.linalg.norm(dp)
    if dist < 1e-3:
        return 0.0
    dv = vel_i - vel_j
    return float(f_c / C_LIGHT * np.dot(dv, dp) / dist)


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized FSPL (batch)
# ═══════════════════════════════════════════════════════════════════════════

def _fspl_vec(d: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Free-space path loss (dB) for arrays of distances and frequencies."""
    d_safe = np.maximum(d, 1.0)
    return 20.0 * np.log10(d_safe) + 20.0 * np.log10(f) - 147.55


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized link-class masks
# ═══════════════════════════════════════════════════════════════════════════

_TYPE_SAT = TYPE_ID["satellite"]   # 0
_TYPE_UAV = TYPE_ID["uav"]         # 1
_TYPE_SHIP = TYPE_ID["ship"]       # 2
_TYPE_BUOY = TYPE_ID["buoy"]       # 3
_TYPE_LAND = TYPE_ID["land"]       # 4

def build_link_class_masks(type_ids: np.ndarray):
    """
    Build boolean (N, N) masks for each link class from integer type IDs.

    Returns (mask_sat, mask_uav, mask_sea, mask_terr) — four (N, N) bool arrays.
    Exactly one is True per (i, j) pair.
    """
    N = len(type_ids)
    ti = type_ids[:, None]  # (N, 1)
    tj = type_ids[None, :]  # (1, N)

    is_sat_i = (ti == _TYPE_SAT)
    is_sat_j = (tj == _TYPE_SAT)
    mask_sat = is_sat_i | is_sat_j

    is_uav_i = (ti == _TYPE_UAV)
    is_uav_j = (tj == _TYPE_UAV)
    mask_uav = (is_uav_i | is_uav_j) & ~mask_sat

    is_surface_i = (ti == _TYPE_SHIP) | (ti == _TYPE_BUOY)
    is_surface_j = (tj == _TYPE_SHIP) | (tj == _TYPE_BUOY)
    mask_sea = is_surface_i & is_surface_j & ~mask_sat & ~mask_uav

    mask_terr = ~mask_sat & ~mask_uav & ~mask_sea

    return mask_sat, mask_uav, mask_sea, mask_terr


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized path-loss computation
# ═══════════════════════════════════════════════════════════════════════════

def vectorized_path_loss(
    dist_3d: np.ndarray,
    dist_2d: np.ndarray,
    positions: np.ndarray,
    type_ids: np.ndarray,
    mask_sat: np.ndarray,
    mask_uav: np.ndarray,
    mask_sea: np.ndarray,
    mask_terr: np.ndarray,
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Batch path-loss computation for all (N, N) pairs.

    Same formulas as the four scalar path_loss_* functions, applied per mask.
    Returns PL_matrix in dB, shape (N, N).
    """
    N = len(type_ids)
    PL = np.zeros((N, N), dtype=np.float64)
    d_safe = np.maximum(dist_3d, 1.0)

    # ── Satellite links ───────────────────────────────────────────────
    n_sat = int(mask_sat.sum())
    if n_sat > 0:
        f_c_sat = cfg.f_c_sat
        idx = np.where(mask_sat)
        d_pairs = d_safe[idx]
        pos_i = positions[idx[0]]
        pos_j = positions[idx[1]]

        h_ground = np.minimum(pos_i[:, 2], pos_j[:, 2])
        h_sat = np.maximum(pos_i[:, 2], pos_j[:, 2])
        d_horiz = np.sqrt((pos_i[:, 0] - pos_j[:, 0])**2 +
                          (pos_i[:, 1] - pos_j[:, 1])**2)
        elevation = np.degrees(np.arctan2(h_sat - h_ground,
                                          np.maximum(d_horiz, 1.0)))
        elevation = np.clip(elevation, 5.0, 90.0)

        fspl = 20.0 * np.log10(d_pairs) + 20.0 * np.log10(f_c_sat) - 147.55
        cl = np.maximum(0.0, 20.0 - elevation * 0.2)
        sf_std = np.maximum(2.0, 8.0 - elevation * 0.06) * cfg.eta_ch
        sf = rng.normal(0.0, 1.0, size=n_sat) * sf_std
        atm = 0.2 * (90.0 - elevation) / 90.0
        PL[idx] = fspl + cl + sf + atm

    # ── UAV-terrestrial links ─────────────────────────────────────────
    n_uav = int(mask_uav.sum())
    if n_uav > 0:
        f_c_local = cfg.f_c_local
        idx = np.where(mask_uav)
        d_pairs = d_safe[idx]
        pos_i = positions[idx[0]]
        pos_j = positions[idx[1]]
        h_uav = np.maximum(pos_i[:, 2], pos_j[:, 2])

        fspl = 20.0 * np.log10(d_pairs) + 20.0 * np.log10(f_c_local) - 147.55
        p_los = np.minimum(1.0, 1.0 / (1.0 + 0.1 * np.exp(-0.01 * (h_uav - 20.0))))

        sf_los = rng.normal(0.0, 4.0 * cfg.eta_ch, size=n_uav)
        sf_nlos = rng.normal(0.0, 8.0 * cfg.eta_ch, size=n_uav)
        is_los = rng.random(size=n_uav) < p_los

        pl_los = fspl + sf_los
        pl_nlos = fspl + 20.0 + sf_nlos
        PL[idx] = np.where(is_los, pl_los, pl_nlos)

    # ── Sea-surface links ─────────────────────────────────────────────
    n_sea = int(mask_sea.sum())
    if n_sea > 0:
        f_c_local = cfg.f_c_local
        wavelength = C_LIGHT / f_c_local
        idx = np.where(mask_sea)
        d2d_pairs = np.maximum(dist_2d[idx], 1.0)
        pos_i = positions[idx[0]]
        pos_j = positions[idx[1]]
        h_tx = np.maximum(pos_i[:, 2], 0.5)
        h_rx = np.maximum(pos_j[:, 2], 0.5)
        d_break = 4.0 * h_tx * h_rx / wavelength

        near = d2d_pairs <= d_break
        fspl_near = 20.0 * np.log10(d2d_pairs) + 20.0 * np.log10(f_c_local) - 147.55
        sf_near = rng.normal(0.0, 3.0 * cfg.eta_ch, size=n_sea)

        pl_far = (40.0 * np.log10(d2d_pairs)
                  - 20.0 * np.log10(h_tx)
                  - 20.0 * np.log10(h_rx))
        sf_far = rng.normal(0.0, 6.0 * cfg.eta_ch, size=n_sea)

        PL[idx] = np.where(near, fspl_near + sf_near, pl_far + sf_far)

    # ── Terrestrial links ─────────────────────────────────────────────
    n_terr = int(mask_terr.sum())
    if n_terr > 0:
        f_c_local = cfg.f_c_local
        idx = np.where(mask_terr)
        d_pairs = d_safe[idx]
        fspl = 20.0 * np.log10(d_pairs) + 20.0 * np.log10(f_c_local) - 147.55
        sf = rng.normal(0.0, 6.0 * cfg.eta_ch, size=n_terr)
        PL[idx] = fspl + 10.0 + sf

    return PL


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized fading
# ═══════════════════════════════════════════════════════════════════════════

def vectorized_fading(
    N: int,
    mask_rician: np.ndarray,
    mask_rayleigh: np.ndarray,
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Batch fading gain |g_ij|^2 for all (N, N) pairs.

    Rician for satellite + uav_terrestrial links, Rayleigh for the rest.
    """
    fading = np.ones((N, N), dtype=np.float64)

    n_ric = int(mask_rician.sum())
    if n_ric > 0:
        K_rice = 10.0 * cfg.eta_ch
        nu = math.sqrt(K_rice / (1.0 + K_rice))
        sigma = math.sqrt(0.5 / (1.0 + K_rice))
        x = rng.normal(nu, sigma, size=n_ric)
        y = rng.normal(0.0, sigma, size=n_ric)
        fading[mask_rician] = x * x + y * y

    n_ray = int(mask_rayleigh.sum())
    if n_ray > 0:
        fading[mask_rayleigh] = rng.exponential(1.0, size=n_ray)

    return fading


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized environmental noise
# ═══════════════════════════════════════════════════════════════════════════

_T_EXT_BY_TYPE = np.array([50.0, 200.0, 300.0, 350.0, 250.0])  # indexed by TYPE_ID

def vectorized_environmental_noise(
    cfg: EnvConfig,
    type_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch noise power for N receivers. Returns shape (N,)."""
    N = len(type_ids)
    t_ext_base = _T_EXT_BY_TYPE[type_ids]
    t_ext = t_ext_base + rng.normal(0.0, 20.0, size=N)
    t_ext = np.maximum(t_ext, 10.0)
    n_env = K_BOLTZMANN * cfg.B_meas * (T0 * cfg.F_rx + t_ext) * cfg.eta_N
    return n_env


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized Doppler shift
# ═══════════════════════════════════════════════════════════════════════════

def vectorized_doppler(
    positions: np.ndarray,
    velocities: np.ndarray,
    f_c_matrix: np.ndarray,
) -> np.ndarray:
    """
    Batch Doppler shift for all (N, N) pairs.

    f_D[i,j] = (f_c / c) * (v_i - v_j) . (p_j - p_i) / ||p_j - p_i||
    """
    dp = positions[None, :, :] - positions[:, None, :]   # (N, N, 3): p_j - p_i
    dv = velocities[:, None, :] - velocities[None, :, :] # (N, N, 3): v_i - v_j
    dist = np.linalg.norm(dp, axis=-1)                    # (N, N)
    dist_safe = np.maximum(dist, 1e-3)
    dot = np.sum(dv * dp, axis=-1)                        # (N, N)
    return f_c_matrix / C_LIGHT * dot / dist_safe
