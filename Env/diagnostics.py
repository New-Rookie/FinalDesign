"""
Configurable parameter-print function for the simulation environment.

When enabled, outputs all 86 parameters from EnvConfig in a formatted,
categorised table at environment startup.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from .config import EnvConfig, PARAM_META, PARAM_RANGES


def print_env_config(config: EnvConfig, enabled: bool = True) -> None:
    """Print every parameter grouped by category.

    Parameters
    ----------
    config : EnvConfig
        The active configuration instance.
    enabled : bool
        If False, the function is silent (no-op).
    """
    if not enabled:
        return

    _CATEGORY_ORDER = [
        "Population",
        "Time / mobility",
        "Discovery timing",
        "Carrier",
        "Link budget",
        "Tx power ceilings",
        "Noise setup",
        "Noise control",
        "Neighbor / topology thresholds",
        "Discovery energy",
        "Scalable counterfactual",
        "Link-selection (Ch 2)",
        "Resource management (Ch 3)",
        "Training / replay",
        "Simulation area",
        "Diagnostic flag",
    ]

    _FIELD_CATEGORIES = {
        "N_total": "Population",
        "N_MEC": "Population",
        "pi_sat": "Population",
        "pi_uav": "Population",
        "pi_ship": "Population",
        "pi_buoy": "Population",
        "pi_land": "Population",
        "N_src": "Population",
        "delta_t_sim": "Time / mobility",
        "T_w": "Discovery timing",
        "delta_t_slot": "Discovery timing",
        "N_slot": "Discovery timing",
        "f_c_local": "Carrier",
        "f_c_sat": "Carrier",
        "G_tx_sat": "Link budget",
        "G_rx_sat": "Link budget",
        "G_tx_uav": "Link budget",
        "G_rx_uav": "Link budget",
        "G_tx_ship": "Link budget",
        "G_rx_ship": "Link budget",
        "G_tx_buoy": "Link budget",
        "G_rx_buoy": "Link budget",
        "G_tx_land": "Link budget",
        "G_rx_land": "Link budget",
        "P_tx_buoy": "Tx power ceilings",
        "P_tx_ship_uav": "Tx power ceilings",
        "P_tx_sat": "Tx power ceilings",
        "B_meas": "Noise setup",
        "F_rx": "Noise setup",
        "eta_N": "Noise control",
        "gt_eta_N": "Noise control",
        "gamma_link": "Neighbor / topology thresholds",
        "P_fa": "Neighbor / topology thresholds",
        "beta_SIC": "Neighbor / topology thresholds",
        "K_max": "Neighbor / topology thresholds",
        "rho_min": "Neighbor / topology thresholds",
        "T_min": "Neighbor / topology thresholds",
        "P_listen": "Discovery energy",
        "P_dec": "Discovery energy",
        "P_sleep": "Discovery energy",
        "C_ops": "Discovery energy",
        "kappa": "Discovery energy",
        "f_cpu_ND": "Discovery energy",
        "E_ref": "Discovery energy",
        "B_cf": "Scalable counterfactual",
        "K_nbr": "Link-selection (Ch 2)",
        "K_sat": "Link-selection (Ch 2)",
        "N_probe": "Link-selection (Ch 2)",
        "L_pkt": "Link-selection (Ch 2)",
        "w_Q": "Link-selection (Ch 2)",
        "w_S": "Link-selection (Ch 2)",
        "eta_sw": "Link-selection (Ch 2)",
        "eta_ch": "Link-selection (Ch 2)",
        "tau_req": "Link-selection (Ch 2)",
        "gamma_ho": "Link-selection (Ch 2)",
        "N_p": "Link-selection (Ch 2)",
        "M_tot": "Resource management (Ch 3)",
        "c_v": "Resource management (Ch 3)",
        "gamma_cmp": "Resource management (Ch 3)",
        "T_max": "Resource management (Ch 3)",
        "E_max": "Resource management (Ch 3)",
        "epsilon_drop": "Resource management (Ch 3)",
        "B_0_buoy": "Resource management (Ch 3)",
        "S_0_buoy": "Resource management (Ch 3)",
        "B_0_ship": "Resource management (Ch 3)",
        "B_0_uav": "Resource management (Ch 3)",
        "F_0_ship": "Resource management (Ch 3)",
        "F_0_uav": "Resource management (Ch 3)",
        "S_0_ship": "Resource management (Ch 3)",
        "S_0_uav": "Resource management (Ch 3)",
        "B_0_sat": "Resource management (Ch 3)",
        "S_0_sat": "Resource management (Ch 3)",
        "S_0_bs": "Resource management (Ch 3)",
        "F_0_bs": "Resource management (Ch 3)",
        "eta_B": "Resource management (Ch 3)",
        "eta_F": "Resource management (Ch 3)",
        "eta_S": "Resource management (Ch 3)",
        "P_rx": "Resource management (Ch 3)",
        "P_mem": "Resource management (Ch 3)",
        "R_local_buoy": "Resource management (Ch 3)",
        "Gamma_max": "Resource management (Ch 3)",
        "K_hist": "Training / replay",
        "B_batch_replay": "Training / replay",
        "print_diagnostics": "Diagnostic flag",
        "area_width": "Simulation area",
        "area_height": "Simulation area",
        "sat_altitude": "Simulation area",
    }

    sep = "=" * 110
    thin = "-" * 110
    header = (f"{'#':>3}  {'Category':<30} {'Symbol':<22} {'Unit':<10} "
              f"{'Type':<8} {'Value':<18} {'Ch.'}")
    print()
    print(sep)
    print("  MARINE IoT ENVIRONMENT — PARAMETER TABLE  (87 parameters)")
    print(sep)
    print(header)
    print(thin)

    idx = 0
    flds = fields(config)
    printed_cats: set = set()

    for cat in _CATEGORY_ORDER:
        cat_fields = [f for f in flds if _FIELD_CATEGORIES.get(f.name) == cat]
        if not cat_fields:
            continue
        if cat in printed_cats:
            continue
        printed_cats.add(cat)
        for f in cat_fields:
            idx += 1
            name = f.name
            value = getattr(config, name)
            meta = PARAM_META.get(name, {})
            symbol = meta.get("symbol", name)
            unit = meta.get("unit", "")
            ptype = meta.get("ptype", "Config")
            chapters = meta.get("chapters", "")
            val_str = _format_value(value)
            print(f"{idx:>3}  {cat:<30} {symbol:<22} {unit:<10} "
                  f"{ptype:<8} {val_str:<18} {chapters}")
        print(thin)

    # any fields not in a category
    for f in flds:
        if f.name not in _FIELD_CATEGORIES:
            idx += 1
            value = getattr(config, f.name)
            meta = PARAM_META.get(f.name, {})
            symbol = meta.get("symbol", f.name)
            unit = meta.get("unit", "")
            ptype = meta.get("ptype", "Config")
            chapters = meta.get("chapters", "")
            val_str = _format_value(value)
            print(f"{idx:>3}  {'(other)':<30} {symbol:<22} {unit:<10} "
                  f"{ptype:<8} {val_str:<18} {chapters}")

    print(sep)
    print(f"  Total parameters listed: {idx}")
    print(sep)
    print()


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)
