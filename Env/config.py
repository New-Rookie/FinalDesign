"""
Unified environment configuration for the integrated air-land-sea-space
MEC-enabled marine IoT simulation.  All 86 parameters from EnvConfig.xlsx
are represented here as a single dataclass with defaults, valid ranges,
metadata, and an optional xlsx loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dbm2w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def _db2lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


# ---------------------------------------------------------------------------
# Parameter metadata (symbol, unit, type, meaning, chapters)
# ---------------------------------------------------------------------------

PARAM_META: Dict[str, Dict] = {}

def _reg(name: str, symbol: str, unit: str, ptype: str,
         meaning: str, chapters: str):
    PARAM_META[name] = dict(symbol=symbol, unit=unit, ptype=ptype,
                            meaning=meaning, chapters=chapters)

# Registrations happen alongside the dataclass field definitions below.
# They are called once at module load time.

# ---------------------------------------------------------------------------
# Main configuration dataclass — 86 parameters
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    # ── Population ────────────────────────────────────────────────────────
    N_total: int = 50
    N_MEC: int = 10
    pi_sat: float = 0.05
    pi_uav: float = 0.15
    pi_ship: float = 0.25
    pi_buoy: float = 0.40
    pi_land: float = 0.15
    N_src: int = 10
    source_activation_ratio: float = 1.0  # fraction of source buoys active per window

    # ── Time / mobility ──────────────────────────────────────────────────
    delta_t_sim: float = 20.0        # ms

    # ── Discovery timing ─────────────────────────────────────────────────
    T_w: float = 100.0               # ms — discovery window duration
    delta_t_slot: float = 2.0        # ms — slot duration
    N_slot: int = 50                  # slots per window

    # ── Carrier ──────────────────────────────────────────────────────────
    f_c_local: float = 5.8e9         # Hz — local / UAV / maritime RF
    f_c_sat: float = 2.0e9           # Hz — satellite links

    # ── Link budget (antenna gains per class, dBi) ───────────────────────
    G_tx_sat: float = 30.0
    G_rx_sat: float = 30.0
    G_tx_uav: float = 5.0
    G_rx_uav: float = 5.0
    G_tx_ship: float = 8.0
    G_rx_ship: float = 8.0
    G_tx_buoy: float = 2.0
    G_rx_buoy: float = 2.0
    G_tx_land: float = 12.0
    G_rx_land: float = 12.0

    # ── Tx power ceilings (dBm) ─────────────────────────────────────────
    P_tx_buoy: float = 23.0
    P_tx_ship_uav: float = 27.0
    P_tx_sat: float = 38.0

    # ── Noise setup ──────────────────────────────────────────────────────
    B_meas: float = 10.0e6           # Hz — measurement bandwidth
    F_rx: float = 3.0                # noise factor (linear)
    eta_N: float = 1.0               # noise-intensity scaling
    gt_eta_N: float = 0.5            # fixed reference noise for GT topology

    # ── Neighbor / topology thresholds ───────────────────────────────────
    gamma_link: float = 5.0          # dB — topology-edge SNR threshold (raised for 500m area)
    P_fa: float = 0.01               # CFAR false-alarm target (comm-appropriate)
    beta_SIC: float = 0.05           # residual SIC fraction
    K_max: int = 2                   # max SIC loops
    rho_min: float = 0.6             # min decodable-slot fraction
    T_min: float = 30.0              # ms — min contact duration

    # ── Discovery energy hardware ────────────────────────────────────────
    P_listen: float = 0.1            # W
    P_dec: float = 0.15              # W
    P_sleep: float = 0.001           # W
    C_ops: float = 1000.0            # ops/iter
    kappa: float = 1.0e-30           # CMOS switched-capacitance coeff. (7nm MEC server)
    f_cpu_ND: float = 1.0e9          # Hz — SIC processor clock
    E_ref: float = 1.0               # J — discovery-energy normaliser

    # ── Scalable counterfactual ──────────────────────────────────────────
    B_cf: int = 24                   # counterfactual minibatch size

    # ── Link-selection (Ch 2) ────────────────────────────────────────────
    K_nbr: int = 8
    K_sat: int = 3
    N_probe: int = 20000
    L_pkt: int = 1024                # bit
    w_Q: float = 0.6
    w_S: float = 0.4
    eta_sw: float = 0.05
    eta_ch: float = 1.0
    tau_req: float = 200.0           # ms
    gamma_ho: float = 3.0            # dB
    N_p: int = 5                     # predictor horizon

    # ── Resource management (Ch 3) ───────────────────────────────────────
    M_tot: float = 60.0e6            # bit (60 Mbit)
    c_v: float = 1000.0              # cycles/bit
    gamma_cmp: float = 0.2           # compression ratio
    T_max: float = 1.5               # s — max tolerable delay (tightened so success/violation are distinguishable)
    E_max: float = 4.0               # J — max tolerable energy (tightened per V10 analysis)
    epsilon_drop: float = 0.0

    # Source ceilings
    B_0_buoy: float = 5.0e6          # Hz
    S_0_buoy: float = 1.0e9          # bit

    # MEC ceilings
    B_0_ship: float = 20.0e6         # Hz
    B_0_uav: float = 10.0e6          # Hz
    F_0_ship: float = 16.0e9         # Hz (GHz expressed in Hz)
    F_0_uav: float = 8.0e9           # Hz
    S_0_ship: float = 8.0e9          # bit
    S_0_uav: float = 4.0e9           # bit

    # Relay / sink ceilings
    B_0_sat: float = 50.0e6          # Hz
    S_0_sat: float = 4.0e9           # bit
    S_0_bs: float = 50.0e9           # bit
    F_0_bs: float = 32.0e9           # Hz

    # Heterogeneous scaling
    eta_B: float = 1.0
    eta_F: float = 1.0
    eta_S: float = 1.0

    # Rx / memory power
    P_rx: float = 0.1                # W
    P_mem: float = 0.01              # W

    # ── Derived / policy helpers ─────────────────────────────────────────
    R_local_buoy: float = 500.0       # m — buoy local-computing region radius (matches area diagonal)
    Gamma_max: float = 1.0e8         # bit/s — throughput normalisation constant

    # ── Training / replay ────────────────────────────────────────────────
    K_hist: int = 6
    B_batch_replay: int = 256

    # ── Diagnostic flag ──────────────────────────────────────────────────
    print_diagnostics: bool = True

    # ── Simulation area ──────────────────────────────────────────────────
    area_width: float = 500.0    # m — 0.5 km
    area_height: float = 500.0   # m — 0.5 km
    sat_altitude: float = 550_000.0  # m — LEO altitude

    # ==================================================================
    # Derived helpers
    # ==================================================================

    @property
    def node_counts(self) -> Dict[str, int]:
        fracs = {
            "satellite": self.pi_sat,
            "uav": self.pi_uav,
            "ship": self.pi_ship,
            "buoy": self.pi_buoy,
            "land": self.pi_land,
        }
        counts: Dict[str, int] = {}
        total_assigned = 0
        sorted_classes = sorted(fracs.items(), key=lambda x: x[1], reverse=True)
        for i, (cls, frac) in enumerate(sorted_classes):
            if i < len(sorted_classes) - 1:
                remaining_classes = len(sorted_classes) - 1 - i
                budget_left = self.N_total - total_assigned - remaining_classes
                c = max(1, min(round(frac * self.N_total), budget_left))
                counts[cls] = c
                total_assigned += c
            else:
                counts[cls] = max(1, self.N_total - total_assigned)
        return counts

    @property
    def active_source_count(self) -> int:
        return max(1, int(round(self.N_src * np.clip(self.source_activation_ratio, 0.1, 1.0))))

    @property
    def M_b(self) -> float:
        return self.M_tot / max(1, self.active_source_count)

    @property
    def gamma_link_linear(self) -> float:
        return _db2lin(self.gamma_link)

    @property
    def gamma_ho_linear(self) -> float:
        return _db2lin(self.gamma_ho)

    def tx_power_w(self, node_type: str) -> float:
        if node_type == "buoy":
            return _dbm2w(self.P_tx_buoy)
        elif node_type in ("ship", "uav"):
            return _dbm2w(self.P_tx_ship_uav)
        elif node_type == "satellite":
            return _dbm2w(self.P_tx_sat)
        elif node_type == "land":
            return _dbm2w(self.P_tx_ship_uav)
        return _dbm2w(self.P_tx_ship_uav)

    def antenna_gains(self, node_type: str) -> Tuple[float, float]:
        """Return (G_tx_linear, G_rx_linear) for a given node class."""
        mapping = {
            "satellite": (self.G_tx_sat, self.G_rx_sat),
            "uav": (self.G_tx_uav, self.G_rx_uav),
            "ship": (self.G_tx_ship, self.G_rx_ship),
            "buoy": (self.G_tx_buoy, self.G_rx_buoy),
            "land": (self.G_tx_land, self.G_rx_land),
        }
        g_tx_db, g_rx_db = mapping.get(node_type, (0.0, 0.0))
        return _db2lin(g_tx_db), _db2lin(g_rx_db)

    def carrier_freq(self, type_i: str, type_j: str) -> float:
        if type_i == "satellite" or type_j == "satellite":
            return self.f_c_sat
        return self.f_c_local

    def scaled_ceilings(self) -> Dict[str, Dict[str, float]]:
        return {
            "ship": {
                "B_max": self.eta_B * self.B_0_ship,
                "F_max": self.eta_F * self.F_0_ship,
                "S_max": self.eta_S * self.S_0_ship,
            },
            "uav": {
                "B_max": self.eta_B * self.B_0_uav,
                "F_max": self.eta_F * self.F_0_uav,
                "S_max": self.eta_S * self.S_0_uav,
            },
        }

    # ==================================================================
    # xlsx loader
    # ==================================================================

    @classmethod
    def from_xlsx(cls, path: str | Path) -> "EnvConfig":
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError("openpyxl is required to load xlsx config") from exc
        wb = openpyxl.load_workbook(str(path), read_only=True)
        ws = wb.active
        cfg = cls()
        field_names = {f.name for f in fields(cls)}
        _symbol_to_field = _build_symbol_map()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1] is None:
                continue
            symbol = str(row[1]).strip()
            fname = _symbol_to_field.get(symbol)
            if fname and fname in field_names:
                default_str = str(row[4]).strip() if row[4] else None
                if default_str:
                    val = _parse_default(default_str, type(getattr(cfg, fname)))
                    if val is not None:
                        setattr(cfg, fname, val)
        wb.close()
        return cfg


# ---------------------------------------------------------------------------
# Symbol-to-field mapping for xlsx parsing
# ---------------------------------------------------------------------------

def _build_symbol_map() -> Dict[str, str]:
    return {
        r"(N_{total})": "N_total",
        r"(N_{MEC})": "N_MEC",
        r"(N_{src})": "N_src",
        r"(\Delta t_{sim})": "delta_t_sim",
        r"(T_w)": "T_w",
        r"(\Delta t_{slot})": "delta_t_slot",
        r"(N_{slot})": "N_slot",
        r"(f_{c,local})": "f_c_local",
        r"(f_{c,sat})": "f_c_sat",
        r"(\eta_N)": "eta_N",
        r"(B_{meas})": "B_meas",
        r"(F_{rx,i})": "F_rx",
        r"(\gamma_{link})": "gamma_link",
        r"(P_{fa})": "P_fa",
        r"(\beta_{SIC})": "beta_SIC",
        r"(K_{max})": "K_max",
        r"(\rho_{min})": "rho_min",
        r"(T_{min})": "T_min",
        r"(\kappa_i)": "kappa",
        r"(B_{cf})": "B_cf",
        r"(K_{nbr})": "K_nbr",
        r"(K_{sat})": "K_sat",
        r"(N_{probe})": "N_probe",
        r"(L_{pkt})": "L_pkt",
        r"(\eta_{sw})": "eta_sw",
        r"(\eta_{ch})": "eta_ch",
        r"(\tau_{req})": "tau_req",
        r"(\gamma_{ho})": "gamma_ho",
        r"(M_{tot})": "M_tot",
        r"(c_v)": "c_v",
        r"(\gamma_{cmp})": "gamma_cmp",
        r"(T_{max})": "T_max",
        r"(E_{max})": "E_max",
        r"(K_{hist})": "K_hist",
        r"(B_{batch,replay})": "B_batch_replay",
    }


def _parse_default(s: str, target_type: type):
    """Best-effort extraction of the numeric default from xlsx."""
    s = s.split("/")[0].strip()
    s = s.replace("\u2013", "-").replace("–", "-")
    try:
        if target_type == int:
            return int(float(s))
        elif target_type == float:
            return float(s)
    except (ValueError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Parameter range definitions (for diagnostics / validation)
# ---------------------------------------------------------------------------

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "N_total":       (10, 80),
    "N_MEC":         (4, 20),
    "eta_N":         (0.5, 2.0),
    "T_w":           (50, 200),
    "delta_t_slot":  (0.5, 2.0),
    "N_slot":        (25, 200),
    "P_fa":          (1e-5, 0.1),
    "beta_SIC":      (0.02, 0.10),
    "K_max":         (1, 4),
    "gamma_link":    (3, 15),
    "rho_min":       (0.5, 0.8),
    "T_min":         (10, 100),
    "B_cf":          (16, 32),
    "K_nbr":         (4, 10),
    "N_probe":       (10000, 50000),
    "L_pkt":         (256, 2048),
    "w_Q":           (0.4, 0.8),
    "w_S":           (0.2, 0.6),
    "eta_sw":        (0.01, 0.20),
    "eta_ch":        (0.75, 1.50),
    "tau_req":       (100, 500),
    "gamma_ho":      (0, 8),
    "M_tot":         (20e6, 100e6),
    "c_v":           (500, 3000),
    "gamma_cmp":     (0.05, 0.50),
    "T_max":         (1.0, 10.0),
    "E_max":         (1.0, 50.0),
    "eta_B":         (0.5, 1.5),
    "eta_F":         (0.5, 1.5),
    "eta_S":         (0.5, 1.5),
    "K_hist":        (4, 10),
    "B_batch_replay": (128, 512),
}


# Register metadata for every parameter (86 total)
_REG = [
    ("N_total", r"N_{total}", "1", "Config", "Total node count", "1,2"),
    ("N_MEC", r"N_{MEC}", "1", "Config", "Total MEC-capable nodes", "3"),
    ("pi_sat", r"\pi_{sat}", "1", "Config", "Satellite class fraction", "1,2,3"),
    ("pi_uav", r"\pi_{uav}", "1", "Config", "UAV class fraction", "1,2,3"),
    ("pi_ship", r"\pi_{ship}", "1", "Config", "Ship class fraction", "1,2,3"),
    ("pi_buoy", r"\pi_{buoy}", "1", "Config", "Buoy class fraction", "1,2,3"),
    ("pi_land", r"\pi_{land}", "1", "Config", "Land-station class fraction", "1,2,3"),
    ("N_src", r"N_{src}", "1", "Config", "Number of active source buoys", "3"),
    ("delta_t_sim", r"\Delta t_{sim}", "ms", "Config", "Mobility & channel update interval", "1,2,3"),
    ("T_w", r"T_w", "ms", "Config", "Discovery-window duration", "1"),
    ("delta_t_slot", r"\Delta t_{slot}", "ms", "Config", "Discovery slot duration", "1"),
    ("N_slot", r"N_{slot}", "1", "Config", "Slots per discovery window", "1"),
    ("f_c_local", r"f_{c,local}", "GHz", "Config", "Carrier for local/UAV/maritime RF links", "1,2,3"),
    ("f_c_sat", r"f_{c,sat}", "GHz", "Config", "Carrier for satellite-involving links", "1,2,3"),
    ("G_tx_sat", r"G_{tx,sat}", "dBi", "Config", "Satellite Tx antenna gain", "1,2,3"),
    ("G_rx_sat", r"G_{rx,sat}", "dBi", "Config", "Satellite Rx antenna gain", "1,2,3"),
    ("G_tx_uav", r"G_{tx,uav}", "dBi", "Config", "UAV Tx antenna gain", "1,2,3"),
    ("G_rx_uav", r"G_{rx,uav}", "dBi", "Config", "UAV Rx antenna gain", "1,2,3"),
    ("G_tx_ship", r"G_{tx,ship}", "dBi", "Config", "Ship Tx antenna gain", "1,2,3"),
    ("G_rx_ship", r"G_{rx,ship}", "dBi", "Config", "Ship Rx antenna gain", "1,2,3"),
    ("G_tx_buoy", r"G_{tx,buoy}", "dBi", "Config", "Buoy Tx antenna gain", "1,2,3"),
    ("G_rx_buoy", r"G_{rx,buoy}", "dBi", "Config", "Buoy Rx antenna gain", "1,2,3"),
    ("G_tx_land", r"G_{tx,land}", "dBi", "Config", "Land-station Tx antenna gain", "1,2,3"),
    ("G_rx_land", r"G_{rx,land}", "dBi", "Config", "Land-station Rx antenna gain", "1,2,3"),
    ("P_tx_buoy", r"P_{tx,buoy}", "dBm", "Config", "Buoy transmit-power ceiling", "1,2,3"),
    ("P_tx_ship_uav", r"P_{tx,ship/UAV}", "dBm", "Config", "Ship/UAV transmit-power ceiling", "1,2,3"),
    ("P_tx_sat", r"P_{tx,sat}", "dBm", "Config", "Satellite EIRP proxy", "1,2,3"),
    ("B_meas", r"B_{meas}", "Hz", "Config", "Measurement bandwidth in RSSI / noise", "1,2,3"),
    ("F_rx", r"F_{rx,i}", "1", "Config", "Receiver noise factor", "1,2,3"),
    ("eta_N", r"\eta_N", "1", "Config", "Noise-intensity scaling factor", "1,3"),
    ("gt_eta_N", r"\eta_N^{GT}", "1", "Config", "Fixed reference noise for GT topology", "1"),
    ("gamma_link", r"\gamma_{link}", "dB", "Config", "Topology-edge SNR threshold", "1,2,3"),
    ("P_fa", r"P_{fa}", "1", "Config", "CFAR false-alarm target", "1"),
    ("beta_SIC", r"\beta_{SIC}", "1", "Config", "Residual SIC fraction", "1"),
    ("K_max", r"K_{max}", "1", "Config", "Maximum SIC loops", "1"),
    ("rho_min", r"\rho_{min}", "1", "Config", "Min decodable-slot fraction in window", "1"),
    ("T_min", r"T_{min}", "ms", "Config", "Minimum contact duration for topology edge", "1"),
    ("P_listen", r"P_{listen,i}", "W", "Config", "Listening power", "1"),
    ("P_dec", r"P_{dec,i}", "W", "Config", "Decoding power", "1"),
    ("P_sleep", r"P_{sleep,i}", "W", "Config", "Sleep power", "1"),
    ("C_ops", r"C_{ops,i}", "ops/iter", "Config", "SIC operations per iteration", "1"),
    ("kappa", r"\kappa_i", "coeff", "Config", "CMOS switched-capacitance coefficient", "1,3"),
    ("f_cpu_ND", r"f^{ND}_{cpu,i}", "Hz", "Config", "SIC processor clock", "1"),
    ("E_ref", r"E_{ref}", "J", "Config", "Discovery-energy normalisation constant", "1"),
    ("B_cf", r"B_{cf}", "1", "Config", "Counterfactual minibatch size", "1"),
    ("K_nbr", r"K_{nbr}", "1", "Config", "Max candidate neighbors after prefilter", "2"),
    ("K_sat", r"K_{sat}", "1", "Config", "Max satellite relay combos per step", "2"),
    ("N_probe", r"N_{probe}", "packets", "Config", "Probe samples per channel regime", "2"),
    ("L_pkt", r"L_{pkt}", "bit", "Config", "Probe packet length", "2"),
    ("w_Q", r"w_Q", "1", "Config", "Path-quality weight", "2"),
    ("w_S", r"w_S", "1", "Config", "Path-stability weight", "2"),
    ("eta_sw", r"\eta_{sw}", "1", "Config", "GMAPPO switching penalty", "2"),
    ("eta_ch", r"\eta_{ch}", "1", "Config", "Channel-condition scale", "2"),
    ("tau_req", r"\tau_{req}", "ms", "Config", "Required sustaining duration", "2"),
    ("gamma_ho", r"\gamma_{ho}", "dB", "Config", "Sustaining SINR threshold", "2"),
    ("N_p", r"N_p", "steps", "Config", "Horizon for discrete survival approx.", "2"),
    ("M_tot", r"M_{tot}", "bit", "Config", "Total source data volume per observation", "3"),
    ("c_v", r"c_v", "cycles/bit", "Config", "CPU cycles per input bit", "3"),
    ("gamma_cmp", r"\gamma_{cmp}", "1", "Config", "Processed-result compression ratio", "3"),
    ("T_max", r"T_{max}", "s", "Config", "Maximum tolerable end-to-end delay", "3"),
    ("E_max", r"E_{max}", "J", "Config", "Maximum tolerable end-to-end energy", "3"),
    ("epsilon_drop", r"\varepsilon_{drop}", "1", "Config", "Strict success criterion threshold", "3"),
    ("B_0_buoy", r"B_{0,buoy}", "Hz", "Config", "Source-buoy bandwidth ceiling", "3"),
    ("S_0_buoy", r"S_{0,buoy}", "bit", "Config", "Source-buoy storage ceiling", "3"),
    ("B_0_ship", r"B_{0,ship}", "Hz", "Config", "Ship baseline bandwidth ceiling", "3"),
    ("B_0_uav", r"B_{0,UAV}", "Hz", "Config", "UAV baseline bandwidth ceiling", "3"),
    ("F_0_ship", r"F_{0,ship}", "Hz", "Config", "Ship baseline CPU ceiling", "3"),
    ("F_0_uav", r"F_{0,UAV}", "Hz", "Config", "UAV baseline CPU ceiling", "3"),
    ("S_0_ship", r"S_{0,ship}", "bit", "Config", "Ship baseline storage ceiling", "3"),
    ("S_0_uav", r"S_{0,UAV}", "bit", "Config", "UAV baseline storage ceiling", "3"),
    ("B_0_sat", r"B_{0,sat}", "Hz", "Config", "Satellite relay bandwidth ceiling", "3"),
    ("S_0_sat", r"S_{0,sat}", "bit", "Config", "Satellite relay storage ceiling", "3"),
    ("S_0_bs", r"S_{0,bs}", "bit", "Config", "Base-station storage ceiling", "3"),
    ("F_0_bs", r"F_{0,bs}", "Hz", "Config", "Base-station compute ceiling", "3"),
    ("eta_B", r"\eta_B", "1", "Config", "Bandwidth scaling factor", "3"),
    ("eta_F", r"\eta_F", "1", "Config", "Compute scaling factor", "3"),
    ("eta_S", r"\eta_S", "1", "Config", "Storage scaling factor", "3"),
    ("P_rx", r"P_{rx,i}", "W", "Config", "Receive-circuit power", "3"),
    ("P_mem", r"P_{mem,i}", "W", "Config", "RAM queue-holding power", "3"),
    ("K_hist", r"K_{hist}", "steps", "Config", "Temporal window for improved MATD3", "3"),
    ("B_batch_replay", r"B_{batch,replay}", "trans.", "Config", "Replay mini-batch size", "3"),
    ("R_local_buoy", r"R_{local}^b", "m", "Derived/Config", "Buoy local-computing region radius", "2,3"),
    ("Gamma_max", r"\Gamma_{max}", "bit/s", "Derived/Config", "Throughput normalisation constant", "3"),
    ("print_diagnostics", "print_diag", "bool", "Config", "Enable parameter printout at startup", "1,2,3"),
    ("area_width", "area_w", "m", "Config", "Simulation area width", "1,2,3"),
    ("area_height", "area_h", "m", "Config", "Simulation area height", "1,2,3"),
    ("sat_altitude", "h_{orbit}", "m", "Config", "LEO satellite altitude", "1,2,3"),
]

for _entry in _REG:
    _reg(*_entry)
