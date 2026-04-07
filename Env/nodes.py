"""
Dynamic node models for the integrated air-land-sea-space IoT simulation.

Each class implements the mobility model specified in the manuscripts:
  - Satellite:    circular-orbit surrogate (optional TLE/SGP4)
  - UAV:          bounded 3-D Gauss–Markov
  - Ship:         2-D coordinated-turn
  - Buoy:         sea-surface drift + wave heave (RF surface node)
  - LandStation:  quasi-static anchor
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Base
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BaseNode:
    node_id: int
    node_type: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy_residual: float = 100.0        # J – initial battery
    is_active: bool = True

    # per-slot Tx power (W) set by the protocol / RL agent each step
    tx_power: float = 0.0

    def update(self, dt: float, rng: np.random.Generator) -> None:
        raise NotImplementedError

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


# ═══════════════════════════════════════════════════════════════════════════
# Satellite — circular-orbit surrogate
# ═══════════════════════════════════════════════════════════════════════════

EARTH_RADIUS = 6_371_000.0  # m
MU_EARTH = 3.986004418e14   # m^3 s^-2

@dataclass
class SatelliteNode(BaseNode):
    altitude: float = 550_000.0            # m  (LEO)
    inclination: float = 0.0               # rad
    raan: float = 0.0                      # rad – right ascension of ascending node
    initial_anomaly: float = 0.0           # rad
    _elapsed: float = 0.0

    def __post_init__(self):
        self.node_type = "satellite"
        r = EARTH_RADIUS + self.altitude
        self._orbital_period = 2.0 * math.pi * math.sqrt(r ** 3 / MU_EARTH)
        self._omega = 2.0 * math.pi / self._orbital_period  # rad/s
        self._r = r
        self._update_position()

    def _update_position(self):
        theta = self.initial_anomaly + self._omega * self._elapsed
        x = self._r * math.cos(theta) * math.cos(self.raan) - \
            self._r * math.sin(theta) * math.cos(self.inclination) * math.sin(self.raan)
        y = self._r * math.cos(theta) * math.sin(self.raan) + \
            self._r * math.sin(theta) * math.cos(self.inclination) * math.cos(self.raan)
        z_orbit = self._r * math.sin(theta) * math.sin(self.inclination)

        # project to local ENU: x_enu, y_enu on the ground-plane, z = altitude
        self.position = np.array([x % 100_000, y % 100_000, self.altitude], dtype=np.float64)

        v_tangential = self._omega * self._r
        vx = -v_tangential * math.sin(theta)
        vy = v_tangential * math.cos(theta)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)

    def update(self, dt: float, rng: np.random.Generator) -> None:
        self._elapsed += dt
        self._update_position()


# ═══════════════════════════════════════════════════════════════════════════
# UAV — bounded 3-D Gauss-Markov
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UAVNode(BaseNode):
    h_min: float = 50.0         # m
    h_max: float = 300.0        # m
    speed_mean: float = 15.0    # m/s
    alpha_gm: float = 0.8       # memory factor
    sigma_v: float = 3.0        # m/s noise std
    max_turn_rate: float = 0.3  # rad/s
    _heading: float = 0.0

    def __post_init__(self):
        self.node_type = "uav"

    def update(self, dt: float, rng: np.random.Generator) -> None:
        noise = rng.normal(0, self.sigma_v, size=3)
        mean_v = np.array([
            self.speed_mean * math.cos(self._heading),
            self.speed_mean * math.sin(self._heading),
            0.0,
        ])
        self.velocity = self.alpha_gm * self.velocity + \
                        (1 - self.alpha_gm) * mean_v + \
                        math.sqrt(1 - self.alpha_gm ** 2) * noise

        self._heading += np.clip(rng.normal(0, 0.1), -self.max_turn_rate * dt,
                                 self.max_turn_rate * dt)
        self.position = self.position + self.velocity * dt
        self.position[2] = np.clip(self.position[2], self.h_min, self.h_max)
        self.position[0] = np.clip(self.position[0], 0, 100_000)
        self.position[1] = np.clip(self.position[1], 0, 100_000)


# ═══════════════════════════════════════════════════════════════════════════
# Ship — 2-D coordinated-turn
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShipNode(BaseNode):
    heading: float = 0.0           # rad
    speed_nominal: float = 8.0     # m/s (~15 kn)
    turn_rate: float = 0.0         # rad/s
    max_turn_rate: float = 0.05    # rad/s
    max_speed: float = 15.0        # m/s
    sigma_turn: float = 0.01
    sigma_speed: float = 0.5

    def __post_init__(self):
        self.node_type = "ship"
        self.position[2] = 10.0  # antenna deck height ~10 m

    def update(self, dt: float, rng: np.random.Generator) -> None:
        self.turn_rate += rng.normal(0, self.sigma_turn)
        self.turn_rate = np.clip(self.turn_rate, -self.max_turn_rate, self.max_turn_rate)
        self.heading += self.turn_rate * dt

        spd = np.linalg.norm(self.velocity[:2])
        spd += rng.normal(0, self.sigma_speed) * dt
        spd = np.clip(spd, 1.0, self.max_speed)

        self.velocity[0] = spd * math.cos(self.heading)
        self.velocity[1] = spd * math.sin(self.heading)
        self.velocity[2] = 0.0

        self.position += self.velocity * dt
        self.position[2] = 10.0  # maintain deck height
        self.position[0] = np.clip(self.position[0], 0, 100_000)
        self.position[1] = np.clip(self.position[1], 0, 100_000)


# ═══════════════════════════════════════════════════════════════════════════
# Buoy — sea-surface drift + wave heave
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BuoyNode(BaseNode):
    drift_speed: float = 0.3         # m/s
    wave_amplitude: float = 1.5      # m
    wave_period: float = 6.0         # s
    sigma_drift: float = 0.05        # m/s
    mast_height: float = 4.0         # m — antenna mast above waterline
    _phase: float = 0.0
    _elapsed: float = 0.0

    def __post_init__(self):
        self.node_type = "buoy"

    def update(self, dt: float, rng: np.random.Generator) -> None:
        self._elapsed += dt
        drift_dir = math.atan2(self.velocity[1], self.velocity[0]) if np.linalg.norm(self.velocity[:2]) > 0.01 else rng.uniform(0, 2 * math.pi)
        drift_dir += rng.normal(0, 0.05)
        spd = self.drift_speed + rng.normal(0, self.sigma_drift)
        spd = max(0.0, spd)

        self.velocity[0] = spd * math.cos(drift_dir)
        self.velocity[1] = spd * math.sin(drift_dir)
        self.velocity[2] = self.wave_amplitude * (2 * math.pi / self.wave_period) * \
                           math.cos(2 * math.pi * self._elapsed / self.wave_period + self._phase)

        self.position[:2] += self.velocity[:2] * dt
        # mast_height above waterline + wave heave
        self.position[2] = self.mast_height + self.wave_amplitude * math.sin(
            2 * math.pi * self._elapsed / self.wave_period + self._phase)
        self.position[0] = np.clip(self.position[0], 0, 100_000)
        self.position[1] = np.clip(self.position[1], 0, 100_000)


# ═══════════════════════════════════════════════════════════════════════════
# Land station — quasi-static anchor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LandStationNode(BaseNode):
    def __post_init__(self):
        self.node_type = "land"
        self.velocity = np.zeros(3)

    def update(self, dt: float, rng: np.random.Generator) -> None:
        pass  # static


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLS = {
    "satellite": SatelliteNode,
    "uav": UAVNode,
    "ship": ShipNode,
    "buoy": BuoyNode,
    "land": LandStationNode,
}


def create_node(node_id: int, node_type: str, rng: np.random.Generator,
                area_w: float = 100_000.0, area_h: float = 100_000.0,
                sat_alt: float = 550_000.0) -> BaseNode:
    """Spawn a node with randomised initial state."""
    if node_type == "satellite":
        node = SatelliteNode(
            node_id=node_id,
            altitude=sat_alt,
            inclination=rng.uniform(0.0, math.pi / 3),
            raan=rng.uniform(0.0, 2 * math.pi),
            initial_anomaly=rng.uniform(0.0, 2 * math.pi),
        )
    elif node_type == "uav":
        node = UAVNode(node_id=node_id)
        node.position = np.array([
            rng.uniform(0, area_w),
            rng.uniform(0, area_h),
            rng.uniform(node.h_min, node.h_max),
        ])
        heading = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(5, node.speed_mean)
        node.velocity = np.array([spd * math.cos(heading),
                                  spd * math.sin(heading), 0.0])
        node._heading = heading
    elif node_type == "ship":
        node = ShipNode(node_id=node_id)
        node.position = np.array([
            rng.uniform(0, area_w),
            rng.uniform(0, area_h),
            10.0,  # deck height
        ])
        node.heading = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(3, node.speed_nominal)
        node.velocity = np.array([spd * math.cos(node.heading),
                                  spd * math.sin(node.heading), 0.0])
    elif node_type == "buoy":
        node = BuoyNode(node_id=node_id)
        node.position = np.array([
            rng.uniform(0, area_w),
            rng.uniform(0, area_h),
            node.mast_height,  # antenna mast above waterline
        ])
        drift_dir = rng.uniform(0, 2 * math.pi)
        node.velocity = np.array([node.drift_speed * math.cos(drift_dir),
                                  node.drift_speed * math.sin(drift_dir), 0.0])
        node._phase = rng.uniform(0, 2 * math.pi)
    elif node_type == "land":
        # place land stations along one edge (coast-line)
        node = LandStationNode(node_id=node_id)
        node.position = np.array([
            rng.uniform(0.85 * area_w, area_w),
            rng.uniform(0, area_h),
            rng.uniform(5, 30),  # small elevation
        ])
    else:
        raise ValueError(f"Unknown node type: {node_type}")
    return node
