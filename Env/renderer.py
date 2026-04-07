"""
Pygame-based game-like visualisation for the marine IoT simulation.

Layout:
  ┌───────────────────────────┬──────────┐
  │                           │  HUD     │
  │   Ocean / scene area      │  panel   │
  │   (960 × 800)             │  (320)   │
  │                           │          │
  └───────────────────────────┴──────────┘

Controls:
  Space  — pause / resume
  +/-    — speed up / slow down
  L      — toggle link display
  N      — toggle node labels
  Q/Esc  — quit
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import EnvConfig
    from .core_env import MarineIoTEnv

try:
    import pygame
    import pygame.freetype
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# Colour palette
COL_OCEAN       = (15, 25, 60)
COL_GRID        = (25, 40, 80)
COL_LAND        = (60, 50, 30)
COL_HUD_BG      = (20, 20, 30)
COL_HUD_TEXT    = (200, 210, 220)
COL_HUD_TITLE   = (100, 180, 255)
COL_SAT         = (255, 255, 255)
COL_UAV         = (80, 220, 120)
COL_SHIP        = (70, 140, 255)
COL_BUOY        = (255, 170, 50)
COL_LAND_NODE   = (160, 120, 80)
COL_LINK_GOOD   = (50, 200, 100, 100)
COL_LINK_MED    = (200, 200, 50, 80)
COL_LINK_BAD    = (200, 50, 50, 60)

SCENE_W, SCENE_H = 960, 800
HUD_W = 320
WIN_W = SCENE_W + HUD_W
WIN_H = SCENE_H


class GameRenderer:
    def __init__(self, cfg: "EnvConfig"):
        if not _HAS_PYGAME:
            raise ImportError("pygame is required for the game renderer")
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Marine IoT Simulation — Air-Land-Sea-Space MEC")
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        self.show_links = True
        self.show_labels = True
        self.paused = False
        self.speed_mult = 1

        # pre-render ocean grid
        self._grid_surf = pygame.Surface((SCENE_W, SCENE_H))
        self._grid_surf.fill(COL_OCEAN)
        for x in range(0, SCENE_W, 40):
            pygame.draw.line(self._grid_surf, COL_GRID, (x, 0), (x, SCENE_H))
        for y in range(0, SCENE_H, 40):
            pygame.draw.line(self._grid_surf, COL_GRID, (0, y), (SCENE_W, y))
        # land strip on right 15%
        land_rect = pygame.Rect(int(SCENE_W * 0.85), 0,
                                int(SCENE_W * 0.15), SCENE_H)
        pygame.draw.rect(self._grid_surf, COL_LAND, land_rect)

        # link surface (alpha)
        self._link_surf = pygame.Surface((SCENE_W, SCENE_H), pygame.SRCALPHA)

    # -----------------------------------------------------------------
    # Coordinate mapping
    # -----------------------------------------------------------------

    def _to_screen(self, pos: np.ndarray) -> tuple:
        x = pos[0] / self.cfg.area_width * SCENE_W
        y = (1.0 - pos[1] / self.cfg.area_height) * SCENE_H
        return int(np.clip(x, 0, SCENE_W - 1)), int(np.clip(y, 0, SCENE_H - 1))

    # -----------------------------------------------------------------
    # Draw nodes
    # -----------------------------------------------------------------

    def _draw_node(self, surface: pygame.Surface, node, env: "MarineIoTEnv"):
        sx, sy = self._to_screen(node.position)
        t = node.node_type

        if t == "satellite":
            # pulsing circle
            r = 6 + int(2 * math.sin(env._step_count * 0.15))
            pygame.draw.circle(surface, COL_SAT, (sx, sy), r)
            pygame.draw.circle(surface, (180, 180, 255), (sx, sy), r, 1)
        elif t == "uav":
            pts = [(sx, sy - 7), (sx - 6, sy + 5), (sx + 6, sy + 5)]
            pygame.draw.polygon(surface, COL_UAV, pts)
        elif t == "ship":
            pygame.draw.polygon(surface, COL_SHIP,
                                [(sx - 6, sy + 4), (sx + 6, sy + 4),
                                 (sx + 4, sy - 5), (sx - 4, sy - 5)])
        elif t == "buoy":
            wave_off = int(2 * math.sin(env._step_count * 0.2 + node.node_id))
            pygame.draw.circle(surface, COL_BUOY, (sx, sy + wave_off), 5)
            pygame.draw.circle(surface, (255, 200, 100), (sx, sy + wave_off), 5, 1)
        elif t == "land":
            pygame.draw.rect(surface, COL_LAND_NODE,
                             (sx - 5, sy - 5, 10, 10))

        if self.show_labels:
            label = self.font_sm.render(f"{node.node_id}", True, (180, 180, 180))
            surface.blit(label, (sx + 7, sy - 5))

    # -----------------------------------------------------------------
    # Draw links
    # -----------------------------------------------------------------

    def _draw_links(self, env: "MarineIoTEnv"):
        self._link_surf.fill((0, 0, 0, 0))
        if not self.show_links:
            return
        gt = env._gt_adj
        if gt is None:
            return
        n = len(env.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if not gt[i, j]:
                    continue
                p1 = self._to_screen(env.nodes[i].position)
                p2 = self._to_screen(env.nodes[j].position)
                lp = env.link_phy.get((j, i)) or env.link_phy.get((i, j))
                if lp and lp.sinr > 10:
                    col = COL_LINK_GOOD
                elif lp and lp.sinr > 2:
                    col = COL_LINK_MED
                else:
                    col = COL_LINK_BAD
                pygame.draw.line(self._link_surf, col, p1, p2, 1)

    # -----------------------------------------------------------------
    # HUD
    # -----------------------------------------------------------------

    def _draw_hud(self, env: "MarineIoTEnv"):
        hud_x = SCENE_W
        pygame.draw.rect(self.screen, COL_HUD_BG,
                         (hud_x, 0, HUD_W, WIN_H))
        y = 12
        self._hud_line("MARINE IoT SIMULATION", y, COL_HUD_TITLE, self.font_lg)
        y += 28
        self._hud_line("Air-Land-Sea-Space MEC", y, (120, 140, 160))
        y += 24
        pygame.draw.line(self.screen, (50, 50, 70),
                         (hud_x + 10, y), (hud_x + HUD_W - 10, y))
        y += 12

        info = env._build_info()
        nc = info["node_counts"]
        lines = [
            f"Step:  {info['step']}",
            f"Window slot: {info['window_slot']}/{env.cfg.N_slot}",
            f"",
            f"--- Node Counts ---",
            f"  Satellite : {nc.get('satellite', 0)}",
            f"  UAV       : {nc.get('uav', 0)}",
            f"  Ship      : {nc.get('ship', 0)}",
            f"  Buoy      : {nc.get('buoy', 0)}",
            f"  Land      : {nc.get('land', 0)}",
            f"  Total     : {info['n_nodes']}",
            f"",
            f"--- Topology ---",
            f"  F1_topo : {info['f1_topo']:.4f}",
            f"  TP={info['tp']}  FP={info['fp']}  FN={info['fn']}",
            f"",
            f"--- Config ---",
            f"  eta_N     : {env.cfg.eta_N}",
            f"  gamma_link: {env.cfg.gamma_link} dB",
            f"  P_fa      : {env.cfg.P_fa:.0e}",
            f"  beta_SIC  : {env.cfg.beta_SIC}",
            f"  Mode      : {env.mode}",
            f"",
            f"--- Controls ---",
            f"  Space : pause/resume",
            f"  +/-   : speed",
            f"  L     : toggle links",
            f"  N     : toggle labels",
            f"  Q/Esc : quit",
        ]
        if self.paused:
            lines.insert(0, "** PAUSED **")

        for line in lines:
            self._hud_line(line, y)
            y += 17

    def _hud_line(self, text: str, y: int,
                  colour=COL_HUD_TEXT, font=None):
        f = font or self.font_md
        surf = f.render(text, True, colour)
        self.screen.blit(surf, (SCENE_W + 14, y))

    # -----------------------------------------------------------------
    # Main render entry
    # -----------------------------------------------------------------

    def render(self, env: "MarineIoTEnv"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.close()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_l:
                    self.show_links = not self.show_links
                elif event.key == pygame.K_n:
                    self.show_labels = not self.show_labels
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_mult = min(16, self.speed_mult * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_mult = max(1, self.speed_mult // 2)

        # background
        self.screen.blit(self._grid_surf, (0, 0))

        # links
        self._draw_links(env)
        self.screen.blit(self._link_surf, (0, 0))

        # nodes
        for node in env.nodes:
            self._draw_node(self.screen, node, env)

        # HUD
        self._draw_hud(env)

        pygame.display.flip()
        self.clock.tick(30)

        if self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
        return None

    def close(self):
        if _HAS_PYGAME:
            pygame.quit()
