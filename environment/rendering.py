"""
Pygame visualization for the EV Dispatch environment.
Shows city zones, EV position, SOC, chargers, ride requests, and live metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from environment.custom_env import EVDispatchEnv

# Zone screen centers (pixel coords)
ZONE_POS = {
    0: (180, 160),   # Residential
    1: (460, 160),   # Downtown
    2: (180, 380),   # Industrial
    3: (460, 380),   # Airport
}


class EVRenderer:
    def __init__(self, env: "EVDispatchEnv", width: int = 900, height: int = 560):
        import pygame

        self.env = env
        self.width = width
        self.height = height
        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("EV Dispatch RL — Charging & Ride Scheduler")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.font_sm = pygame.font.SysFont("segoeui", 14)
        self.font_lg = pygame.font.SysFont("segoeui", 26, bold=True)
        self._surface_cache: Optional[np.ndarray] = None

    def render(self, mode: str = "human"):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Atmosphere: soft city dusk gradient feel
        self.screen.fill((18, 28, 42))
        self._draw_panel_bg()
        self._draw_roads()
        self._draw_zones()
        self._draw_ride()
        self._draw_ev()
        self._draw_hud()

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(self.env.metadata.get("render_fps", 8))
            return None

        # rgb_array
        data = pygame.surfarray.array3d(self.screen)
        return np.transpose(data, (1, 0, 2))

    def _draw_panel_bg(self):
        pygame = self.pygame
        # Map area
        pygame.draw.rect(self.screen, (28, 42, 58), (40, 70, 560, 430), border_radius=12)
        # Side panel
        pygame.draw.rect(self.screen, (24, 36, 52), (620, 70, 250, 430), border_radius=12)
        title = self.font_lg.render("EV Dispatch Mission", True, (236, 242, 248))
        self.screen.blit(title, (40, 22))
        sub = self.font_sm.render(
            "Learn when to take rides vs charge under price & congestion",
            True,
            (160, 178, 198),
        )
        self.screen.blit(sub, (40, 50))

    def _draw_roads(self):
        pygame = self.pygame
        road = (55, 72, 92)
        pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for a, b in pairs:
            pygame.draw.line(self.screen, road, ZONE_POS[a], ZONE_POS[b], 10)

    def _draw_zones(self):
        from environment.custom_env import ZONE_NAMES, CHARGER_ZONES

        pygame = self.pygame
        colors = {
            0: (72, 140, 110),
            1: (70, 120, 180),
            2: (150, 110, 70),
            3: (120, 100, 170),
        }
        for z, (x, y) in ZONE_POS.items():
            pygame.draw.circle(self.screen, colors[z], (x, y), 48)
            pygame.draw.circle(self.screen, (230, 236, 244), (x, y), 48, 2)
            label = self.font_sm.render(ZONE_NAMES[z], True, (245, 248, 252))
            rect = label.get_rect(center=(x, y - 8))
            self.screen.blit(label, rect)
            if z in CHARGER_ZONES:
                q = self.env.charger_queues[z]
                bolt = self.font.render("⚡", True, (255, 214, 90))
                self.screen.blit(bolt, (x - 10, y + 8))
                qtxt = self.font_sm.render(f"Q:{q}", True, (255, 230, 150))
                self.screen.blit(qtxt, (x - 14, y + 28))

    def _draw_ride(self):
        pygame = self.pygame
        env = self.env
        if env.pending_origin >= 0 and not env.on_trip:
            ox, oy = ZONE_POS[env.pending_origin]
            dx, dy = ZONE_POS[env.pending_dest]
            pygame.draw.circle(self.screen, (255, 120, 90), (ox, oy - 55), 8)
            pygame.draw.line(self.screen, (255, 150, 120), (ox, oy - 55), (dx, dy - 55), 2)
            pygame.draw.circle(self.screen, (90, 220, 140), (dx, dy - 55), 8)
            tag = self.font_sm.render("REQUEST", True, (255, 180, 150))
            self.screen.blit(tag, (ox - 28, oy - 78))
        if env.on_trip:
            target = env.trip_origin if env.trip_phase == 0 else env.trip_dest
            tx, ty = ZONE_POS[target]
            pygame.draw.circle(self.screen, (255, 210, 70), (tx, ty), 56, 3)
            phase = "PICKUP" if env.trip_phase == 0 else "DROPOFF"
            tag = self.font_sm.render(phase, True, (255, 220, 100))
            self.screen.blit(tag, (tx - 28, ty - 78))

    def _draw_ev(self):
        pygame = self.pygame
        x, y = ZONE_POS[self.env.zone]
        # Car body
        body = pygame.Rect(0, 0, 36, 20)
        body.center = (x, y + 4)
        color = (80, 210, 180) if not self.env.is_charging else (255, 200, 80)
        pygame.draw.rect(self.screen, color, body, border_radius=6)
        pygame.draw.circle(self.screen, (30, 30, 30), (x - 10, y + 14), 4)
        pygame.draw.circle(self.screen, (30, 30, 30), (x + 10, y + 14), 4)
        if self.env.is_charging:
            c = self.font_sm.render("CHARGING", True, (255, 220, 120))
            self.screen.blit(c, (x - 34, y + 24))

    def _draw_hud(self):
        from environment.custom_env import EVDispatchEnv

        pygame = self.pygame
        info = self.env.last_info or self.env._get_info()
        soc = float(info.get("soc", self.env.soc))
        x0, y0 = 640, 90

        lines = [
            f"Step: {info.get('step', 0)} / {self.env.max_steps}",
            f"Zone: {info.get('zone_name', '')}",
            f"SOC: {soc * 100:.1f}%",
            f"Price: {info.get('electricity_price', self.env._electricity_price()):.2f}",
            f"On trip: {info.get('on_trip', False)}",
            f"Charging: {info.get('is_charging', False)}",
            f"Rides done: {info.get('rides_completed', 0)}",
            f"Rides missed: {info.get('rides_missed', 0)}",
            f"Revenue: {info.get('revenue', 0)}",
            f"Energy cost: {info.get('energy_cost', 0)}",
            f"Last reward: {info.get('last_reward', 0):.2f}",
        ]
        hdr = self.font.render("Live Telemetry", True, (220, 230, 240))
        self.screen.blit(hdr, (x0, y0))
        y = y0 + 36
        for line in lines:
            surf = self.font_sm.render(line, True, (190, 205, 220))
            self.screen.blit(surf, (x0, y))
            y += 22

        # SOC bar
        pygame.draw.rect(self.screen, (40, 55, 70), (x0, y + 10, 200, 18), border_radius=4)
        fill_w = int(200 * np.clip(soc, 0, 1))
        bar_color = (70, 200, 120) if soc > 0.3 else (220, 90, 70)
        pygame.draw.rect(
            self.screen, bar_color, (x0, y + 10, fill_w, 18), border_radius=4
        )
        soc_l = self.font_sm.render("Battery", True, (200, 210, 220))
        self.screen.blit(soc_l, (x0, y - 8))

        # Action legend footer
        meanings = EVDispatchEnv.action_meanings()
        act = int(info.get("last_action", 7))
        act_txt = self.font_sm.render(
            f"Action: [{act}] {meanings.get(act, '')}", True, (255, 214, 120)
        )
        self.screen.blit(act_txt, (x0, 470))

    def close(self):
        if self.pygame:
            self.pygame.display.quit()
            self.pygame.quit()
