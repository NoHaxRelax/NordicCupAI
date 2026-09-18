"""Viewer pacing and buttons; simulation time steps remain unchanged."""

import math

import pygame


class PlaybackControls:
    SPEEDS = (1, 2, 5, 10, 20)
    HEIGHT = 52
    MAX_FRAME_WORK_SECONDS = .04

    def __init__(self, speed=1):
        self.speed = 1
        self.pending_seconds = 0.
        self.actual_speed = 0.
        self._rate_elapsed = 0.
        self._rate_simulated = 0.
        self.select(speed)

    def select(self, speed):
        if speed not in self.SPEEDS:
            raise ValueError(f"speed must be one of {self.SPEEDS}")
        if speed != self.speed:
            self.speed = speed
            # Changing speed must not drain an old high-speed backlog first.
            self.pending_seconds = 0.
            self._rate_elapsed = self._rate_simulated = 0.
            self.actual_speed = 0.

    def advance(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and non-negative")
        # Bound catch-up after slow frames/window dragging. No simulation tick
        # is skipped: lag simply lowers the achieved wall-clock speed.
        self.pending_seconds = min(self.pending_seconds + elapsed * self.speed,
                                   .25 * self.speed)
        self._rate_elapsed += elapsed
        if self._rate_elapsed >= .5:
            self.actual_speed = self._rate_simulated / self._rate_elapsed
            self._rate_elapsed = self._rate_simulated = 0.

    def consume_step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.pending_seconds + 1e-9 < dt:
            return False
        self.pending_seconds = max(0., self.pending_seconds - dt)
        self._rate_simulated += dt
        return True

    def buttons(self, size):
        width, height = size
        button_width = min(52, max(28, (width - 86) // len(self.SPEEDS) - 6))
        return [(speed, pygame.Rect(72 + i * (button_width + 6),
                                   max(0, height - self.HEIGHT) + 10, button_width, 32))
                for i, speed in enumerate(self.SPEEDS)]

    def handle_event(self, event, size):
        if event.type == pygame.KEYDOWN:
            keys = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5)
            if event.key in keys:
                self.select(self.SPEEDS[keys.index(event.key)])
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for speed, rect in self.buttons(size):
                if rect.collidepoint(event.pos):
                    self.select(speed)
                    return True
        return False

    def draw(self, screen, font):
        top = max(0, screen.get_height() - self.HEIGHT)
        pygame.draw.rect(screen, (18, 28, 39), (0, top, screen.get_width(), self.HEIGHT))
        pygame.draw.line(screen, (53, 68, 82), (0, top), (screen.get_width(), top))
        screen.blit(font.render("Speed", True, (224, 233, 241)), (12, top + 17))
        buttons = self.buttons(screen.get_size())
        for speed, rect in buttons:
            selected = speed == self.speed
            pygame.draw.rect(screen, (29, 111, 107) if selected else (37, 50, 66), rect, border_radius=5)
            pygame.draw.rect(screen, (92, 220, 189) if selected else (90, 111, 134),
                             rect, width=2 if selected else 1, border_radius=5)
            label = font.render(f"{speed}x", True, (240, 246, 251))
            screen.blit(label, label.get_rect(center=rect.center))
        status = font.render(f"Actual: {self.actual_speed:.1f}x   |   Keys 1-5 select speed",
                             True, (160, 178, 194))
        x = buttons[-1][1].right + 18
        if x + status.get_width() <= screen.get_width() - 8:
            screen.blit(status, (x, top + 17))
