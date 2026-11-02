"""Composite panels used across screens."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pygame

from mcts.search import SearchResult
from visualization.theme import (
    Theme,
    draw_panel,
    draw_pill,
    draw_text,
    draw_wrapped_text,
    format_number,
    format_percent,
    rounded_rect,
)


class ProbabilityBars:
    def __init__(self, rect: pygame.Rect, theme: Theme) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        self.selected_action: Optional[int] = None

    def draw(self, surface: pygame.Surface, result: SearchResult | None, thinking: bool = False) -> None:
        theme = self.theme
        y = self.rect.top
        header = theme.font(13, bold=True)
        small = theme.font(11)
        draw_text(surface, "MCTS ACTION PROBABILITY", header, theme.TEXT_DIM, (self.rect.left, y))
        draw_text(
            surface,
            "bar = visit share   tick = network prior",
            small,
            theme.TEXT_FAINT,
            (self.rect.right, y + 2),
            anchor="topright",
        )
        y += 27
        if result is None:
            draw_wrapped_text(
                surface,
                "The AI's search statistics will appear here after it thinks.",
                theme.font(13),
                theme.TEXT_FAINT,
                (self.rect.left, y + 8),
                self.rect.width,
            )
            return
        probabilities = result.action_probs
        priors = result.nn_prior
        selected = self.selected_action
        if selected is None and probabilities.sum() > 0:
            selected = int(probabilities.argmax())
        bar_height = 18
        gap = 6
        label_width = 34
        value_width = 50
        for action in range(7):
            row_rect = pygame.Rect(self.rect.left, y, self.rect.width, bar_height)
            is_selected = action == selected
            if is_selected:
                rounded_rect(surface, row_rect.inflate(10, 4), (*theme.ACCENT, 16), radius=7)
            draw_text(
                surface,
                f"C{action + 1}",
                theme.font(12, bold=is_selected),
                theme.ACCENT if is_selected else theme.TEXT_DIM,
                (self.rect.left + 2, row_rect.centery),
                anchor="midleft",
            )
            track = pygame.Rect(
                self.rect.left + label_width,
                row_rect.top + 4,
                self.rect.width - label_width - value_width,
                14,
            )
            rounded_rect(surface, track, (11, 17, 28), radius=6)
            width = int(track.width * float(probabilities[action]))
            if width > 0:
                color = theme.ACCENT if is_selected else theme.ACCENT_DARK
                rounded_rect(surface, pygame.Rect(track.left, track.top, max(2, width), track.height), color, radius=6)
            prior_x = track.left + int(track.width * float(priors[action]))
            pygame.draw.line(surface, theme.GOLD, (prior_x, track.top - 2), (prior_x, track.bottom + 2), 2)
            draw_text(
                surface,
                f"{probabilities[action] * 100:4.1f}%",
                theme.font(11, bold=is_selected),
                theme.TEXT if is_selected else theme.TEXT_DIM,
                (self.rect.right - 2, row_rect.centery),
                anchor="midright",
            )
            y += bar_height + gap
        y += 5
        draw_text(surface, "VISITS", theme.font(11, bold=True), theme.TEXT_FAINT, (self.rect.left, y))
        x = self.rect.left + 55
        for action in range(7):
            text = f"{int(result.visit_counts[action])}"
            draw_text(
                surface,
                text,
                theme.font(11, bold=action == selected),
                theme.ACCENT if action == selected else theme.TEXT_FAINT,
                (x + action * (self.rect.width - 55) / 7, y),
                anchor="topleft",
            )
        y += 22
        value = result.root_value
        sign = "+" if value >= 0 else ""
        draw_text(
            surface,
            f"Position value (AI perspective): {sign}{value:.2f}",
            theme.font(12),
            theme.GREEN if value > 0.05 else theme.RED if value < -0.05 else theme.TEXT_DIM,
            (self.rect.left, y),
        )
        draw_text(
            surface,
            f"{result.simulations} simulated futures",
            theme.font(12),
            theme.TEXT_FAINT,
            (self.rect.right, y),
            anchor="topright",
        )


class MoveHistory:
    def __init__(self, rect: pygame.Rect, theme: Theme, max_visible: int = 10) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        self.max_visible = max_visible

    def draw(self, surface: pygame.Surface, moves: Sequence[int], players: Sequence[int] = ()) -> None:
        theme = self.theme
        draw_text(surface, "MOVE HISTORY", theme.font(12, bold=True), theme.TEXT_DIM, (self.rect.left, self.rect.top))
        if not moves:
            draw_text(
                surface,
                "No moves yet",
                theme.font(12),
                theme.TEXT_FAINT,
                (self.rect.left, self.rect.top + 25),
            )
            return
        display = list(zip(moves, players)) if players else [(m, 1 + i % 2) for i, m in enumerate(moves)]
        display = display[-self.max_visible :]
        x = self.rect.left
        y = self.rect.top + 24
        per_row = 7
        for index, (move, player) in enumerate(display):
            col = index % per_row
            row = index // per_row
            chip = pygame.Rect(x + col * 43, y + row * 27, 37, 22)
            color = theme.PLAYER_ONE if player == 1 else theme.PLAYER_TWO
            rounded_rect(surface, chip, (*color, 22), radius=6, border=1, border_color=(*color, 80))
            draw_text(surface, str(move + 1), theme.font(10, bold=True), color, chip.center, anchor="center")
        if len(moves) > self.max_visible:
            draw_text(
                surface,
                f"+{len(moves) - self.max_visible} earlier",
                theme.font(10),
                theme.TEXT_FAINT,
                (self.rect.right, self.rect.top),
                anchor="topright",
            )


def draw_model_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
    pills: Sequence[tuple[str, tuple[int, int, int]]] = (),
) -> None:
    rounded_rect(surface, rect, theme.SURFACE_ALT, radius=14, border=1, border_color=theme.BORDER_SOFT)
    pygame.draw.rect(surface, accent, pygame.Rect(rect.left + 14, rect.top + 16, 5, 30), border_radius=2)
    draw_text(surface, title, theme.font(16, bold=True), theme.TEXT, (rect.left + 30, rect.top + 13))
    draw_text(surface, subtitle, theme.font(11), theme.TEXT_DIM, (rect.left + 30, rect.top + 36))
    x = rect.left + 30
    for text, color in pills:
        pill = draw_pill(surface, text, theme, (x, rect.bottom - 27), color=color, font_size=10)
        x = pill.right + 6


class ScrollState:
    def __init__(self, total: int = 0, visible: int = 1) -> None:
        self.offset = 0
        self.total = 0
        self.visible = max(1, visible)
        self.set_total(total)

    def set_total(self, total: int) -> None:
        self.total = max(0, total)
        max_offset = max(0, self.total - self.visible)
        self.offset = min(self.offset, max_offset)

    def scroll(self, delta: int) -> None:
        max_offset = max(0, self.total - self.visible)
        self.offset = max(0, min(max_offset, self.offset + delta))

    def wheel(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEWHEEL:
            self.scroll(-event.y)
            return True
        return False


def draw_empty_state(surface: pygame.Surface, rect: pygame.Rect, theme: Theme, title: str, body: str) -> None:
    center = rect.center
    pygame.draw.circle(surface, theme.SURFACE_ALT, center, 28)
    pygame.draw.circle(surface, theme.TEXT_FAINT, center, 28, width=1)
    draw_text(surface, "?", theme.font(22, bold=True), theme.TEXT_DIM, center, anchor="center")
    draw_text(surface, title, theme.font(17, bold=True), theme.TEXT, (center[0], center[1] + 50), anchor="midtop")
    lines = draw_wrapped_text(
        surface,
        body,
        theme.font(12),
        theme.TEXT_DIM,
        (rect.left + 30, center[1] + 78),
        rect.width - 60,
        line_height=19,
    )
    _ = lines
