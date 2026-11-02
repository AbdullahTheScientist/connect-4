"""Animated Connect Four board renderer. Game state is never duplicated here."""
from __future__ import annotations

import math
import time
from typing import Optional

import pygame

from game.connect4 import ConnectFour
from visualization.theme import Theme, draw_text, draw_token, ease_out_cubic, rounded_rect


class BoardView:
    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int], theme: Theme) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        margin = 18
        self.cell = int(min((self.rect.width - margin * 2) / 7, (self.rect.height - margin * 2) / 6))
        board_width = self.cell * 7
        board_height = self.cell * 6
        self.board_rect = pygame.Rect(0, 0, board_width, board_height)
        self.board_rect.center = self.rect.center
        self.animation_start = 0.0
        self.animation_duration = 0.38
        self.animation_row: Optional[int] = None
        self.animation_col: Optional[int] = None
        self.animation_player = 0
        self.animate_enabled = True
        self.hover_column: Optional[int] = None

    def reset(self) -> None:
        self.animation_row = None
        self.animation_col = None
        self.animation_player = 0
        self.animation_start = 0.0
        self.hover_column = None

    @property
    def is_animating(self) -> bool:
        if self.animation_row is None:
            return False
        return (time.perf_counter() - self.animation_start) < self.animation_duration

    def animate(self, row: int, col: int, player: int) -> None:
        self.animation_row = int(row)
        self.animation_col = int(col)
        self.animation_player = int(player)
        self.animation_start = time.perf_counter()
        if not self.animate_enabled:
            self.animation_duration = 0.0

    def column_from_position(self, position: tuple[int, int], game: ConnectFour) -> Optional[int]:
        if not self.board_rect.collidepoint(position):
            return None
        col = int((position[0] - self.board_rect.left) // self.cell)
        if 0 <= col < 7 and game.is_legal(col):
            return col
        return None

    def center_for(self, row: int, col: int) -> tuple[int, int]:
        return (
            self.board_rect.left + int((col + 0.5) * self.cell),
            self.board_rect.top + int((row + 0.5) * self.cell),
        )

    def draw(
        self,
        surface: pygame.Surface,
        game: ConnectFour,
        mouse_pos: tuple[int, int],
        interactive: bool = False,
    ) -> None:
        theme = self.theme
        shadow = self.board_rect.inflate(26, 26).move(0, 8)
        rounded_rect(surface, shadow, (2, 5, 12, 145), radius=26)
        border = self.board_rect.inflate(20, 20)
        rounded_rect(surface, border, theme.BOARD_BLUE_DARK, radius=22, border=1, border_color=(47, 113, 162))
        rounded_rect(surface, self.board_rect, theme.BOARD_BLUE, radius=16)

        column = self.column_from_position(mouse_pos, game) if interactive else None
        self.hover_column = column
        if column is not None:
            x = self.board_rect.left + column * self.cell
            preview = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
            pygame.draw.rect(preview, (*theme.ACCENT, 22), preview.get_rect(), border_radius=14)
            surface.blit(preview, (x, self.board_rect.top - self.cell + 4))
            draw_token(
                surface,
                (x + self.cell // 2, self.board_rect.top - self.cell // 2 + 4),
                int(self.cell * 0.34),
                game.get_current_player(),
                theme,
                alpha=110,
            )

        animating = self.is_animating
        for row in range(6):
            for col in range(7):
                center = self.center_for(row, col)
                pygame.draw.circle(surface, theme.BOARD_BLUE_DARK, center, int(self.cell * 0.39))
                pygame.draw.circle(surface, (10, 27, 46), center, int(self.cell * 0.35))
                if (
                    animating
                    and row == self.animation_row
                    and col == self.animation_col
                ):
                    continue
                player = int(game.board[row, col])
                if player:
                    draw_token(surface, center, int(self.cell * 0.34), player, theme)

        if animating and self.animation_row is not None and self.animation_col is not None:
            t = (time.perf_counter() - self.animation_start) / max(1e-6, self.animation_duration)
            t = max(0.0, min(1.0, t))
            eased = ease_out_cubic(t)
            start_y = self.board_rect.top - self.cell // 2
            final_y = self.center_for(self.animation_row, self.animation_col)[1]
            y = int(start_y + (final_y - start_y) * eased)
            x = self.center_for(self.animation_row, self.animation_col)[0]
            draw_token(surface, (x, y), int(self.cell * 0.34), self.animation_player, theme)

        winning_line = game.get_winning_line()
        if winning_line and not animating:
            pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 5.0)
            points = [self.center_for(row, col) for row, col in winning_line]
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            if len(points) >= 2:
                pygame.draw.line(overlay, (255, 255, 255, int(70 + pulse * 80)), points[0], points[-1], 4)
            surface.blit(overlay, (0, 0))
            for row, col in winning_line:
                draw_token(
                    surface,
                    self.center_for(row, col),
                    int(self.cell * 0.34),
                    game.board[row, col],
                    theme,
                    highlight=True,
                )

        if game.last_action is not None and not animating and not winning_line:
            col = int(game.last_action)
            top = self.board_rect.top - 4
            center_x = self.board_rect.left + int((col + 0.5) * self.cell)
            pygame.draw.circle(surface, theme.WHITE, (center_x, top), 3)

        for col in range(7):
            x = self.board_rect.left + int((col + 0.5) * self.cell)
            draw_text(
                surface,
                str(col + 1),
                theme.font(11, bold=True),
                theme.TEXT_FAINT,
                (x, self.board_rect.bottom + 9),
                anchor="midtop",
            )
