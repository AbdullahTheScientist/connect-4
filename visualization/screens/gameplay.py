from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import pygame

from visualization.components.board import BoardView
from visualization.components.buttons import Button
from visualization.components.panels import MoveHistory, ProbabilityBars
from visualization.controller import GameSession, ModelOption, Screen
from visualization.theme import (
    draw_panel,
    draw_pill,
    draw_spinner,
    draw_text,
    draw_token,
    draw_wrapped_text,
    format_seconds,
    rounded_rect,
)


class GameplayScreen(Screen):
    def __init__(self, app, checkpoint: str | Path | None, simulations: int = 50) -> None:
        super().__init__(app)
        self.simulations = int(simulations)
        self.option = app.option_for_path(checkpoint)
        if self.option is None and checkpoint is not None:
            self.option = ModelOption(
                key=str(checkpoint),
                label=Path(checkpoint).stem,
                detail=str(checkpoint),
                path=str(checkpoint),
                model_type="checkpoint",
            )
        if self.option is None:
            self.option = app.option_for_path(app._preferred_model_path())
        self.checkpoint_path = self.option.path if self.option else None
        self.agent = app.make_agent(self.option, self.simulations, seed=101, name=self.option.label if self.option else "AI")
        self.session = GameSession(app.config, {2: self.agent}, human_player=1, base_seed=1234)
        self.board = BoardView((42, 138, 650, 544), self.theme)
        speed = app.settings.get("animation_speed", "normal")
        self.board.animate_enabled = speed != "off"
        self.board.animation_duration = 0.16 if speed == "fast" else 0.38
        self.probability_panel = ProbabilityBars(pygame.Rect(746, 458, 468, 248), self.theme)
        self.move_history = MoveHistory(pygame.Rect(64, 708, 610, 64), self.theme)
        self.buttons = [
            Button((716, 736, 132, 42), "Restart", self._restart, self.theme, style="primary", font_size=14),
            Button((860, 736, 150, 42), "Change AI", self._change_ai, self.theme, style="secondary", font_size=14),
            Button((1022, 736, 166, 42), "Main Menu", self._menu, self.theme, style="ghost", font_size=14),
        ]
        self._spin_angle = 0.0
        self._last_ai_action: Optional[int] = None
        self._result_time = 0.0

    def _restart(self) -> None:
        self.session.reset()
        self.board.reset()
        self._last_ai_action = None
        self._result_time = 0.0
        self.probability_panel.selected_action = None

    def _menu(self) -> None:
        self.session.generation += 1
        self.app.show_main_menu()

    def _change_ai(self) -> None:
        self.session.generation += 1
        options = self.app.checkpoint_options()
        self.app.show_model_selector(
            "CHANGE OPPONENT",
            "Pick a different saved network. The game will restart.",
            options,
            lambda option: self.app.show_gameplay(option.path, simulations=self.app.default_simulations),
            lambda: self.app.show_gameplay(self.checkpoint_path, simulations=self.simulations),
        )

    def on_exit(self) -> None:
        self.session.generation += 1

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._menu()
                return
            if event.key == pygame.K_r:
                self._restart()
                return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.session.can_human_move() and not self.board.is_animating:
                column = self.board.column_from_position(event.pos, self.session.game)
                if column is not None:
                    move = self.session.human_move(column)
                    if move:
                        self.board.animate(move.row, move.action, move.player)

    def update(self, dt: float) -> None:
        self._spin_angle += dt * 6.0
        event = self.session.poll()
        if event is not None:
            self.board.animate(event.row, event.action, event.player)
            if event.ai:
                self._last_ai_action = event.action
                self.probability_panel.selected_action = event.action
            if self.session.game.is_terminal():
                self._result_time = 0.0
        if self.session.game.is_terminal():
            self._result_time += dt
        if (
            not self.session.game.is_terminal()
            and not self.session.thinking
            and not self.board.is_animating
            and self.session.current_player != self.session.human_player
        ):
            self.session.start_ai_turn()

    def _draw_header(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "PLAY", theme.font(10, bold=True), theme.ACCENT, (44, 24))
        draw_text(surface, "Human vs AI", theme.font(29, bold=True), theme.TEXT, (42, 38))
        draw_text(surface, "You are Red and move first. The AI searches with MCTS.", theme.font(12), theme.TEXT_DIM, (44, 74))
        label = self.option.label if self.option else "checkpoint"
        draw_pill(surface, f"AI  {label}", theme, (736, 34), theme.ACCENT, 11)
        draw_pill(surface, f"{self.simulations} simulations", theme, (1040, 34), theme.PURPLE, 11)
        pygame.draw.line(surface, theme.BORDER_SOFT, (42, 112), (1214, 112), 1)

    def _draw_players(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_panel(surface, pygame.Rect(716, 138, 472, 84), theme)
        draw_token(surface, (748, 180), 21, 1, theme)
        draw_text(surface, "YOU", theme.font(12, bold=True), theme.TEXT, (780, 158))
        draw_text(surface, "Human", theme.font(11), theme.TEXT_DIM, (780, 176))
        pygame.draw.line(surface, theme.BORDER_SOFT, (920, 152), (920, 208), 1)
        draw_token(surface, (948, 180), 21, 2, theme)
        draw_text(surface, "ALPHAZERO", theme.font(12, bold=True), theme.TEXT, (978, 158))
        draw_text(surface, self.option.label if self.option else "AI", theme.font(11), theme.TEXT_DIM, (978, 176))

    def _turn_text(self) -> tuple[str, str, tuple[int, int, int]]:
        session = self.session
        if session.game.is_terminal():
            winner = session.game.get_winner()
            if winner == 0:
                return "DRAW", "The board is full. No winner.", self.theme.GOLD
            if winner == session.human_player:
                return "YOU WIN", "A clean win against the neural network.", self.theme.GREEN
            return "AI WINS", "The network found the winning line.", self.theme.RED
        if session.thinking:
            return "AI THINKING", f"Exploring {self.simulations} MCTS simulations...", self.theme.ACCENT
        if session.current_player == session.human_player:
            return "YOUR MOVE", "Click a column to drop your red piece.", self.theme.PLAYER_ONE
        return "AI TURN", "Preparing search.", self.theme.PLAYER_TWO

    def _draw_status(self, surface: pygame.Surface) -> None:
        theme = self.theme
        panel = pygame.Rect(716, 236, 472, 146)
        draw_panel(surface, panel, theme)
        title, subtitle, color = self._turn_text()
        if self.session.thinking:
            draw_spinner(surface, (754, 278), 13, self._spin_angle, theme.ACCENT, 3)
            draw_text(surface, title, theme.font(21, bold=True), color, (780, 261))
        else:
            pygame.draw.circle(surface, color, (750, 274), 7)
            draw_text(surface, title, theme.font(21, bold=True), color, (768, 261))
        draw_text(surface, subtitle, theme.font(12), theme.TEXT_DIM, (744, 296))
        detail_y = 324
        if self.session.last_error:
            draw_text(surface, f"Error: {self.session.last_error}", theme.font(11), theme.RED, (744, detail_y))
        elif self._last_ai_action is not None:
            draw_text(
                surface,
                f"Last AI move: column {self._last_ai_action + 1}",
                theme.font(12),
                theme.TEXT,
                (744, detail_y),
            )
            if self.session.last_search is not None and self.session.last_search.simulations:
                value = self.session.last_search.root_value
                draw_text(
                    surface,
                    f"Search verdict: {value:+.2f}  |  {self.session.last_search.simulations} simulations",
                    theme.font(11),
                    theme.TEXT_DIM,
                    (744, detail_y + 20),
                )
        else:
            draw_text(surface, "The board is connected to the real game engine.", theme.font(11), theme.TEXT_FAINT, (744, detail_y))
        if self.session.game.is_terminal():
            pygame.draw.circle(surface, color, (1150, 274), 21, width=2)
            draw_text(surface, "GAME", theme.font(9, bold=True), color, (1150, 267), anchor="midtop")
            draw_text(surface, "OVER", theme.font(9, bold=True), color, (1150, 279), anchor="midtop")

    def _draw_analysis(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_panel(surface, pygame.Rect(716, 390, 500, 336), theme, "AI SEARCH", "What the search explored on its latest turn")
        if self.session.thinking:
            draw_pill(surface, "SEARCHING", theme, (1042, 410), theme.ACCENT, 9)
        self.probability_panel.selected_action = self._last_ai_action
        self.probability_panel.draw(surface, self.session.last_search, self.session.thinking)
        draw_text(
            surface,
            "Values are from the AI's current-player perspective. "+chr(43)+"1 is winning, -1 is losing.",
            theme.font(10),
            theme.TEXT_FAINT,
            (746, 700),
        )

    def _draw_result_overlay(self, surface: pygame.Surface) -> None:
        if not self.session.game.is_terminal() or self.board.is_animating or self._result_time < 0.15:
            return
        theme = self.theme
        title, subtitle, color = self._turn_text()
        overlay = pygame.Surface((650, 544), pygame.SRCALPHA)
        overlay.fill((4, 8, 15, 60))
        surface.blit(overlay, (42, 138))
        rect = pygame.Rect(176, 322, 382, 144)
        rounded_rect(surface, rect, (*theme.SURFACE, 245), radius=20, border=1, border_color=color)
        draw_text(surface, title, theme.font(31, bold=True), color, (rect.centerx, rect.top + 25), anchor="midtop")
        draw_text(surface, subtitle, theme.font(12), theme.TEXT_DIM, (rect.centerx, rect.top + 70), anchor="midtop")
        draw_text(surface, "R to restart", theme.font(11, bold=True), theme.TEXT_FAINT, (rect.centerx, rect.top + 104), anchor="midtop")

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        self._draw_header(surface)
        interactive = self.session.can_human_move() and not self.board.is_animating
        self.board.draw(surface, self.session.game, pygame.mouse.get_pos(), interactive=interactive)
        self._draw_players(surface)
        self._draw_status(surface)
        self._draw_analysis(surface)
        self.move_history.draw(surface, self.session.moves, self.session.players)
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
        self._draw_result_overlay(surface)
        if self.session.thinking:
            draw_text(
                surface,
                f"{format_seconds(time.perf_counter() - self.session.thinking_started)}",
                theme.font(10),
                theme.TEXT_FAINT,
                (1214, 64),
                anchor="topright",
            )
