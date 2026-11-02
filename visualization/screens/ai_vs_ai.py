from __future__ import annotations

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
    format_seconds,
    rounded_rect,
)


class AIVsAIScreen(Screen):
    def __init__(
        self,
        app,
        model_a: str | Path | ModelOption | None,
        model_b: str | Path | ModelOption | None,
        simulations: int = 50,
        a_is_red: bool = True,
    ) -> None:
        super().__init__(app)
        self.simulations = max(2, int(simulations))
        self.option_a = self._resolve_option(model_a, prefer_best=True)
        self.option_b = self._resolve_option(model_b, prefer_best=False)
        self.a_is_red = bool(a_is_red)
        self.players = {1: self.option_a, 2: self.option_b} if self.a_is_red else {1: self.option_b, 2: self.option_a}
        self.agent_a = app.make_agent(self.option_a, self.simulations, seed=201, name=self.option_a.label)
        self.agent_b = app.make_agent(self.option_b, self.simulations, seed=301, name=self.option_b.label)
        self.session = GameSession(
            app.config,
            {1: self.agent_a, 2: self.agent_b},
            human_player=None,
            base_seed=4242,
        )
        self.board = BoardView((42, 138, 650, 544), self.theme)
        speed = app.settings.get("animation_speed", "normal")
        self.board.animate_enabled = speed != "off"
        self.board.animation_duration = 0.16 if speed == "fast" else 0.38
        self.probability_panel = ProbabilityBars(pygame.Rect(746, 458, 468, 248), self.theme)
        self.move_history = MoveHistory(pygame.Rect(64, 708, 610, 64), self.theme)
        self.paused = False
        self._spin_angle = 0.0
        self._last_ai_action: Optional[int] = None
        self._last_ai_player: Optional[int] = None
        self._result_time = 0.0
        self.buttons = [
            Button((716, 736, 116, 42), "New Game", self._restart, self.theme, style="primary", font_size=13),
            Button((840, 736, 112, 42), "Pause", self._toggle_pause, self.theme, style="secondary", font_size=13),
            Button((960, 736, 112, 42), "Swap", self._swap_colors, self.theme, style="secondary", font_size=13),
            Button((1080, 736, 108, 42), "Menu", self._menu, self.theme, style="ghost", font_size=13),
            Button((1122, 68, 30, 28), "-", lambda: self._adjust_sims(-10), self.theme, style="ghost", font_size=15),
            Button((1200, 68, 30, 28), "+", lambda: self._adjust_sims(10), self.theme, style="ghost", font_size=15),
            Button((1112, 150, 92, 36), "Change A", self._change_a, self.theme, style="ghost", font_size=10),
            Button((1112, 234, 92, 36), "Change B", self._change_b, self.theme, style="ghost", font_size=10),
        ]

    def _resolve_option(self, value, prefer_best: bool) -> ModelOption:
        if isinstance(value, ModelOption):
            return value
        if isinstance(value, str) and value in ("random", "tactical"):
            for option in self.app.checkpoint_options(include_special=True):
                if option.key == value:
                    return option
        if value is not None:
            option = self.app.option_for_path(value)
            if option:
                return option
            if Path(value).exists():
                return ModelOption(key=str(value), label=Path(value).stem, detail="checkpoint", path=str(value))
        target = self.app._preferred_model_path() if prefer_best else self.app._initial_or_preferred_path()
        option = self.app.option_for_path(target)
        if option is None:
            option = ModelOption(key=str(target), label=Path(target).stem, detail="checkpoint", path=str(target))
        return option

    def _restart(self) -> None:
        self.session.reset()
        self.board.reset()
        self._last_ai_action = None
        self._last_ai_player = None
        self._result_time = 0.0
        self.probability_panel.selected_action = None
        self.paused = False
        self.buttons[1].label = "Pause"

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.buttons[1].label = "Resume" if self.paused else "Pause"

    def _swap_colors(self) -> None:
        self.app.set_screen(
            AIVsAIScreen(
                self.app,
                self.option_a,
                self.option_b,
                simulations=self.simulations,
                a_is_red=not self.a_is_red,
            )
        )

    def _menu(self) -> None:
        self.session.generation += 1
        self.app.show_main_menu()

    def _adjust_sims(self, delta: int) -> None:
        value = max(10, min(400, self.simulations + delta))
        if value == self.simulations:
            return
        self.app.set_screen(
            AIVsAIScreen(
                self.app,
                self.option_a,
                self.option_b,
                simulations=value,
                a_is_red=self.a_is_red,
            )
        )

    def _choose(self, which: str) -> None:
        self.session.generation += 1
        options = self.app.checkpoint_options(include_special=True)

        def select(option: ModelOption) -> None:
            self.app.set_screen(
                AIVsAIScreen(
                    self.app,
                    option if which == "A" else self.option_a,
                    option if which == "B" else self.option_b,
                    simulations=self.simulations,
                    a_is_red=self.a_is_red,
                )
            )

        self.app.show_model_selector(
            f"SELECT MODEL {which}",
            "Checkpoints and real non-neural baselines are available for this match.",
            options,
            select,
            lambda: self.app.set_screen(
                AIVsAIScreen(
                    self.app,
                    self.option_a,
                    self.option_b,
                    simulations=self.simulations,
                    a_is_red=self.a_is_red,
                )
            ),
        )

    def _change_a(self) -> None:
        self._choose("A")

    def _change_b(self) -> None:
        self._choose("B")

    def on_exit(self) -> None:
        self.session.generation += 1

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._menu()
                return
            if event.key == pygame.K_SPACE:
                self._toggle_pause()
                return
            if event.key == pygame.K_r:
                self._restart()
                return
            if event.key == pygame.K_s:
                self._swap_colors()
                return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def update(self, dt: float) -> None:
        self._spin_angle += dt * 6.0
        event = self.session.poll()
        if event is not None:
            self.board.animate(event.row, event.action, event.player)
            self._last_ai_action = event.action
            self._last_ai_player = event.player
            self.probability_panel.selected_action = event.action
            if self.session.game.is_terminal():
                self._result_time = 0.0
        if self.session.game.is_terminal():
            self._result_time += dt
        if (
            not self.paused
            and not self.session.game.is_terminal()
            and not self.session.thinking
            and not self.board.is_animating
        ):
            self.session.start_ai_turn()

    def _player_option(self, player: int) -> ModelOption:
        return self.players[player]

    def _draw_header(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "ARENA", theme.font(10, bold=True), theme.PURPLE, (44, 24))
        draw_text(surface, "AI vs AI", theme.font(29, bold=True), theme.TEXT, (42, 38))
        draw_text(surface, "Two independent agents. Every move comes from their own search.", theme.font(12), theme.TEXT_DIM, (44, 74))
        draw_text(surface, "MCTS BUDGET", theme.font(9, bold=True), theme.TEXT_FAINT, (1032, 51))
        draw_text(
            surface,
            str(self.simulations),
            theme.font(22, bold=True),
            theme.PURPLE,
            (1155, 48),
            anchor="midtop",
        )
        for button in self.buttons[4:6]:
            button.draw(surface, theme, pygame.mouse.get_pos())
        pygame.draw.line(surface, theme.BORDER_SOFT, (42, 112), (1214, 112), 1)

    def _model_card(self, surface: pygame.Surface, player: int, rect: pygame.Rect) -> None:
        theme = self.theme
        option = self._player_option(player)
        accent = theme.PLAYER_ONE if player == 1 else theme.PLAYER_TWO
        rounded_rect(surface, rect, theme.SURFACE, radius=13, border=1, border_color=theme.BORDER_SOFT)
        draw_token(surface, (rect.left + 30, rect.centery), 18, player, theme)
        draw_text(
            surface,
            "MODEL A" if (player == 1) == self.a_is_red else "MODEL B",
            theme.font(9, bold=True),
            accent,
            (rect.left + 56, rect.top + 12),
        )
        draw_text(surface, option.label, theme.font(17, bold=True), theme.TEXT, (rect.left + 56, rect.top + 25))
        draw_text(surface, option.detail[:52], theme.font(10), theme.TEXT_DIM, (rect.left + 56, rect.top + 49))
        badge = "RANDOM" if option.kind == "random" else "TACTICAL" if option.kind == "tactical" else option.model_type.upper()
        draw_pill(surface, badge, theme, (rect.right - 82, rect.top + 10), accent, 8)

    def _draw_models(self, surface: pygame.Surface) -> None:
        self._model_card(surface, 1, pygame.Rect(716, 132, 386, 76))
        self._model_card(surface, 2, pygame.Rect(716, 216, 386, 76))
        for button in self.buttons[6:8]:
            button.draw(surface, self.theme, pygame.mouse.get_pos())

    def _draw_status(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_panel(surface, pygame.Rect(716, 302, 472, 80), theme)
        if self.session.game.is_terminal():
            winner = self.session.game.get_winner()
            if winner == 0:
                title, color = "DRAW", theme.GOLD
            else:
                option = self._player_option(winner)
                title, color = f"{option.label} WINS", theme.GREEN
            draw_text(surface, title, theme.font(20, bold=True), color, (740, 322))
            draw_text(surface, f"Game complete in {self.session.game.move_count} moves", theme.font(11), theme.TEXT_DIM, (740, 350))
        elif self.paused:
            draw_text(surface, "PAUSED", theme.font(20, bold=True), theme.GOLD, (740, 322))
            draw_text(surface, "Resume to let the agents continue.", theme.font(11), theme.TEXT_DIM, (740, 350))
        else:
            player = self.session.current_player
            option = self._player_option(player)
            color = theme.PLAYER_ONE if player == 1 else theme.PLAYER_TWO
            draw_token(surface, (746, 342), 12, player, theme)
            draw_text(surface, f"{option.label} TO MOVE", theme.font(19, bold=True), color, (768, 322))
            if self.session.thinking:
                draw_spinner(surface, (1162, 342), 11, self._spin_angle, theme.ACCENT, 2)
                draw_text(
                    surface,
                    f"thinking {format_seconds(time.perf_counter() - self.session.thinking_started)}",
                    theme.font(10),
                    theme.TEXT_DIM,
                    (1142, 335),
                    anchor="topright",
                )
            else:
                draw_text(surface, "waiting for animation", theme.font(10), theme.TEXT_FAINT, (1164, 336), anchor="topright")

    def _draw_analysis(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_panel(surface, pygame.Rect(716, 390, 500, 336), theme, "LIVE SEARCH", "Latest MCTS visit distribution")
        if self.session.thinking:
            draw_pill(surface, "SEARCHING", theme, (1042, 410), theme.ACCENT, 9)
        self.probability_panel.selected_action = self._last_ai_action
        self.probability_panel.draw(surface, self.session.last_search, self.session.thinking)
        if self._last_ai_player is not None:
            draw_text(
                surface,
                f"Last decision: Player {self._last_ai_player} ({self._player_option(self._last_ai_player).label})",
                theme.font(10),
                theme.TEXT_FAINT,
                (746, 700),
            )

    def _draw_result_overlay(self, surface: pygame.Surface) -> None:
        if not self.session.game.is_terminal() or self.board.is_animating or self._result_time < 0.15:
            return
        theme = self.theme
        winner = self.session.game.get_winner()
        if winner == 0:
            title, subtitle, color = "DRAW", "Neither agent broke through.", theme.GOLD
        else:
            option = self._player_option(winner)
            title = f"{option.label} WINS"
            subtitle = f"Victory in {self.session.game.move_count} moves."
            color = theme.GREEN
        overlay = pygame.Surface((650, 544), pygame.SRCALPHA)
        overlay.fill((4, 8, 15, 55))
        surface.blit(overlay, (42, 138))
        rect = pygame.Rect(176, 322, 382, 144)
        rounded_rect(surface, rect, (*theme.SURFACE, 245), radius=20, border=1, border_color=color)
        draw_text(surface, title, theme.font(27, bold=True), color, (rect.centerx, rect.top + 26), anchor="midtop")
        draw_text(surface, subtitle, theme.font(12), theme.TEXT_DIM, (rect.centerx, rect.top + 66), anchor="midtop")
        draw_text(surface, "R restart  |  SPACE pause", theme.font(10, bold=True), theme.TEXT_FAINT, (rect.centerx, rect.top + 102), anchor="midtop")

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        self._draw_header(surface)
        self.board.draw(surface, self.session.game, pygame.mouse.get_pos(), interactive=False)
        self._draw_models(surface)
        self._draw_status(surface)
        self._draw_analysis(surface)
        self.move_history.draw(surface, self.session.moves, self.session.players)
        for button in self.buttons[:4]:
            button.draw(surface, theme, pygame.mouse.get_pos())
        self._draw_result_overlay(surface)
