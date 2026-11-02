from __future__ import annotations

from pathlib import Path

import pygame

from visualization.components.buttons import MenuButton
from visualization.controller import Screen
from visualization.theme import (
    draw_orb,
    draw_panel,
    draw_pill,
    draw_text,
    draw_token,
    rounded_rect,
)


class MainMenuScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.buttons: list[MenuButton] = []
        self.records = app.checkpoint_records()
        self._build_buttons()

    def _build_buttons(self) -> None:
        entries = [
            (
                "Play Human vs AI",
                "Take on a checkpoint with live MCTS analysis",
                self._play_human,
                self.theme.ACCENT,
                "01",
            ),
            (
                "Watch AI vs AI",
                "Two real agents, configurable search budget",
                self._watch_ai,
                self.theme.PURPLE,
                "02",
            ),
            (
                "Compare Initial vs Trained",
                "Batch evaluation and live before/after match",
                self._compare,
                self.theme.GOLD,
                "03",
            ),
            (
                "Model Checkpoint Manager",
                "Inspect, select and compare saved networks",
                self._checkpoints,
                self.theme.GREEN,
                "04",
            ),
            (
                "Training Monitor",
                "Launch, pause and observe the real trainer",
                self._training,
                self.theme.ACCENT,
                "05",
            ),
            (
                "Evaluation Dashboard",
                "Loss curves, win-rate history and promotions",
                self._evaluation,
                self.theme.GOLD,
                "06",
            ),
            ("Settings", "Device, search budget and animation", self._settings, self.theme.TEXT_DIM, "07"),
            ("Exit", "Close AlphaZero Lab", self._exit, self.theme.RED, "08"),
        ]
        self.buttons = []
        y = 156
        for title, subtitle, callback, accent, number in entries:
            rect = pygame.Rect(548, y, 668, 64)
            self.buttons.append(MenuButton(rect, title, subtitle, callback, accent, number))
            y += 74

    def _play_human(self) -> None:
        options = self.app.checkpoint_options()
        if not options:
            self.app.toast("No checkpoints found")
            return
        self.app.show_model_selector(
            "CHOOSE YOUR OPPONENT",
            "Every option is an actual saved network. Select one to start Human vs AI.",
            options,
            lambda option: self.app.show_gameplay(option.path, simulations=self.app.default_simulations),
            self.app.show_main_menu,
        )

    def _watch_ai(self) -> None:
        path_a = self.app._preferred_model_path()
        path_b = self.app._initial_or_preferred_path()
        self.app.show_ai_vs_ai(path_a, path_b, simulations=self.app.default_simulations)

    def _compare(self) -> None:
        self.app.show_comparison()

    def _checkpoints(self) -> None:
        self.app.show_checkpoint_manager()

    def _training(self) -> None:
        self.app.show_training_monitor()

    def _evaluation(self) -> None:
        self.app.show_evaluation_dashboard()

    def _settings(self) -> None:
        self.app.show_settings()

    def _exit(self) -> None:
        self.app.running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.running = False
                return
            if pygame.K_1 <= event.key <= pygame.K_8:
                index = event.key - pygame.K_1
                if index < len(self.buttons):
                    self.buttons[index].callback()
                return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                break

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_orb(surface, (82, 92), 200, theme.ACCENT, alpha=25)
        draw_orb(surface, (610, 710), 250, theme.PURPLE, alpha=14)
        draw_text(surface, "ALPHAZERO LAB", theme.font(12, bold=True), theme.ACCENT, (64, 42))
        draw_text(surface, "CONNECT", theme.font(58, bold=True), theme.TEXT, (62, 69))
        draw_text(surface, "FOUR", theme.font(58, bold=True), theme.ACCENT, (62, 119))
        draw_text(
            surface,
            "Self-play reinforcement learning, from random weights to a playing agent.",
            theme.font(14),
            theme.TEXT_DIM,
            (66, 193),
        )

        panel = pygame.Rect(62, 240, 432, 372)
        draw_panel(surface, panel, theme, "SYSTEM", "Live state from the local experiment")
        records = self.records
        best_records = [r for r in records if r.get("model_type") == "best"]
        best_iter = max((r.get("iteration", 0) for r in best_records), default=None)
        experiments = len({r.get("experiment") for r in records}) if records else 0
        draw_text(surface, "CHECKPOINTS", theme.font(10, bold=True), theme.TEXT_FAINT, (86, 335))
        draw_text(surface, f"{len(records):02d}", theme.font(34, bold=True), theme.TEXT, (86, 351))
        draw_text(surface, "EXPERIMENTS", theme.font(10, bold=True), theme.TEXT_FAINT, (206, 335))
        draw_text(surface, f"{experiments:02d}", theme.font(34, bold=True), theme.TEXT, (206, 351))
        draw_text(surface, "BEST ITERATION", theme.font(10, bold=True), theme.TEXT_FAINT, (86, 415))
        draw_text(
            surface,
            f"{best_iter:04d}" if best_iter is not None else "--",
            theme.font(34, bold=True),
            theme.GOLD,
            (86, 431),
        )
        draw_text(surface, "COMPUTE", theme.font(10, bold=True), theme.TEXT_FAINT, (238, 415))
        draw_text(surface, str(self.app.device).upper(), theme.font(23, bold=True), theme.ACCENT, (238, 437))

        pygame.draw.line(surface, theme.BORDER_SOFT, (86, 499), (470, 499), 1)
        draw_text(surface, "HOW IT LEARNS", theme.font(11, bold=True), theme.TEXT_FAINT, (86, 515))
        steps = [
            "1  Best network plays itself through MCTS",
            "2  Visits become policy targets, outcomes value targets",
            "3  Candidate trains, then faces the best in the arena",
            "4  Only a passing candidate replaces best.pt",
        ]
        y = 537
        for step in steps:
            draw_text(surface, step, theme.font(11), theme.TEXT_DIM, (86, y))
            y += 18

        mini = pygame.Rect(328, 274, 134, 72)
        rounded_rect(surface, mini, theme.BOARD_BLUE_DARK, radius=12)
        for row in range(2):
            for col in range(4):
                center = (mini.left + 20 + col * 31, mini.top + 22 + row * 30)
                pygame.draw.circle(surface, (9, 25, 42), center, 11)
                if (row, col) in ((1, 0), (0, 1), (1, 1), (1, 2)):
                    player = 1 if (row, col) in ((1, 0), (1, 1)) else 2
                    draw_token(surface, center, 9, player, theme)
        draw_pill(surface, "MCTS + CNN", theme, (328, 358), theme.ACCENT, 9)

        draw_text(surface, "COMMAND CENTER", theme.font(12, bold=True), theme.ACCENT, (550, 78))
        draw_text(surface, "Choose a workspace", theme.font(31, bold=True), theme.TEXT, (548, 99))
        draw_text(
            surface,
            "All gameplay, evaluation and monitoring is connected to real local artifacts.",
            theme.font(13),
            theme.TEXT_DIM,
            (550, 133),
        )
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())

        footer_y = 768
        pygame.draw.line(surface, theme.BORDER_SOFT, (64, footer_y - 12), (1216, footer_y - 12), 1)
        draw_text(
            surface,
            "PYTORCH  |  PYGAME  |  TENSORBOARD",
            theme.font(10, bold=True),
            theme.TEXT_FAINT,
            (64, footer_y),
        )
        draw_text(
            surface,
            "Keys 1-8 navigate  |  ESC exits",
            theme.font(10),
            theme.TEXT_FAINT,
            (1216, footer_y),
            anchor="topright",
        )
