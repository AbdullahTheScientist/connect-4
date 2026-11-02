from __future__ import annotations

import platform
import sys
from pathlib import Path

import pygame
import torch

from utils.config import PROJECT_ROOT, resolve_path
from visualization.components.buttons import Button
from visualization.controller import Screen
from visualization.theme import draw_panel, draw_pill, draw_text, rounded_rect


class SettingsScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.buttons = [
            Button((60, 736, 150, 42), "Main Menu", self._menu, self.theme, style="ghost", font_size=13),
            Button((898, 254, 44, 38), "-", lambda: self._adjust_sims(-10), self.theme, style="secondary", font_size=16),
            Button((1044, 254, 44, 38), "+", lambda: self._adjust_sims(10), self.theme, style="secondary", font_size=16),
            Button((806, 460, 206, 40), "Cycle Animation", self._cycle_animation, self.theme, style="secondary", font_size=13),
            Button((1022, 460, 168, 40), "Save Settings", self._save, self.theme, style="primary", font_size=13),
            Button((806, 580, 384, 40), "Open Project Folder", self._open_folder, self.theme, style="ghost", font_size=13),
        ]

    def _menu(self) -> None:
        self.app.save_ui_settings()
        self.app.show_main_menu()

    def _adjust_sims(self, delta: int) -> None:
        self.app.default_simulations = max(10, min(400, self.app.default_simulations + delta))
        self.app.settings["simulations"] = self.app.default_simulations

    def _cycle_animation(self) -> None:
        order = ["normal", "fast", "off"]
        current = self.app.settings.get("animation_speed", "normal")
        index = order.index(current) if current in order else 0
        self.app.settings["animation_speed"] = order[(index + 1) % len(order)]

    def _save(self) -> None:
        self.app.save_ui_settings()
        self.app.toast("UI settings saved")

    def _open_folder(self) -> None:
        try:
            import subprocess

            if sys.platform.startswith("win"):
                import os

                os.startfile(PROJECT_ROOT)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(PROJECT_ROOT)])
            else:
                subprocess.Popen(["xdg-open", str(PROJECT_ROOT)])
        except Exception as exc:
            self.app.toast(f"Could not open folder: {exc}")

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
            return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "SYSTEM", theme.font(10, bold=True), theme.TEXT_DIM, (66, 38))
        draw_text(surface, "Settings", theme.font(31, bold=True), theme.TEXT, (64, 55))
        draw_text(
            surface,
            "Runtime facts and local UI preferences. Training hyperparameters live in YAML.",
            theme.font(13),
            theme.TEXT_DIM,
            (66, 96),
        )

        draw_panel(surface, pygame.Rect(60, 138, 680, 560), theme, "RUNTIME", "Actual values from this machine")
        runtime = [
            ("Python", platform.python_version()),
            ("PyTorch", torch.__version__),
            ("Compute device", str(self.app.device).upper()),
            ("CUDA available", "YES" if torch.cuda.is_available() else "NO"),
            ("CPU threads", str(torch.get_num_threads())),
            ("Platform", platform.platform()),
            ("Config file", str(self.app.config.get("_config_path", "configs/default.yaml"))),
            ("Checkpoint root", str(resolve_path(self.app.config, "checkpoints"))),
            ("Log root", str(resolve_path(self.app.config, "logs"))),
            ("Results root", str(resolve_path(self.app.config, "results"))),
        ]
        y = 226
        for label, value in runtime:
            draw_text(surface, label.upper(), theme.font(9, bold=True), theme.TEXT_FAINT, (84, y))
            draw_text(surface, str(value), theme.font(11), theme.TEXT, (250, y - 2))
            y += 36
        pygame.draw.line(surface, theme.BORDER_SOFT, (84, 592), (716, 592), 1)
        draw_text(
            surface,
            "The neural network, MCTS, arena and trainer run in this Python environment.",
            theme.font(11),
            theme.TEXT_DIM,
            (84, 610),
        )
        draw_text(
            surface,
            "UI settings apply only to gameplay; training hyperparameters are never altered by the GUI.",
            theme.font(10),
            theme.TEXT_FAINT,
            (84, 632),
        )
        draw_text(surface, f"Project root: {PROJECT_ROOT}", theme.font(9), theme.TEXT_FAINT, (84, 660))

        draw_panel(surface, pygame.Rect(760, 138, 456, 566), theme, "INTERFACE", "Applied to newly opened games")
        draw_text(surface, "DEFAULT MCTS SIMULATIONS", theme.font(9, bold=True), theme.TEXT_FAINT, (788, 226))
        draw_text(surface, str(self.app.default_simulations), theme.font(38, bold=True), theme.ACCENT, (806, 250))
        draw_text(
            surface,
            "search budget per AI move",
            theme.font(11),
            theme.TEXT_DIM,
            (806, 308),
        )
        draw_text(
            surface,
            "Higher values search deeper but take longer. The default config starts at 50.",
            theme.font(11),
            theme.TEXT_FAINT,
            (806, 338),
        )

        draw_text(surface, "PIECE ANIMATION", theme.font(9, bold=True), theme.TEXT_FAINT, (788, 402))
        speed = self.app.settings.get("animation_speed", "normal")
        draw_pill(surface, speed.upper(), theme, (806, 420), theme.GOLD, 10)
        draw_text(
            surface,
            "Normal, fast, or instant piece placement.",
            theme.font(11),
            theme.TEXT_FAINT,
            (920, 422),
        )
        draw_text(surface, "UI PERSISTENCE", theme.font(9, bold=True), theme.TEXT_FAINT, (788, 520))
        draw_text(
            surface,
            "Saved to results/ui_settings.json. Does not change training configs.",
            theme.font(10),
            theme.TEXT_DIM,
            (806, 538),
        )
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
