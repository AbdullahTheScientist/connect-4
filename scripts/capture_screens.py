"""Render real headless screenshots of every screen into docs/screenshots."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from utils.config import load_config
from visualization.controller import App


def capture(app: App, output: Path, name: str) -> None:
    app.screen.blit(app.background, (0, 0))
    if app.current_screen is not None:
        app.current_screen.update(1 / 60)
        app.current_screen.draw(app.screen)
    path = output / name
    pygame.image.save(app.screen, path)
    print(path)


def wait_until(app: App, predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if app.current_screen is not None:
            app.current_screen.update(1 / 60)
        if predicate():
            return True
        time.sleep(0.01)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "screenshots"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/default.yaml")
    app = App(config, headless=True)
    try:
        best = app._preferred_model_path()
        initial = app._initial_or_preferred_path()

        app.show_main_menu()
        capture(app, output, "main_menu.png")

        app.show_gameplay(best)
        screen = app.current_screen
        first = screen.session.human_move(3)
        screen.board.animate(first.row, first.action, first.player)
        wait_until(app, lambda: screen.session.can_human_move() and not screen.board.is_animating)
        second = screen.session.human_move(2)
        if second:
            screen.board.animate(second.row, second.action, second.player)
        wait_until(app, lambda: screen.session.can_human_move() and not screen.board.is_animating)
        capture(app, output, "gameplay.png")

        app.show_ai_vs_ai(best, initial)
        screen = app.current_screen
        wait_until(app, lambda: len(screen.session.move_events) >= 5, timeout=30.0)
        screen._toggle_pause()
        wait_until(
            app,
            lambda: not screen.board.is_animating and not screen.session.thinking,
            timeout=20.0,
        )
        capture(app, output, "ai_vs_ai.png")

        app.show_comparison()
        capture(app, output, "comparison.png")

        app.show_checkpoint_manager()
        capture(app, output, "checkpoints.png")

        app.show_training_monitor()
        capture(app, output, "training_monitor.png")

        app.show_evaluation_dashboard()
        capture(app, output, "evaluation_dashboard.png")

        app.show_settings()
        capture(app, output, "settings.png")

        app.show_model_selector(
            "CHOOSE YOUR OPPONENT",
            "Every option below is a real saved network or a real non-neural baseline.",
            app.checkpoint_options(include_special=True),
            lambda option: None,
            app.show_main_menu,
            include_special=True,
        )
        capture(app, output, "model_selector.png")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
