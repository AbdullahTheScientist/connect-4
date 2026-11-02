from __future__ import annotations

import json
import time
from pathlib import Path

import pygame

from game.connect4 import ConnectFour
from visualization.controller import App, GameSession


class FirstLegalAgent:
    def select_action(self, game: ConnectFour, rng=None) -> int:
        legal = game.get_legal_actions()
        return int(legal[0]) if len(legal) else 0

    last_result = None


def _config(tmp_path: Path) -> dict:
    from utils.config import load_config

    config = load_config("configs/default.yaml")
    config["paths"] = {
        "checkpoints": str(tmp_path / "checkpoints"),
        "logs": str(tmp_path / "logs"),
        "results": str(tmp_path / "results"),
    }
    config["ui"]["width"] = 1280
    config["ui"]["height"] = 800
    return config


def test_game_session_runs_real_ai_turn() -> None:
    config = {"paths": {}}
    session = GameSession(config, {2: FirstLegalAgent()}, human_player=1, base_seed=7)
    assert session.human_move(3) is not None
    assert session.current_player == 2
    assert session.start_ai_turn()
    deadline = time.time() + 3
    event = None
    while time.time() < deadline and event is None:
        event = session.poll()
        time.sleep(0.005)
    assert event is not None
    assert event.ai
    assert session.game.board[5, 0] == 2


def test_all_screens_render_headless(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app = App(config, headless=True)
    try:
        screens = []
        app.show_main_menu()
        screens.append(app.current_screen)
        app.show_gameplay()
        screens.append(app.current_screen)
        app.show_ai_vs_ai()
        screens.append(app.current_screen)
        app.show_comparison()
        screens.append(app.current_screen)
        app.show_checkpoint_manager()
        screens.append(app.current_screen)
        app.show_training_monitor()
        screens.append(app.current_screen)
        app.show_evaluation_dashboard()
        screens.append(app.current_screen)
        app.show_settings()
        screens.append(app.current_screen)
        for screen in screens:
            surface = pygame.Surface((1280, 800))
            screen.update(1 / 60)
            screen.draw(surface)
        assert all(screen is not None for screen in screens)
    finally:
        app.close()


def test_settings_persist_and_adjust(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app = App(config, headless=True)
    try:
        app.show_settings()
        screen = app.current_screen
        start = app.default_simulations
        screen._adjust_sims(10)
        assert app.default_simulations == start + 10
        screen._cycle_animation()
        app.save_ui_settings()
        payload = json.loads((tmp_path / "results" / "ui_settings.json").read_text(encoding="utf-8"))
        assert payload["simulations"] == start + 10
        assert payload["animation_speed"] in {"normal", "fast", "off"}
    finally:
        app.close()
