"""Application controller and real-time human/AI game session controller."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pygame

from evaluation.arena import Agent, MCTSAgent
from evaluation.baseline import RandomAgent, TacticalAgent
from game.connect4 import ConnectFour
from mcts.search import SearchResult
from model.checkpoint import create_initial_checkpoint, list_checkpoints, load_model
from model.network import AlphaZeroNetwork
from utils.config import PROJECT_ROOT, resolve_device, resolve_path
from utils.logging import read_json, write_json
from visualization.theme import Theme, gradient_surface, draw_orb


@dataclass
class ModelOption:
    key: str
    label: str
    detail: str
    kind: str = "checkpoint"
    path: Optional[str] = None
    experiment: str = ""
    iteration: int = 0
    model_type: str = "checkpoint"
    metadata: Optional[dict] = None


class Screen:
    def __init__(self, app: "App") -> None:
        self.app = app
        self.theme = app.theme

    def on_enter(self) -> None:
        pass

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        raise NotImplementedError


class ModelLibrary:
    def __init__(self, device: str | os.PathLike | None = None) -> None:
        self.device = resolve_device(None if device is None else str(device))
        self._cache: Dict[str, tuple[float, AlphaZeroNetwork, dict]] = {}

    def load(self, path: str | Path) -> tuple[AlphaZeroNetwork, dict]:
        path = Path(path)
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        if key in self._cache:
            cached_mtime, model, payload = self._cache[key]
            if cached_mtime == mtime:
                return model, payload
        model, payload = load_model(path, device=self.device)
        self._cache[key] = (mtime, model, payload)
        return model, payload

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class MoveEvent:
    player: int
    action: int
    row: int
    search: Optional[SearchResult]
    elapsed: float
    ai: bool


class GameSession:
    """Threaded MCTS turn management. The Pygame event loop stays responsive."""

    def __init__(
        self,
        config: dict,
        agents: Dict[int, Agent],
        human_player: Optional[int] = None,
        base_seed: int = 0,
    ) -> None:
        self.config = config
        self.agents = agents
        self.human_player = human_player
        self.base_seed = int(base_seed)
        self.game = ConnectFour()
        self.moves: list[int] = []
        self.players: list[int] = []
        self.move_events: list[MoveEvent] = []
        self.thinking = False
        self.thinking_player: Optional[int] = None
        self.thinking_started = 0.0
        self.last_search: Optional[SearchResult] = None
        self.last_error: Optional[str] = None
        self.generation = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def reset(self) -> None:
        self.generation += 1
        self.game = ConnectFour()
        self.moves.clear()
        self.players.clear()
        self.move_events.clear()
        self.thinking = False
        self.thinking_player = None
        self.last_search = None
        self.last_error = None
        self._queue = queue.Queue()

    @property
    def current_player(self) -> int:
        return self.game.get_current_player()

    @property
    def is_human_turn(self) -> bool:
        return (
            self.human_player is not None
            and not self.game.is_terminal()
            and self.current_player == self.human_player
        )

    def agent_for(self, player: int) -> Optional[Agent]:
        return self.agents.get(player)

    def can_human_move(self) -> bool:
        return self.is_human_turn and not self.thinking and not self.game.is_terminal()

    def human_move(self, column: int) -> Optional[MoveEvent]:
        if not self.can_human_move() or not self.game.is_legal(column):
            return None
        return self._apply_action(column, ai=False)

    def start_ai_turn(self) -> bool:
        if self.game.is_terminal() or self.thinking:
            return False
        player = self.current_player
        if player == self.human_player:
            return False
        agent = self.agents.get(player)
        if agent is None:
            return False
        self.thinking = True
        self.thinking_player = player
        self.thinking_started = time.perf_counter()
        self.last_error = None
        snapshot = self.game.clone()
        generation = self.generation
        seed = self.base_seed + self.game.move_count * 101 + player

        def worker() -> None:
            start = time.perf_counter()
            try:
                action = int(agent.select_action(snapshot, rng=np.random.default_rng(seed)))
                result = getattr(agent, "last_result", None)
                self._queue.put((generation, player, action, result, time.perf_counter() - start, None))
            except Exception as exc:  # pragma: no cover - surfaced in UI
                self._queue.put((generation, player, None, None, time.perf_counter() - start, str(exc)))

        self._thread = threading.Thread(target=worker, daemon=True, name=f"ai-turn-p{player}")
        self._thread.start()
        return True

    def poll(self) -> Optional[MoveEvent]:
        try:
            generation, player, action, result, elapsed, error = self._queue.get_nowait()
        except queue.Empty:
            return None
        if generation != self.generation:
            return None
        self.thinking = False
        self.thinking_player = None
        if error:
            self.last_error = error
            return None
        if self.game.is_terminal() or player != self.game.get_current_player():
            return None
        if action is None or not self.game.is_legal(int(action)):
            self.last_error = f"Illegal AI action: {action}"
            return None
        self.last_search = result
        return self._apply_action(int(action), ai=True, elapsed=elapsed, search=result)

    def _apply_action(
        self,
        action: int,
        ai: bool,
        elapsed: float = 0.0,
        search: Optional[SearchResult] = None,
    ) -> Optional[MoveEvent]:
        if self.game.is_terminal() or not self.game.is_legal(action):
            return None
        player = self.game.get_current_player()
        _, _, _, info = self.game.step(action)
        event = MoveEvent(
            player=player,
            action=int(action),
            row=int(info["row"]),
            search=search,
            elapsed=elapsed,
            ai=ai,
        )
        self.moves.append(int(action))
        self.players.append(player)
        self.move_events.append(event)
        return event


class App:
    def __init__(
        self,
        config: dict,
        headless: bool = False,
        mode: str = "menu",
        checkpoint: str | Path | None = None,
        checkpoint_b: str | Path | None = None,
        simulations: int | None = None,
    ) -> None:
        self.config = config
        self.headless = headless
        width = int(config.get("ui", {}).get("width", 1280))
        height = int(config.get("ui", {}).get("height", 800))
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Connect Four | AlphaZero Lab")
        flags = 0
        self.screen = pygame.display.set_mode((width, height), flags)
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.running = True
        self.theme = Theme()
        self.device = resolve_device(None)
        self.library = ModelLibrary(self.device)
        self.current_screen: Optional[Screen] = None
        self.fps = int(config.get("ui", {}).get("fps", 60))
        self.background = self._make_background()
        self._toast_message: Optional[str] = None
        self._toast_until = 0.0
        self.training_process = None
        self.training_experiment: Optional[str] = None
        self.training_control_path: Optional[Path] = None
        self._ensure_initial_checkpoint()
        self.settings = self._load_ui_settings()
        self.default_simulations = int(
            simulations
            if simulations is not None
            else self.settings.get("simulations", config.get("ui", {}).get("default_simulations", 50))
        )

        if mode == "human":
            path = Path(checkpoint) if checkpoint else self._preferred_model_path()
            self.show_gameplay(path, simulations=self.default_simulations)
        elif mode == "ai":
            path_a = Path(checkpoint) if checkpoint else self._preferred_model_path()
            path_b = Path(checkpoint_b) if checkpoint_b else self._initial_or_preferred_path()
            self.show_ai_vs_ai(path_a, path_b, simulations=self.default_simulations)
        else:
            self.show_main_menu()

    def _make_background(self) -> pygame.Surface:
        background = gradient_surface((self.width, self.height), self.theme.BG_TOP, self.theme.BG_BOTTOM)
        draw_orb(background, (self.width - 90, 40), 280, self.theme.ACCENT, alpha=15)
        draw_orb(background, (90, self.height - 40), 260, self.theme.PURPLE, alpha=14)
        grid = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for x in range(0, self.width, 48):
            pygame.draw.line(grid, (255, 255, 255, 4), (x, 0), (x, self.height))
        for y in range(0, self.height, 48):
            pygame.draw.line(grid, (255, 255, 255, 4), (0, y), (self.width, y))
        background.blit(grid, (0, 0))
        return background

    def _ui_settings_path(self) -> Path:
        return resolve_path(self.config, "results") / "ui_settings.json"

    def _load_ui_settings(self) -> dict:
        data = read_json(self._ui_settings_path(), {}) or {}
        ui = self.config.get("ui", {})
        return {
            "simulations": int(data.get("simulations", ui.get("default_simulations", 50))),
            "animation_speed": data.get("animation_speed", "normal"),
            "show_analysis": data.get("show_analysis", True),
        }

    def save_ui_settings(self) -> None:
        self.settings["simulations"] = int(self.default_simulations)
        write_json(self._ui_settings_path(), self.settings)

    def _checkpoint_root(self) -> Path:
        return resolve_path(self.config, "checkpoints")

    def _ensure_initial_checkpoint(self) -> None:
        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.rglob("initial.pt")):
            return
        if not any(root.rglob("*.pt")):
            create_initial_checkpoint(
                root / "initial.pt",
                {"model": self.config.get("model", {})},
                seed=int(self.config.get("experiment", {}).get("seed", 42)),
            )

    def refresh_checkpoints(self) -> list[dict]:
        self.library.clear()
        return list_checkpoints(self._checkpoint_root(), recursive=True)

    def checkpoint_records(self) -> list[dict]:
        return list_checkpoints(self._checkpoint_root(), recursive=True)

    def checkpoint_options(self, include_special: bool = False) -> list[ModelOption]:
        records = self.checkpoint_records()
        options: list[ModelOption] = []
        for record in records:
            label = f"{record['name']}"
            if record["name"] in ("best", "latest", "initial"):
                label = record["name"].upper()
            elif record["name"].startswith("iteration_"):
                label = f"ITER {record.get('iteration', 0):04d}"
            experiment = record.get("experiment", "default")
            detail = f"{experiment} | iteration {record.get('iteration', 0)}"
            options.append(
                ModelOption(
                    key=record["path"],
                    label=label,
                    detail=detail,
                    kind="checkpoint",
                    path=record["path"],
                    experiment=experiment,
                    iteration=int(record.get("iteration", 0)),
                    model_type=record.get("model_type", "checkpoint"),
                    metadata=record,
                )
            )
        priority = {"best": 0, "latest": 1, "initial": 2}
        options.sort(key=lambda option: (priority.get(option.model_type, 3), -option.iteration, option.experiment))
        if include_special:
            options.extend(
                [
                    ModelOption(
                        key="random",
                        label="RANDOM AGENT",
                        detail="Uniform legal moves, no neural network",
                        kind="random",
                    ),
                    ModelOption(
                        key="tactical",
                        label="TACTICAL AGENT",
                        detail="Immediate wins/blocks heuristic",
                        kind="tactical",
                    ),
                ]
            )
        return options

    def option_for_path(self, path: str | Path | None) -> Optional[ModelOption]:
        if path is None:
            return None
        resolved = str(Path(path).resolve())
        for option in self.checkpoint_options():
            if option.path and str(Path(option.path).resolve()) == resolved:
                return option
        return None

    def _preferred_model_path(self) -> Path:
        records = self.checkpoint_records()
        for record in records:
            if record.get("model_type") == "best":
                return Path(record["path"])
        for record in records:
            if record.get("model_type") == "latest":
                return Path(record["path"])
        if records:
            return Path(records[0]["path"])
        root = self._checkpoint_root()
        initial = root / "initial.pt"
        if not initial.exists():
            self._ensure_initial_checkpoint()
        return initial

    def _initial_or_preferred_path(self) -> Path:
        for record in self.checkpoint_records():
            if record.get("model_type") == "initial":
                return Path(record["path"])
        return self._preferred_model_path()

    def make_agent(
        self,
        option: ModelOption | str | Path,
        simulations: int,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> Agent:
        if isinstance(option, (str, Path)):
            resolved = self.option_for_path(option)
            option = resolved or ModelOption(
                key=str(option), label=Path(option).stem, detail="", path=str(option)
            )
        if option.kind == "random":
            return RandomAgent(seed=seed)
        if option.kind == "tactical":
            return TacticalAgent(seed=seed)
        if not option.path:
            raise ValueError("Checkpoint option has no path")
        model, _ = self.library.load(option.path)
        return MCTSAgent(
            model,
            self.config,
            device=self.device,
            simulations=simulations,
            name=name or option.label,
            temperature=0.0,
            seed=seed,
        )

    def set_screen(self, screen: Screen) -> None:
        if self.current_screen is not None:
            try:
                self.current_screen.on_exit()
            except Exception:
                pass
        self.current_screen = screen
        screen.on_enter()

    def toast(self, message: str, duration: float = 2.8) -> None:
        self._toast_message = message
        self._toast_until = time.perf_counter() + duration

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if self.current_screen is not None:
                    self.current_screen.handle_event(event)
            if self.current_screen is not None:
                self.current_screen.update(dt)
            self.screen.blit(self.background, (0, 0))
            if self.current_screen is not None:
                self.current_screen.draw(self.screen)
            self._draw_toast()
            pygame.display.flip()
        self.close()

    def _draw_toast(self) -> None:
        if not self._toast_message or time.perf_counter() > self._toast_until:
            return
        from visualization.theme import draw_text, rounded_rect

        font = self.theme.font(13, bold=True)
        width = font.size(self._toast_message)[0] + 40
        rect = pygame.Rect(self.width - width - 28, 22, width, 40)
        rounded_rect(self.screen, rect, (*self.theme.SURFACE_LIGHT, 245), radius=10, border=1, border_color=self.theme.ACCENT_DARK)
        draw_text(self.screen, self._toast_message, font, self.theme.TEXT, rect.center, anchor="center")

    def close(self) -> None:
        if self.current_screen is not None:
            try:
                self.current_screen.on_exit()
            except Exception:
                pass
        try:
            write_json(self._ui_settings_path(), self.settings)
        except Exception:
            pass
        pygame.quit()

    def show_main_menu(self) -> None:
        from visualization.screens.main_menu import MainMenuScreen

        self.set_screen(MainMenuScreen(self))

    def show_gameplay(self, checkpoint: str | Path | None = None, simulations: int | None = None) -> None:
        from visualization.screens.gameplay import GameplayScreen

        self.set_screen(GameplayScreen(self, checkpoint=checkpoint, simulations=simulations or self.default_simulations))

    def show_ai_vs_ai(
        self,
        model_a: str | Path | None = None,
        model_b: str | Path | None = None,
        simulations: int | None = None,
    ) -> None:
        from visualization.screens.ai_vs_ai import AIVsAIScreen

        self.set_screen(
            AIVsAIScreen(
                self,
                model_a=model_a,
                model_b=model_b,
                simulations=simulations or self.default_simulations,
            )
        )

    def show_model_selector(
        self,
        title: str,
        subtitle: str,
        options: list[ModelOption],
        on_select: Callable[[ModelOption], None],
        on_cancel: Callable[[], None],
        include_special: bool = False,
    ) -> None:
        from visualization.screens.model_selector import ModelSelectorScreen

        self.set_screen(
            ModelSelectorScreen(self, title, subtitle, options, on_select, on_cancel, include_special=include_special)
        )

    def show_checkpoint_manager(self) -> None:
        from visualization.screens.checkpoint_manager import CheckpointManagerScreen

        self.set_screen(CheckpointManagerScreen(self))

    def show_training_monitor(self) -> None:
        from visualization.screens.training_monitor import TrainingMonitorScreen

        self.set_screen(TrainingMonitorScreen(self))

    def show_evaluation_dashboard(self) -> None:
        from visualization.screens.evaluation_dashboard import EvaluationDashboardScreen

        self.set_screen(EvaluationDashboardScreen(self))

    def show_comparison(self) -> None:
        from visualization.screens.comparison import ComparisonScreen

        self.set_screen(ComparisonScreen(self))

    def show_settings(self) -> None:
        from visualization.screens.settings import SettingsScreen

        self.set_screen(SettingsScreen(self))
