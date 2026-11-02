from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import pygame

from evaluation.arena import Arena
from evaluation.metrics import EvaluationResult, save_evaluation_result
from visualization.components.buttons import Button
from visualization.components.charts import BarChart
from visualization.controller import ModelOption, Screen
from visualization.theme import (
    draw_panel,
    draw_pill,
    draw_stat_card,
    draw_text,
    draw_token,
    format_percent,
    rounded_rect,
)
from utils.config import resolve_path
from utils.logging import read_json


class ComparisonScreen(Screen):
    def __init__(self, app, simulations: Optional[int] = None, games: int = 20) -> None:
        super().__init__(app)
        self.records = app.checkpoint_records()
        self.trained_option = self._find_trained()
        self.initial_option = self._find_initial()
        self.best_option = self._find_best()
        self.simulations = int(simulations or app.default_simulations)
        self.games = int(games)
        self.result: Optional[EvaluationResult] = self._load_saved_result()
        self.progress = 0
        self.progress_total = self.games
        self.running = False
        self.status = (
            f"Loaded saved result: {self.result.summary()}" if self.result is not None else "Ready"
        )
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._last_elapsed = 0.0
        self.buttons = [
            Button((66, 722, 150, 44), "Main Menu", self._menu, self.theme, style="ghost"),
            Button((470, 722, 172, 44), "Run Batch Eval", self._run, self.theme, style="primary"),
            Button((654, 722, 168, 44), "Watch Live", self._watch_live, self.theme, style="secondary"),
            Button((1010, 722, 46, 44), "-", lambda: self._adjust_sims(-10), self.theme, style="ghost"),
            Button((1164, 722, 46, 44), "+", lambda: self._adjust_sims(10), self.theme, style="ghost"),
        ]
        self._update_enabled()

    def _load_saved_result(self) -> Optional[EvaluationResult]:
        path = resolve_path(self.app.config, "results") / "ui_comparisons.json"
        history = read_json(path, []) or []
        if not isinstance(history, list) or not history:
            return None
        record = history[-1]
        try:
            return EvaluationResult(
                agent_a=str(record.get("agent_a", "Initial")),
                agent_b=str(record.get("agent_b", "Trained")),
                wins=int(record.get("wins", 0)),
                losses=int(record.get("losses", 0)),
                draws=int(record.get("draws", 0)),
                games=int(record.get("games", 0)),
                total_moves=int(record.get("total_moves", 0)),
                wins_as_first=int(record.get("wins_as_first", 0)),
                wins_as_second=int(record.get("wins_as_second", 0)),
                games_as_first=int(record.get("games_as_first", 0)),
                games_as_second=int(record.get("games_as_second", 0)),
                elapsed_seconds=float(record.get("elapsed_seconds", 0.0)),
                simulations=int(record.get("simulations", 0)),
                iteration=record.get("iteration"),
                metadata=record.get("metadata", {}) or {},
            )
        except (TypeError, ValueError):
            return None

    def _find_trained(self) -> Optional[ModelOption]:
        candidates = [r for r in self.records if r.get("model_type") in ("latest", "best", "candidate")]
        candidates = [r for r in candidates if r.get("model_type") != "initial"]
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: (r.get("model_type") == "latest", r.get("iteration", 0), r.get("created_at") or 0),
            reverse=True,
        )
        record = candidates[0]
        return self.app.option_for_path(record["path"])

    def _find_initial(self) -> Optional[ModelOption]:
        if self.trained_option:
            experiment = self.trained_option.experiment
            matching = [
                r for r in self.records
                if r.get("model_type") == "initial" and r.get("experiment") == experiment
            ]
            if matching:
                return self.app.option_for_path(matching[0]["path"])
        initials = [r for r in self.records if r.get("model_type") == "initial"]
        return self.app.option_for_path(initials[0]["path"]) if initials else None

    def _find_best(self) -> Optional[ModelOption]:
        bests = [r for r in self.records if r.get("model_type") == "best"]
        if not bests:
            return None
        bests.sort(key=lambda r: (r.get("iteration", 0), r.get("created_at") or 0), reverse=True)
        return self.app.option_for_path(bests[0]["path"])

    def _update_enabled(self) -> None:
        can_compare = bool(self.initial_option and self.trained_option)
        self.buttons[1].enabled = can_compare and not self.running
        self.buttons[2].enabled = can_compare
        for index in (3, 4):
            self.buttons[index].enabled = can_compare and not self.running

    def _menu(self) -> None:
        self.running = False
        self.app.show_main_menu()

    def _adjust_sims(self, delta: int) -> None:
        self.simulations = max(10, min(400, self.simulations + delta))

    def _watch_live(self) -> None:
        if self.initial_option and self.trained_option:
            self.app.show_ai_vs_ai(
                self.initial_option.path,
                self.trained_option.path,
                simulations=self.simulations,
            )

    def _run(self) -> None:
        if self.running or not self.initial_option or not self.trained_option:
            return
        self.running = True
        self.result = None
        self.progress = 0
        self.progress_total = self.games
        self.status = "Building initial and trained agents..."
        self._start_time = time.perf_counter()
        self._update_enabled()
        initial = self.initial_option
        trained = self.trained_option
        sims = self.simulations
        games = self.games

        def worker() -> None:
            try:
                agent_initial = self.app.make_agent(initial, sims, seed=500, name=f"Initial ({initial.experiment})")
                agent_trained = self.app.make_agent(trained, sims, seed=600, name=f"Trained ({trained.experiment})")
                arena = Arena(self.app.config)

                def progress(done: int, total: int, partial: EvaluationResult) -> None:
                    self._queue.put(("progress", done, total, partial))

                result = arena.evaluate(
                    agent_initial,
                    agent_trained,
                    games=games,
                    seed=123,
                    simulations=sims,
                    iteration=trained.iteration,
                    progress_callback=progress,
                )
                result.metadata.update(
                    {
                        "initial_path": str(initial.path),
                        "trained_path": str(trained.path),
                        "initial_experiment": initial.experiment,
                        "trained_experiment": trained.experiment,
                    }
                )
                output = resolve_path(self.app.config, "results") / "ui_comparisons.json"
                save_evaluation_result(result, output, record_type="initial_vs_trained")
                self._queue.put(("done", result))
            except Exception as exc:  # pragma: no cover - surfaced in UI
                self._queue.put(("error", str(exc)))

        self._thread = threading.Thread(target=worker, daemon=True, name="ui-comparison")
        self._thread.start()

    def on_exit(self) -> None:
        self.running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
            return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def update(self, dt: float) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            if message[0] == "progress":
                _, done, total, partial = message
                self.progress = done
                self.progress_total = total
                self.status = f"Playing game {done}/{total} - {partial.wins}W/{partial.losses}L/{partial.draws}D"
            elif message[0] == "done":
                self.result = message[1]
                self.running = False
                self._last_elapsed = time.perf_counter() - self._start_time
                self.status = self.result.summary()
                self._update_enabled()
            elif message[0] == "error":
                self.status = f"Evaluation failed: {message[1]}"
                self.running = False
                self._update_enabled()

    def _model_box(self, surface: pygame.Surface, option: Optional[ModelOption], rect: pygame.Rect, title: str, player: int) -> None:
        theme = self.theme
        accent = theme.PLAYER_ONE if player == 1 else theme.PLAYER_TWO
        rounded_rect(surface, rect, theme.SURFACE, radius=16, border=1, border_color=theme.BORDER_SOFT)
        draw_token(surface, (rect.left + 35, rect.top + 36), 19, player, theme)
        draw_text(surface, title, theme.font(10, bold=True), accent, (rect.left + 64, rect.top + 18))
        if option:
            draw_text(surface, option.label, theme.font(20, bold=True), theme.TEXT, (rect.left + 64, rect.top + 37))
            draw_text(surface, option.detail, theme.font(11), theme.TEXT_DIM, (rect.left + 22, rect.top + 76))
            draw_pill(surface, f"ITER {option.iteration}", theme, (rect.left + 22, rect.bottom - 36), accent, 9)
            draw_pill(surface, option.model_type.upper(), theme, (rect.left + 116, rect.bottom - 36), theme.TEXT_DIM, 9)
            if option.path:
                name = Path(option.path).name
                draw_text(surface, name, theme.font(10), theme.TEXT_FAINT, (rect.right - 18, rect.bottom - 28), anchor="bottomright")
        else:
            draw_text(surface, "Not available", theme.font(17, bold=True), theme.TEXT_FAINT, (rect.left + 64, rect.top + 37))
            draw_text(surface, "Train a model to unlock this comparison.", theme.font(11), theme.TEXT_FAINT, (rect.left + 22, rect.top + 76))

    def _draw_results(self, surface: pygame.Surface) -> None:
        theme = self.theme
        panel = pygame.Rect(470, 140, 744, 558)
        draw_panel(surface, panel, theme, "BEFORE / AFTER ARENA", "Candidate scores come from real alternating-colour games")
        if self.result is None:
            rounded_rect(surface, pygame.Rect(500, 235, 684, 250), (12, 18, 29), radius=14)
            draw_text(surface, "NO BATCH RESULT YET", theme.font(11, bold=True), theme.TEXT_FAINT, (522, 256))
            body = (
                "Run a batch evaluation to play the randomly initialized network against the trained checkpoint. "
                "Every game uses the real network and MCTS. Colours alternate automatically."
            )
            from visualization.theme import draw_wrapped_text

            draw_wrapped_text(surface, body, theme.font(14), theme.TEXT_DIM, (522, 286), 630, line_height=23)
            draw_text(surface, f"{self.games} games  |  {self.simulations} simulations per move", theme.font(13), theme.ACCENT, (522, 390))
            if self.running:
                draw_text(surface, self.status, theme.font(13, bold=True), theme.GOLD, (522, 436))
                bar = pygame.Rect(522, 462, 640, 7)
                rounded_rect(surface, bar, theme.SURFACE_ALT, radius=3)
                width = int(bar.width * self.progress / max(1, self.progress_total))
                rounded_rect(surface, pygame.Rect(bar.left, bar.top, max(2, width), bar.height), theme.GOLD, radius=3)
            elif self.initial_option is None or self.trained_option is None:
                draw_text(surface, "Trained checkpoint missing. Open Training Monitor first.", theme.font(12), theme.RED, (522, 436))
            return

        result = self.result
        stat_y = 238
        draw_stat_card(
            surface,
            pygame.Rect(498, stat_y, 146, 94),
            theme,
            "Initial wins",
            str(result.wins),
            theme.PLAYER_ONE,
            "random baseline",
        )
        draw_stat_card(
            surface,
            pygame.Rect(658, stat_y, 146, 94),
            theme,
            "Trained wins",
            str(result.losses),
            theme.PLAYER_TWO,
            "from initial perspective",
        )
        draw_stat_card(
            surface,
            pygame.Rect(818, stat_y, 146, 94),
            theme,
            "Draws",
            str(result.draws),
            theme.GOLD,
            f"{result.games} total games",
        )
        draw_stat_card(
            surface,
            pygame.Rect(978, stat_y, 206, 94),
            theme,
            "Trained score",
            f"{(result.losses + 0.5 * result.draws) / max(1, result.games):.1%}",
            theme.GREEN,
            "win=1 draw=.5",
        )
        chart = BarChart(
            pygame.Rect(498, 344, 686, 212),
            theme,
            "BATCH RESULT",
            ["Initial", "Trained", "Draw"],
            [result.wins, result.losses, result.draws],
            [theme.PLAYER_ONE, theme.PLAYER_TWO, theme.GOLD],
            max_value=max(1, result.games),
            value_format="{:.0f}",
        )
        chart.draw(surface)
        draw_text(
            surface,
            f"Average game length: {result.avg_game_length:.1f} moves    |    "
            f"{result.simulations} simulations    |    {result.elapsed_seconds:.1f}s",
            theme.font(12),
            theme.TEXT_DIM,
            (498, 574),
        )
        draw_text(surface, self.status, theme.font(11), theme.TEXT_FAINT, (498, 598))
        draw_text(
            surface,
            "Win percentage shows the initial model's perspective; trained score is its complement.",
            theme.font(10),
            theme.TEXT_FAINT,
            (498, 620),
        )

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "EXPERIMENT", theme.font(10, bold=True), theme.GOLD, (66, 38))
        draw_text(surface, "Initial vs Trained", theme.font(31, bold=True), theme.TEXT, (64, 55))
        draw_text(
            surface,
            "A direct, empirical check that self-play changed the network without pretending results.",
            theme.font(13),
            theme.TEXT_DIM,
            (66, 96),
        )
        self._model_box(surface, self.initial_option, pygame.Rect(66, 140, 368, 164), "INITIAL RANDOM MODEL", 1)
        self._model_box(surface, self.trained_option, pygame.Rect(66, 320, 368, 164), "LATEST TRAINED MODEL", 2)
        best = self.best_option
        rounded_rect(surface, pygame.Rect(66, 500, 368, 120), theme.SURFACE, radius=16, border=1, border_color=theme.BORDER_SOFT)
        draw_text(surface, "BEST CHECKPOINT", theme.font(10, bold=True), theme.GREEN, (88, 520))
        draw_text(surface, best.label if best else "--", theme.font(21, bold=True), theme.TEXT, (88, 540))
        iteration = best.iteration if best else 0
        draw_text(surface, f"iteration {iteration:04d}", theme.font(12), theme.TEXT_DIM, (88, 570))
        draw_text(
            surface,
            f"MCTS simulations: {self.simulations}",
            theme.font(11),
            theme.ACCENT,
            (88, 592),
        )
        draw_text(surface, "SIMULATIONS", theme.font(9, bold=True), theme.TEXT_FAINT, (1010, 714))
        draw_text(surface, str(self.simulations), theme.font(23, bold=True), theme.TEXT, (1110, 724), anchor="midleft")
        self._draw_results(surface)
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
