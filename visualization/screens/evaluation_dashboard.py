from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pygame

from evaluation.reporting import generate_before_after_chart, generate_experiment_report
from utils.config import log_dir, resolve_path, results_dir, training_dir
from utils.logging import read_json, read_jsonl
from visualization.components.buttons import Button
from visualization.components.charts import BarChart, LineChart, points_from
from visualization.controller import Screen
from visualization.theme import (
    draw_panel,
    draw_pill,
    draw_text,
    format_percent,
    rounded_rect,
)


class EvaluationDashboardScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.logs_root = resolve_path(app.config, "logs")
        self.experiments: list[str] = []
        self.active_experiment: Optional[str] = None
        self.metrics: list[dict] = []
        self.history: list[dict] = []
        self.promotions: list[dict] = []
        self.comparisons: list[dict] = []
        self.buttons = [
            Button((60, 732, 130, 42), "Main Menu", self._menu, self.theme, style="ghost", font_size=13),
            Button((204, 732, 110, 42), "Refresh", self._refresh, self.theme, style="secondary", font_size=12),
            Button((328, 732, 156, 42), "Export PNGs", self._export, self.theme, style="primary", font_size=12),
            Button((1080, 732, 60, 42), "<", self._prev, self.theme, style="ghost", font_size=15),
            Button((1154, 732, 60, 42), ">", self._next, self.theme, style="ghost", font_size=15),
        ]
        self._refresh(initial=True)

    def _refresh(self, initial: bool = False) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        names = []
        for directory in self.logs_root.iterdir():
            if not directory.is_dir():
                continue
            if (directory / "metrics.jsonl").exists() or (directory / "status.json").exists():
                names.append(directory.name)
        names.sort(
            key=lambda name: (self.logs_root / name / "metrics.jsonl").stat().st_mtime
            if (self.logs_root / name / "metrics.jsonl").exists()
            else 0,
            reverse=True,
        )
        self.experiments = names
        if not self.active_experiment or self.active_experiment not in names:
            self.active_experiment = names[0] if names else None
        self._load()
        if not initial:
            self.app.toast("Evaluation data refreshed")

    def _load(self) -> None:
        self.metrics = []
        self.history = []
        self.promotions = []
        if self.active_experiment:
            self.metrics = read_jsonl(log_dir(self.app.config, self.active_experiment) / "metrics.jsonl")
            self.history = read_json(results_dir(self.app.config, self.active_experiment) / "evaluations.json", []) or []
            self.promotions = read_json(training_dir(self.app.config, self.active_experiment) / "promotions.json", []) or []
        self.comparisons = read_json(resolve_path(self.app.config, "results") / "ui_comparisons.json", []) or []

    def _set(self, name: str) -> None:
        self.active_experiment = name
        self._load()

    def _prev(self) -> None:
        if not self.experiments:
            return
        index = self.experiments.index(self.active_experiment) if self.active_experiment in self.experiments else 0
        self._set(self.experiments[(index - 1) % len(self.experiments)])

    def _next(self) -> None:
        if not self.experiments:
            return
        index = self.experiments.index(self.active_experiment) if self.active_experiment in self.experiments else 0
        self._set(self.experiments[(index + 1) % len(self.experiments)])

    def _menu(self) -> None:
        self.app.show_main_menu()

    def _export(self) -> None:
        if not self.active_experiment or not self.metrics:
            self.app.toast("No metrics to export")
            return
        try:
            paths = generate_experiment_report(self.app.config, self.active_experiment)
            before = generate_before_after_chart(self.app.config)
            count = len(paths) + (1 if before else 0)
            self.app.toast(f"Exported {count} real charts to results/{self.active_experiment}/plots")
        except Exception as exc:
            self.app.toast(f"Export failed: {exc}")

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
            return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def _latest_candidate(self) -> Optional[dict]:
        for record in reversed(self.history):
            if record.get("type") == "candidate_vs_best":
                return record
        return None

    def _latest_random(self) -> Optional[dict]:
        for record in reversed(self.history):
            if record.get("type") == "best_vs_random":
                return record
        return None

    def _latest_tactical(self) -> Optional[dict]:
        for record in reversed(self.history):
            if record.get("type") == "best_vs_tactical":
                return record
        return None

    def _draw_header(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "ANALYTICS", theme.font(10, bold=True), theme.GOLD, (66, 38))
        draw_text(surface, "Evaluation Dashboard", theme.font(31, bold=True), theme.TEXT, (64, 55))
        draw_text(
            surface,
            "Every curve below is read from metrics.jsonl and the evaluation history.",
            theme.font(13),
            theme.TEXT_DIM,
            (66, 96),
        )
        draw_text(surface, "RUN", theme.font(9, bold=True), theme.TEXT_FAINT, (936, 52))
        draw_text(surface, self.active_experiment or "no data", theme.font(16, bold=True), theme.TEXT, (936, 68))
        if self.experiments:
            draw_text(
                surface,
                f"{self.experiments.index(self.active_experiment) + 1 if self.active_experiment in self.experiments else 1} / {len(self.experiments)}",
                theme.font(10),
                theme.TEXT_FAINT,
                (1154, 74),
            )
        pygame.draw.line(surface, theme.BORDER_SOFT, (64, 120), (1216, 120), 1)

    def _draw_charts(self, surface: pygame.Surface) -> None:
        theme = self.theme
        loss = LineChart(
            pygame.Rect(60, 138, 676, 286),
            theme,
            "LOSS PROGRESSION",
            [
                ("policy", theme.ACCENT, points_from(self.metrics, "policy_loss")),
                ("value", theme.PURPLE, points_from(self.metrics, "value_loss")),
                ("total", theme.GOLD, points_from(self.metrics, "total_loss")),
            ],
            zero_floor=True,
        )
        loss.draw(surface)
        candidate_points = []
        for record in self.history:
            if record.get("type") == "candidate_vs_best" and record.get("iteration") is not None:
                value = record.get("win_rate")
                if value is not None:
                    candidate_points.append((float(record["iteration"]), float(value)))
        random_points = []
        tactical_points = []
        for record in self.history:
            if record.get("iteration") is None or record.get("win_rate") is None:
                continue
            if record.get("type") == "best_vs_random":
                random_points.append((float(record["iteration"]), float(record["win_rate"])))
            elif record.get("type") == "best_vs_tactical":
                tactical_points.append((float(record["iteration"]), float(record["win_rate"])))
        win = LineChart(
            pygame.Rect(750, 138, 466, 286),
            theme,
            "WIN-RATE PROGRESSION",
            [
                ("candidate vs best", theme.ACCENT, candidate_points),
                ("vs random", theme.GREEN, random_points),
                ("vs tactical", theme.PURPLE, tactical_points),
            ],
            y_range=(0.0, 1.0),
            y_format="{:.0%}",
        )
        win.draw(surface)

        latest = self._latest_candidate()
        if latest:
            bar = BarChart(
                pygame.Rect(60, 440, 470, 258),
                theme,
                f"LATEST CANDIDATE GATE - ITER {latest.get('iteration')}",
                ["Wins", "Losses", "Draws"],
                [latest.get("wins", 0), latest.get("losses", 0), latest.get("draws", 0)],
                [theme.GREEN, theme.RED, theme.GOLD],
                max_value=max(1, int(latest.get("games", 0) or 1)),
                value_format="{:.0f}",
            )
            bar.draw(surface)
        else:
            rounded_rect(surface, pygame.Rect(60, 440, 470, 258), theme.SURFACE, radius=15, border=1, border_color=theme.BORDER_SOFT)
            draw_text(surface, "CANDIDATE GATE", theme.font(14, bold=True), theme.TEXT, (80, 456))
            draw_text(surface, "No candidate evaluation yet.", theme.font(12), theme.TEXT_FAINT, (80, 492))

        panel = pygame.Rect(546, 440, 670, 258)
        draw_panel(surface, panel, theme, "PROMOTION HISTORY", "Only arena scores open the best-model gate")
        x = 568
        y = 504
        if self.promotions:
            for record in self.promotions[-10:]:
                pill = draw_pill(
                    surface,
                    f"iter {record.get('iteration')}  {record.get('score', 0):.0%}",
                    theme,
                    (x, y),
                    theme.GREEN,
                    10,
                )
                x = pill.right + 8
                if x > 1140:
                    x = 568
                    y += 30
            draw_text(
                surface,
                f"{len(self.promotions)} promotion(s) recorded.",
                theme.font(11),
                theme.TEXT_DIM,
                (568, 562),
            )
        else:
            draw_text(
                surface,
                "No promotions yet. A candidate must reach the configured arena threshold.",
                theme.font(12),
                theme.TEXT_FAINT,
                (568, 510),
            )
        random_result = self._latest_random()
        tactical_result = self._latest_tactical()
        y = 588
        if random_result:
            draw_text(
                surface,
                f"Latest vs random: {format_percent(random_result.get('win_rate'))} wins  |  "
                f"score {format_percent(random_result.get('score'))}",
                theme.font(11),
                theme.GREEN,
                (568, y),
            )
            y += 22
        if tactical_result:
            draw_text(
                surface,
                f"Latest vs tactical: {format_percent(tactical_result.get('win_rate'))} wins  |  "
                f"score {format_percent(tactical_result.get('score'))}",
                theme.font(11),
                theme.PURPLE,
                (568, y),
            )
            y += 22
        if self.comparisons:
            latest_comp = self.comparisons[-1]
            draw_text(
                surface,
                f"Before/after: initial {latest_comp.get('wins', 0)}W / trained {latest_comp.get('losses', 0)}W / "
                f"{latest_comp.get('draws', 0)}D over {latest_comp.get('games', 0)} games",
                theme.font(10),
                theme.TEXT_DIM,
                (568, 652),
            )

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        self._draw_header(surface)
        self._draw_charts(surface)
        if not self.metrics:
            rounded_rect(surface, pygame.Rect(80, 190, 500, 96), (*theme.SURFACE, 235), radius=14, border=1, border_color=theme.BORDER_SOFT)
            draw_text(surface, "No metrics for this experiment.", theme.font(15, bold=True), theme.TEXT, (100, 210))
            draw_text(surface, "Run training first, then refresh this dashboard.", theme.font(11), theme.TEXT_DIM, (100, 238))
        draw_text(
            surface,
            "Charts are non-fabricated: each point maps to a recorded iteration or arena game.",
            theme.font(10),
            theme.TEXT_FAINT,
            (60, 712),
        )
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
