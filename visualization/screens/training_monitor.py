from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pygame

from utils.config import PROJECT_ROOT, resolve_path, training_dir
from utils.logging import TrainingControl, read_json, read_jsonl
from visualization.components.buttons import Button
from visualization.components.charts import LineChart, points_from
from visualization.controller import Screen
from visualization.theme import (
    draw_panel,
    draw_pill,
    draw_stat_card,
    draw_text,
    format_number,
    format_percent,
    rounded_rect,
)


class TrainingMonitorScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.logs_root = resolve_path(app.config, "logs")
        self.active_experiment: Optional[str] = None
        self.experiments: list[str] = []
        self.metrics: list[dict] = []
        self.latest: dict = {}
        self.status: dict = {}
        self.best_iteration: Optional[int] = None
        self.last_refresh = 0.0
        self._finished_notified = False
        self.buttons = [
            Button((60, 736, 130, 42), "Main Menu", self._menu, self.theme, style="ghost", font_size=13),
            Button((218, 736, 142, 42), "Start New", self._start_new, self.theme, style="primary", font_size=13),
            Button((372, 736, 104, 42), "Pause", self._pause, self.theme, style="secondary", font_size=13),
            Button((488, 736, 108, 42), "Resume", self._resume, self.theme, style="secondary", font_size=13),
            Button((608, 736, 100, 42), "Stop", self._stop, self.theme, style="danger", font_size=13),
            Button((720, 736, 172, 42), "Resume Checkpoint", self._resume_checkpoint, self.theme, style="secondary", font_size=12),
            Button((904, 736, 112, 42), "Refresh", self._refresh, self.theme, style="ghost", font_size=12),
            Button((1070, 66, 34, 30), "<", self._prev_experiment, self.theme, style="ghost", font_size=14),
            Button((1178, 66, 34, 30), ">", self._next_experiment, self.theme, style="ghost", font_size=14),
        ]
        self._refresh(initial=True)
        if app.training_experiment and app.training_experiment in self.experiments:
            self.active_experiment = app.training_experiment
            self._load_active()

    @property
    def running(self) -> bool:
        process = self.app.training_process
        if process is not None and process.poll() is None:
            return True
        status_name = self.status.get("status", "")
        return status_name in ("running", "paused", "starting")

    @property
    def paused(self) -> bool:
        return self.status.get("status") == "paused"

    def _latest_checkpoint(self) -> Optional[Path]:
        if not self.active_experiment:
            return None
        path = training_dir(self.app.config, self.active_experiment) / "latest.pt"
        return path if path.exists() else None

    def _refresh(self, initial: bool = False) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        candidates = []
        for directory in self.logs_root.iterdir():
            if not directory.is_dir():
                continue
            metrics_path = directory / "metrics.jsonl"
            status_path = directory / "status.json"
            if metrics_path.exists() or status_path.exists():
                mtime = max(
                    metrics_path.stat().st_mtime if metrics_path.exists() else 0,
                    status_path.stat().st_mtime if status_path.exists() else 0,
                )
                candidates.append((mtime, directory.name))
        candidates.sort(reverse=True)
        self.experiments = [name for _, name in candidates]
        if not initial and self.active_experiment and self.active_experiment in self.experiments:
            pass
        elif self.experiments:
            self.active_experiment = self.experiments[0]
        else:
            self.active_experiment = None
        self._load_active()
        self.last_refresh = time.perf_counter()
        if not initial:
            self.app.toast("Training metrics refreshed")

    def _load_active(self) -> None:
        self.metrics = []
        self.latest = {}
        self.status = {}
        self.best_iteration = None
        if not self.active_experiment:
            return
        log_dir = self.logs_root / self.active_experiment
        self.metrics = read_jsonl(log_dir / "metrics.jsonl")
        self.latest = self.metrics[-1] if self.metrics else {}
        self.status = read_json(log_dir / "status.json", {}) or {}
        best_path = training_dir(self.app.config, self.active_experiment) / "best.pt"
        if best_path.exists():
            from model.checkpoint import checkpoint_metadata

            metadata = checkpoint_metadata(best_path)
            if metadata:
                self.best_iteration = int(metadata.get("iteration", 0))
        process = self.app.training_process
        if process is not None:
            code = process.poll()
            if code is not None and not self._finished_notified:
                self._finished_notified = True
                self.app.toast(f"Training process exited with code {code}")
                self.status = read_json(log_dir / "status.json", {}) or self.status

    def _menu(self) -> None:
        self.app.show_main_menu()

    def _set_active(self, name: str) -> None:
        self.active_experiment = name
        self._finished_notified = False
        self._load_active()

    def _prev_experiment(self) -> None:
        if not self.experiments:
            return
        index = self.experiments.index(self.active_experiment) if self.active_experiment in self.experiments else 0
        self._set_active(self.experiments[(index - 1) % len(self.experiments)])

    def _next_experiment(self) -> None:
        if not self.experiments:
            return
        index = self.experiments.index(self.active_experiment) if self.active_experiment in self.experiments else 0
        self._set_active(self.experiments[(index + 1) % len(self.experiments)])

    def _launch(self, experiment: str, resume: Optional[Path] = None) -> None:
        process = self.app.training_process
        if process is not None and process.poll() is None:
            self.app.toast("A training process is already running")
            return
        config_path = self.app.config.get("_config_path")
        if not config_path:
            config_path = str(PROJECT_ROOT / "configs" / "default.yaml")
        logs_dir = self.logs_root / experiment
        logs_dir.mkdir(parents=True, exist_ok=True)
        control_path = logs_dir / "control.json"
        if control_path.exists():
            control_path.unlink()
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config",
            str(config_path),
            "--experiment",
            experiment,
            "--control-file",
            str(control_path),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_handle = (logs_dir / "ui_subprocess.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception as exc:
            log_handle.close()
            self.app.toast(f"Could not start training: {exc}")
            return
        finally:
            try:
                log_handle.close()
            except Exception:
                pass
        self.app.training_process = process
        self.app.training_experiment = experiment
        self.app.training_control_path = control_path
        self._finished_notified = False
        self.active_experiment = experiment
        self._load_active()
        self.app.toast(f"Training launched: {experiment}")

    def _start_new(self) -> None:
        experiment = "ui_" + time.strftime("%Y%m%d_%H%M%S")
        self._launch(experiment)

    def _resume_checkpoint(self) -> None:
        if not self.active_experiment:
            self.app.toast("No experiment selected")
            return
        checkpoint = self._latest_checkpoint()
        if checkpoint is None:
            self.app.toast("No latest.pt for this experiment")
            return
        self._launch(self.active_experiment, resume=checkpoint)

    def _control(self) -> TrainingControl:
        if self.app.training_control_path is not None:
            return TrainingControl(self.app.training_control_path)
        if self.active_experiment:
            return TrainingControl(self.logs_root / self.active_experiment / "control.json")
        return TrainingControl()

    def _pause(self) -> None:
        self._control().request(paused=True, stop=False)
        self.app.toast("Pause requested. Trainer will pause safely.")

    def _resume(self) -> None:
        self._control().request(paused=False, stop=False)
        self.app.toast("Resume requested")

    def _stop(self) -> None:
        self._control().request(paused=False, stop=True)
        self.app.toast("Stop requested. Current iteration will finish safely.")

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
            return
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def update(self, dt: float) -> None:
        if time.perf_counter() - self.last_refresh > 1.0:
            process = self.app.training_process
            if process is not None:
                code = process.poll()
                if code is not None and not self._finished_notified:
                    self._finished_notified = True
                    self.app.toast(f"Training process exited with code {code}")
                    self._refresh(initial=True)
                    self._finished_notified = True
            self._refresh(initial=True)
            self._finished_notified = bool(self.app.training_process and self.app.training_process.poll() is not None)
        self._update_enabled()

    def _update_enabled(self) -> None:
        running = self.running
        paused = self.paused
        self.buttons[1].enabled = not running
        self.buttons[2].enabled = running and not paused
        self.buttons[3].enabled = paused
        self.buttons[4].enabled = running
        self.buttons[5].enabled = (not running) and self._latest_checkpoint() is not None
        self.buttons[7].enabled = len(self.experiments) > 1
        self.buttons[8].enabled = len(self.experiments) > 1

    def _draw_header(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "OPERATIONS", theme.font(10, bold=True), theme.ACCENT, (66, 38))
        draw_text(surface, "Training Monitor", theme.font(31, bold=True), theme.TEXT, (64, 55))
        draw_text(
            surface,
            "Real self-play iterations, losses, promotion gates and checkpoints.",
            theme.font(13),
            theme.TEXT_DIM,
            (66, 96),
        )
        experiment = self.active_experiment or "no experiments"
        draw_text(surface, "ACTIVE RUN", theme.font(9, bold=True), theme.TEXT_FAINT, (936, 48))
        draw_text(surface, experiment, theme.font(16, bold=True), theme.TEXT, (936, 64))
        status = self.status.get("status", "idle")
        color = theme.GREEN if status == "running" else theme.GOLD if status == "paused" else theme.RED if status == "failed" else theme.TEXT_DIM
        draw_pill(surface, status.upper(), theme, (936, 88), color, 9)
        pygame.draw.line(surface, theme.BORDER_SOFT, (64, 118), (1216, 118), 1)

    def _draw_metrics(self, surface: pygame.Surface) -> None:
        theme = self.theme
        m = self.latest
        status = self.status.get("status", "idle")
        y1 = 136
        width = 220
        gap = 13
        x_positions = [60, 60 + width + gap, 60 + 2 * (width + gap), 60 + 3 * (width + gap), 60 + 4 * (width + gap)]
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[0], y1, width, 86),
            theme,
            "Iteration",
            f"{int(m.get('iteration', self.status.get('iteration', 0))):04d}" if m else "--",
            theme.ACCENT,
            f"{status}",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[1], y1, width, 86),
            theme,
            "Games / iteration",
            format_number(m.get("games_played")),
            theme.PURPLE,
            f"avg {m.get('avg_game_length', 0):.1f} moves" if m else "--",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[2], y1, width, 86),
            theme,
            "Replay buffer",
            format_number(m.get("buffer_size")),
            theme.GREEN,
            "training examples",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[3], y1, width, 86),
            theme,
            "Learning rate",
            f"{m.get('learning_rate', 0):.2e}" if m else "--",
            theme.GOLD,
            "AdamW",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[4], y1, width, 86),
            theme,
            "Best iteration",
            f"{self.best_iteration:04d}" if self.best_iteration is not None else "--",
            theme.GREEN,
            f"{int(m.get('promotions_total', 0)):02d} promotion(s)" if m else "no best yet",
        )

        y2 = 236
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[0], y2, width, 86),
            theme,
            "Policy loss",
            format_number(m.get("policy_loss"), 4),
            theme.ACCENT,
            "cross entropy",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[1], y2, width, 86),
            theme,
            "Value loss",
            format_number(m.get("value_loss"), 4),
            theme.PURPLE,
            "MSE",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[2], y2, width, 86),
            theme,
            "Total loss",
            format_number(m.get("total_loss"), 4),
            theme.GOLD,
            "policy + value + L2",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[3], y2, width, 86),
            theme,
            "Win vs random",
            format_percent(m.get("win_rate_vs_random")),
            theme.GREEN,
            f"score {format_percent(m.get('score_vs_random'))}",
        )
        draw_stat_card(
            surface,
            pygame.Rect(x_positions[4], y2, width, 86),
            theme,
            "Candidate vs best",
            format_percent(m.get("candidate_score_vs_best")),
            theme.RED if m and not m.get("promoted") else theme.GREEN,
            "PROMOTED" if m.get("promoted") else "not promoted",
        )

    def _draw_charts(self, surface: pygame.Surface) -> None:
        theme = self.theme
        loss_chart = LineChart(
            pygame.Rect(60, 340, 676, 324),
            theme,
            "LOSS PROGRESSION",
            [
                ("policy", theme.ACCENT, points_from(self.metrics, "policy_loss")),
                ("value", theme.PURPLE, points_from(self.metrics, "value_loss")),
                ("total", theme.GOLD, points_from(self.metrics, "total_loss")),
            ],
            zero_floor=True,
        )
        loss_chart.draw(surface)
        win_chart = LineChart(
            pygame.Rect(750, 340, 466, 324),
            theme,
            "ARENA SCORES",
            [
                ("vs random", theme.GREEN, points_from(self.metrics, "win_rate_vs_random")),
                ("candidate vs best", theme.ACCENT, points_from(self.metrics, "candidate_score_vs_best")),
            ],
            y_range=(0.0, 1.0),
            y_format="{:.0%}",
        )
        win_chart.draw(surface)

    def _draw_footer_info(self, surface: pygame.Surface) -> None:
        theme = self.theme
        best = None
        if self.active_experiment:
            path = training_dir(self.app.config, self.active_experiment) / "best.pt"
            best = path if path.exists() else None
        draw_text(surface, "CURRENT BEST", theme.font(9, bold=True), theme.TEXT_FAINT, (66, 678))
        draw_text(surface, str(best) if best else "not created yet", theme.font(10), theme.TEXT_DIM, (66, 691))
        draw_text(surface, "PROMOTIONS", theme.font(9, bold=True), theme.TEXT_FAINT, (750, 678))
        promoted = [m for m in self.metrics if m.get("promoted")]
        x = 750
        for record in promoted[-6:]:
            pill = draw_pill(surface, f"iter {record.get('iteration')}", theme, (x, 690), theme.GREEN, 8)
            x = pill.right + 7
        if not promoted:
            draw_text(surface, "No candidate has passed the gate yet.", theme.font(9), theme.TEXT_FAINT, (750, 691))
        draw_text(
            surface,
            f"Subprocess log: {self.active_experiment or '--'} / ui_subprocess.log",
            theme.font(9),
            theme.TEXT_FAINT,
            (1216, 678),
            anchor="topright",
        )

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        self._draw_header(surface)
        self._draw_metrics(surface)
        self._draw_charts(surface)
        self._draw_footer_info(surface)
        if not self.metrics:
            rounded_rect(surface, pygame.Rect(80, 400, 500, 100), (*theme.SURFACE, 235), radius=14, border=1, border_color=theme.BORDER_SOFT)
            draw_text(surface, "No iteration metrics yet.", theme.font(15, bold=True), theme.TEXT, (100, 420))
            draw_text(
                surface,
                "Start a run or select an experiment containing metrics.jsonl.",
                theme.font(11),
                theme.TEXT_DIM,
                (100, 448),
            )
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
