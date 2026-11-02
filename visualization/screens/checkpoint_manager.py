from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pygame

from visualization.components.buttons import Button
from visualization.components.panels import ScrollState
from visualization.controller import Screen
from visualization.theme import (
    draw_panel,
    draw_pill,
    draw_text,
    draw_wrapped_text,
    format_number,
    rounded_rect,
)


class CheckpointManagerScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.records: list[dict] = app.checkpoint_records()
        self.scroll = ScrollState(len(self.records), visible=7)
        self.selected: Optional[int] = 0 if self.records else None
        self.mark_a: Optional[str] = None
        self.mark_b: Optional[str] = None
        if self.records:
            self.mark_a = self.records[0]["path"]
        self.rows_visible = 7
        self.buttons = [
            Button((60, 724, 132, 44), "Main Menu", self._menu, self.theme, style="ghost"),
            Button((204, 724, 142, 44), "Refresh", self._refresh, self.theme, style="secondary"),
            Button((1022, 164, 170, 30), "Use in Game", self._play, self.theme, style="primary", font_size=11),
            Button((1022, 202, 78, 28), "Mark A", self._mark_a, self.theme, style="secondary", font_size=10),
            Button((1108, 202, 84, 28), "Mark B", self._mark_b, self.theme, style="secondary", font_size=10),
            Button((1022, 638, 172, 36), "Compare A vs B", self._compare, self.theme, style="secondary", font_size=12),
            Button((860, 638, 136, 36), "Open Folder", self._open_folder, self.theme, style="ghost", font_size=11),
        ]
        self._update_enabled()

    def _update_enabled(self) -> None:
        has_selection = self.selected is not None and 0 <= (self.selected or 0) < len(self.records)
        has_marks = bool(self.mark_a and self.mark_b and self.mark_a != self.mark_b)
        self.buttons[2].enabled = has_selection
        self.buttons[3].enabled = has_selection
        self.buttons[4].enabled = has_selection
        self.buttons[5].enabled = has_marks
        self.buttons[6].enabled = has_selection

    def _refresh(self) -> None:
        current = self.records[self.selected]["path"] if self.selected is not None and self.records else None
        self.records = self.app.refresh_checkpoints()
        self.scroll = ScrollState(len(self.records), visible=self.rows_visible)
        self.selected = None
        for index, record in enumerate(self.records):
            if record["path"] == current:
                self.selected = index
                break
        if self.selected is None and self.records:
            self.selected = 0
        if self.mark_a and not any(r["path"] == self.mark_a for r in self.records):
            self.mark_a = None
        if self.mark_b and not any(r["path"] == self.mark_b for r in self.records):
            self.mark_b = None
        self._update_enabled()
        self.app.toast("Checkpoint list refreshed")

    def _menu(self) -> None:
        self.app.show_main_menu()

    def _selected_record(self) -> Optional[dict]:
        if self.selected is None or not (0 <= self.selected < len(self.records)):
            return None
        return self.records[self.selected]

    def _play(self) -> None:
        record = self._selected_record()
        if record:
            self.app.show_gameplay(record["path"], simulations=self.app.default_simulations)

    def _mark_a(self) -> None:
        record = self._selected_record()
        if record:
            self.mark_a = record["path"]
            self._update_enabled()
            self.app.toast(f"Model A: {record['name']}")

    def _mark_b(self) -> None:
        record = self._selected_record()
        if record:
            self.mark_b = record["path"]
            self._update_enabled()
            self.app.toast(f"Model B: {record['name']}")

    def _compare(self) -> None:
        if self.mark_a and self.mark_b:
            self.app.show_ai_vs_ai(self.mark_a, self.mark_b, simulations=self.app.default_simulations)

    def _open_folder(self) -> None:
        record = self._selected_record()
        if not record:
            return
        path = Path(record["path"]).resolve().parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            self.app.toast(f"Could not open folder: {exc}")

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
            return
        self.scroll.wheel(event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
            self.scroll.scroll(-1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
            self.scroll.scroll(1)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if pygame.Rect(60, 150, 760, 540).collidepoint(event.pos):
                index = self.scroll.offset + (event.pos[1] - 196) // 64
                if 0 <= index < len(self.records) and 196 <= event.pos[1] < 196 + self.rows_visible * 64:
                    self.selected = index
                    self._update_enabled()
        for button in self.buttons:
            if button.handle_event(event, pygame.mouse.get_pos()):
                return

    def _draw_list(self, surface: pygame.Surface) -> None:
        theme = self.theme
        rounded_rect(surface, pygame.Rect(60, 150, 760, 540), (*theme.SURFACE, 245), radius=16, border=1, border_color=theme.BORDER_SOFT)
        draw_text(surface, "CHECKPOINT", theme.font(9, bold=True), theme.TEXT_FAINT, (82, 166))
        draw_text(surface, "EXPERIMENT", theme.font(9, bold=True), theme.TEXT_FAINT, (432, 166))
        draw_text(surface, "ITER", theme.font(9, bold=True), theme.TEXT_FAINT, (622, 166))
        draw_text(surface, "A / B", theme.font(9, bold=True), theme.TEXT_FAINT, (710, 166))
        pygame.draw.line(surface, theme.BORDER_SOFT, (74, 184), (806, 184), 1)
        visible = self.records[self.scroll.offset : self.scroll.offset + self.rows_visible]
        for slot, record in enumerate(visible):
            index = self.scroll.offset + slot
            rect = pygame.Rect(72, 196 + slot * 64, 730, 54)
            selected = index == self.selected
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            fill = theme.SURFACE_LIGHT if selected else (theme.SURFACE_ALT if hovered else (17, 24, 37))
            rounded_rect(surface, rect, fill, radius=10, border=1, border_color=theme.ACCENT if selected else theme.BORDER_SOFT)
            accent = theme.GREEN if record.get("model_type") == "best" else theme.GOLD if record.get("model_type") == "initial" else theme.ACCENT
            pygame.draw.rect(surface, accent, pygame.Rect(rect.left + 10, rect.top + 13, 3, 28), border_radius=2)
            draw_text(surface, record["name"], theme.font(14, bold=True), theme.TEXT, (rect.left + 22, rect.top + 8))
            draw_text(surface, record.get("filename", ""), theme.font(9), theme.TEXT_FAINT, (rect.left + 22, rect.top + 31))
            draw_text(surface, record.get("experiment", "default"), theme.font(11), theme.TEXT_DIM, (rect.left + 360, rect.top + 18))
            draw_text(surface, f"{record.get('iteration', 0):04d}", theme.font(12, bold=True), theme.TEXT, (rect.left + 550, rect.top + 18))
            if record["path"] == self.mark_a:
                draw_pill(surface, "A", theme, (rect.left + 638, rect.top + 15), theme.PLAYER_ONE, 9)
            if record["path"] == self.mark_b:
                draw_pill(surface, "B", theme, (rect.left + 680, rect.top + 15), theme.PLAYER_TWO, 9)
        if len(self.records) > self.rows_visible:
            track = pygame.Rect(810, 196, 4, self.rows_visible * 64 - 10)
            rounded_rect(surface, track, theme.SURFACE_ALT, radius=2)
            thumb_h = max(24, int(track.height * self.rows_visible / len(self.records)))
            max_off = max(1, len(self.records) - self.rows_visible)
            thumb_y = track.top + int((track.height - thumb_h) * self.scroll.offset / max_off)
            rounded_rect(surface, pygame.Rect(track.left, thumb_y, 4, thumb_h), theme.ACCENT_DARK, radius=2)

    def _draw_details(self, surface: pygame.Surface) -> None:
        theme = self.theme
        panel = pygame.Rect(840, 150, 376, 536)
        draw_panel(surface, panel, theme, "MODEL DETAIL", "Read from the checkpoint file")
        record = self._selected_record()
        latest = next((r for r in self.records if r.get("model_type") == "latest"), None)
        best = next((r for r in self.records if r.get("model_type") == "best"), None)
        draw_text(surface, "LATEST", theme.font(9, bold=True), theme.TEXT_FAINT, (860, 232))
        draw_text(surface, latest["name"] if latest else "--", theme.font(13, bold=True), theme.TEXT, (860, 246))
        draw_text(surface, "BEST", theme.font(9, bold=True), theme.TEXT_FAINT, (1060, 232))
        draw_text(surface, best["name"] if best else "--", theme.font(13, bold=True), theme.GREEN, (1060, 246))
        pygame.draw.line(surface, theme.BORDER_SOFT, (860, 274), (1196, 274), 1)
        if record is None:
            draw_text(surface, "Select a checkpoint from the list.", theme.font(12), theme.TEXT_FAINT, (860, 292))
            return
        y = 288
        fields = [
            ("Name", record["name"]),
            ("Experiment", record.get("experiment", "default")),
            ("Type", str(record.get("model_type", "checkpoint")).upper()),
            ("Iteration", f"{record.get('iteration', 0):04d}"),
            ("Parameters", format_number(record.get("parameter_count"))),
            ("Size", f"{record.get('size_bytes', 0) / (1024 * 1024):.2f} MB"),
            ("Channels", str(record.get("model_config", {}).get("channels", "--"))),
            ("Residual blocks", str(record.get("model_config", {}).get("residual_blocks", "--"))),
        ]
        for label, value in fields:
            draw_text(surface, label.upper(), theme.font(9, bold=True), theme.TEXT_FAINT, (860, y))
            draw_text(surface, str(value), theme.font(11, bold=True), theme.TEXT, (860, y + 12))
            y += 31
        created = record.get("created_at_iso") or (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.get("created_at", 0)))
            if record.get("created_at")
            else "--"
        )
        draw_text(surface, "CREATED", theme.font(9, bold=True), theme.TEXT_FAINT, (860, y + 2))
        draw_text(surface, created, theme.font(10), theme.TEXT_DIM, (860, y + 14))
        y += 31
        path_text = record["path"]
        if len(path_text) > 44:
            path_text = "..." + path_text[-41:]
        draw_text(surface, "PATH", theme.font(9, bold=True), theme.TEXT_FAINT, (860, y + 2))
        draw_text(surface, path_text, theme.font(8), theme.TEXT_FAINT, (860, y + 15))
        comparison = ""
        if self.mark_a and self.mark_b:
            comparison = f"A: {Path(self.mark_a).parent.name}/{Path(self.mark_a).stem}   B: {Path(self.mark_b).parent.name}/{Path(self.mark_b).stem}"
        draw_text(surface, "MARKS", theme.font(9, bold=True), theme.TEXT_FAINT, (860, 608))
        draw_text(surface, comparison[:48] if comparison else "Select and mark two models.", theme.font(8), theme.TEXT_DIM, (860, 621))

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "STORAGE", theme.font(10, bold=True), theme.GREEN, (66, 38))
        draw_text(surface, "Checkpoint Manager", theme.font(31, bold=True), theme.TEXT, (64, 55))
        draw_text(
            surface,
            "Inspect real self-play artifacts. The manager never overwrites best.pt during browsing.",
            theme.font(13),
            theme.TEXT_DIM,
            (66, 96),
        )
        self._draw_list(surface)
        self._draw_details(surface)
        for button in self.buttons:
            button.draw(surface, theme, pygame.mouse.get_pos())
        if len(self.records) > self.rows_visible:
            draw_text(
                surface,
                f"{len(self.records)} checkpoints  |  scroll for more",
                theme.font(10),
                theme.TEXT_FAINT,
                (60, 704),
            )
