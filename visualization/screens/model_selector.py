from __future__ import annotations

from typing import Callable

import pygame

from visualization.components.buttons import Button
from visualization.components.panels import ScrollState
from visualization.controller import ModelOption, Screen
from visualization.theme import draw_pill, draw_text, rounded_rect


class ModelSelectorScreen(Screen):
    def __init__(
        self,
        app,
        title: str,
        subtitle: str,
        options: list[ModelOption],
        on_select: Callable[[ModelOption], None],
        on_cancel: Callable[[], None],
        include_special: bool = False,
    ) -> None:
        super().__init__(app)
        self.title = title
        self.subtitle = subtitle
        self.options = list(options)
        if include_special and not any(option.kind in ("random", "tactical") for option in self.options):
            self.options.extend([option for option in app.checkpoint_options(include_special=True) if option.kind in ("random", "tactical")])
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.rows_visible = 5
        self.scroll = ScrollState(total=(len(self.options) + 1) // 2, visible=self.rows_visible)
        self.back_button = Button((64, 716, 150, 44), "Back", self._cancel, self.theme, style="ghost")
        self.hovered_index: int | None = None
        self.card_rects: list[tuple[int, pygame.Rect]] = []

    def _cancel(self) -> None:
        self.on_cancel()

    def _visible_indices(self) -> list[int]:
        start = self.scroll.offset * 2
        count = self.rows_visible * 2
        return list(range(start, min(len(self.options), start + count)))

    def _build_layout(self) -> None:
        self.card_rects = []
        for slot, index in enumerate(self._visible_indices()):
            row = slot // 2
            col = slot % 2
            rect = pygame.Rect(70 + col * 584, 184 + row * 100, 556, 86)
            self.card_rects.append((index, rect))

    def handle_event(self, event: pygame.event.Event) -> None:
        self.scroll.wheel(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._cancel()
            elif event.key == pygame.K_DOWN:
                self.scroll.scroll(1)
            elif event.key == pygame.K_UP:
                self.scroll.scroll(-1)
            return
        if self.back_button.handle_event(event, pygame.mouse.get_pos()):
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
            self.scroll.scroll(-1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
            self.scroll.scroll(1)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._build_layout()
            for index, rect in self.card_rects:
                if rect.collidepoint(event.pos):
                    self.on_select(self.options[index])
                    return

    def update(self, dt: float) -> None:
        self._build_layout()

    def _card(self, surface: pygame.Surface, index: int, rect: pygame.Rect, mouse: tuple[int, int]) -> None:
        option = self.options[index]
        hovered = rect.collidepoint(mouse)
        self.hovered_index = index if hovered else self.hovered_index
        if option.kind == "random":
            accent = self.theme.GOLD
        elif option.kind == "tactical":
            accent = self.theme.PURPLE
        elif option.model_type == "best":
            accent = self.theme.GREEN
        elif option.model_type == "initial":
            accent = self.theme.TEXT_DIM
        else:
            accent = self.theme.ACCENT
        fill = self.theme.SURFACE_LIGHT if hovered else self.theme.SURFACE
        rounded_rect(surface, rect, fill, radius=14, border=1, border_color=accent if hovered else self.theme.BORDER_SOFT)
        pygame.draw.rect(surface, accent, pygame.Rect(rect.left + 15, rect.top + 15, 5, rect.height - 30), border_radius=2)
        draw_text(surface, option.label, self.theme.font(18, bold=True), self.theme.TEXT, (rect.left + 34, rect.top + 14))
        draw_text(surface, option.detail, self.theme.font(11), self.theme.TEXT_DIM, (rect.left + 34, rect.top + 40))
        if option.kind == "checkpoint":
            draw_pill(surface, option.model_type.upper(), self.theme, (rect.left + 34, rect.top + 60), accent, 9)
            if option.experiment:
                draw_text(
                    surface,
                    option.experiment,
                    self.theme.font(10),
                    self.theme.TEXT_FAINT,
                    (rect.right - 16, rect.top + 62),
                    anchor="topright",
                )
        elif option.kind == "random":
            draw_pill(surface, "NON-NEURAL BASELINE", self.theme, (rect.left + 34, rect.top + 60), accent, 9)
        else:
            draw_pill(surface, "NON-NEURAL BASELINE", self.theme, (rect.left + 34, rect.top + 60), accent, 9)
        if hovered:
            draw_text(surface, "SELECT", self.theme.font(11, bold=True), accent, (rect.right - 18, rect.centery), anchor="midright")

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        draw_text(surface, "MODEL LIBRARY", theme.font(11, bold=True), theme.ACCENT, (70, 64))
        draw_text(surface, self.title, theme.font(34, bold=True), theme.TEXT, (68, 84))
        draw_text(surface, self.subtitle, theme.font(13), theme.TEXT_DIM, (70, 128))
        visible = self._visible_indices()
        if not self.options:
            draw_text(surface, "No checkpoints available.", theme.font(17), theme.TEXT_DIM, (70, 220))
        for slot, index in enumerate(visible):
            col = slot % 2
            row = slot // 2
            rect = pygame.Rect(70 + col * 584, 184 + row * 100, 556, 86)
            self._card(surface, index, rect, pygame.mouse.get_pos())
        total_rows = max(1, (len(self.options) + 1) // 2)
        if total_rows > self.rows_visible:
            track = pygame.Rect(1222, 184, 5, self.rows_visible * 100 - 14)
            rounded_rect(surface, track, theme.SURFACE_ALT, radius=3)
            thumb_height = max(24, int(track.height * self.rows_visible / total_rows))
            max_offset = max(1, total_rows - self.rows_visible)
            thumb_y = track.top + int((track.height - thumb_height) * self.scroll.offset / max_offset)
            rounded_rect(surface, pygame.Rect(track.left, thumb_y, 5, thumb_height), theme.ACCENT_DARK, radius=3)
        self.back_button.draw(surface, theme, pygame.mouse.get_pos())
        draw_text(
            surface,
            f"{len(self.options)} model options  |  scroll with mouse wheel",
            theme.font(11),
            theme.TEXT_FAINT,
            (250, 737),
        )
