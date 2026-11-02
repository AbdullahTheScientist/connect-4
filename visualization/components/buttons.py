"""Reusable interactive button components."""
from __future__ import annotations

from typing import Callable, Optional

import pygame

from visualization.theme import Theme, draw_text, draw_wrapped_text, rounded_rect


class Button:
    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        label: str,
        callback: Callable[[], None] | None = None,
        theme: Optional[Theme] = None,
        style: str = "secondary",
        font_size: int = 15,
        enabled: bool = True,
        icon: str | None = None,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.theme = theme
        self.style = style
        self.font_size = font_size
        self.enabled = enabled
        self.icon = icon
        self.hovered = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event, mouse_pos: tuple[int, int]) -> bool:
        self.hovered = self.rect.collidepoint(mouse_pos)
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            self.pressed = True
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.hovered:
                self.pressed = False
                if self.callback:
                    self.callback()
                return True
            self.pressed = False
        return False

    def draw(self, surface: pygame.Surface, theme: Theme, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)
        button = pygame.Rect(self.rect)
        if self.hovered:
            button.y -= 1
        if self.style == "primary":
            fill = theme.ACCENT
            text_color = (8, 20, 28)
            border = theme.ACCENT
            hover_fill = (94, 222, 244)
        elif self.style == "danger":
            fill = (64, 34, 43)
            text_color = theme.RED
            border = (104, 52, 62)
            hover_fill = (82, 40, 51)
        elif self.style == "ghost":
            fill = (0, 0, 0, 0)
            text_color = theme.TEXT_DIM
            border = theme.BORDER_SOFT
            hover_fill = theme.SURFACE_ALT
        else:
            fill = theme.SURFACE_ALT
            text_color = theme.TEXT
            border = theme.BORDER
            hover_fill = theme.SURFACE_LIGHT
        if not self.enabled:
            fill = (28, 35, 48)
            text_color = theme.TEXT_FAINT
            border = theme.BORDER_SOFT
        elif self.hovered:
            fill = hover_fill
        if self.pressed and self.enabled:
            button.y += 1
        shadow = button.move(0, 3)
        rounded_rect(surface, shadow, (4, 8, 14, 120), radius=11)
        if self.style == "ghost":
            if self.hovered:
                rounded_rect(surface, button, (*theme.SURFACE_ALT, 160), radius=10)
            pygame.draw.rect(surface, border, button, width=1, border_radius=10)
        else:
            rounded_rect(
                surface,
                button,
                (*fill, 255) if isinstance(fill, tuple) else fill,
                radius=10,
                border=1,
                border_color=border,
            )
        label = f"{self.icon}  {self.label}" if self.icon else self.label
        draw_text(surface, label, theme.font(self.font_size, bold=True), text_color, button.center, anchor="center")


class MenuButton:
    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        title: str,
        subtitle: str,
        callback: Callable[[], None],
        accent: tuple[int, int, int],
        number: str,
        enabled: bool = True,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.title = title
        self.subtitle = subtitle
        self.callback = callback
        self.accent = accent
        self.number = number
        self.enabled = enabled
        self.hovered = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event, mouse_pos: tuple[int, int]) -> bool:
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            self.pressed = True
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.hovered:
                self.pressed = False
                self.callback()
                return True
            self.pressed = False
        return False

    def draw(self, surface: pygame.Surface, theme: Theme, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)
        rect = pygame.Rect(self.rect)
        if not self.enabled:
            base = (23, 30, 43)
            title_color = theme.TEXT_FAINT
            subtitle_color = theme.TEXT_FAINT
            accent = (59, 70, 87)
        else:
            base = theme.SURFACE_ALT if self.hovered else theme.SURFACE
            title_color = theme.TEXT
            subtitle_color = theme.TEXT_DIM
            accent = self.accent
        if self.hovered and self.enabled:
            rect.x += 4
            glow = pygame.Surface((rect.width + 20, rect.height + 16), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*accent, 18), glow.get_rect(), border_radius=18)
            surface.blit(glow, (rect.x - 10, rect.y - 8))
        rounded_rect(surface, rect, (*base, 245), radius=14, border=1, border_color=theme.BORDER_SOFT)
        pygame.draw.rect(surface, accent, pygame.Rect(rect.left, rect.top + 15, 4, rect.height - 30), border_radius=2)
        number_font = theme.font(12, bold=True)
        draw_text(surface, self.number, number_font, accent, (rect.left + 24, rect.top + 16))
        draw_text(surface, self.title, theme.font(18, bold=True), title_color, (rect.left + 55, rect.top + 12))
        draw_text(surface, self.subtitle, theme.font(12), subtitle_color, (rect.left + 55, rect.top + 38))
        if self.enabled:
            draw_text(surface, ">", theme.font(18, bold=True), accent, (rect.right - 27, rect.centery), anchor="center")
        if self.pressed and self.enabled:
            rect.y += 1


class IconButton(Button):
    def __init__(self, rect, label, callback=None, theme=None, style="ghost", font_size=14, enabled=True):
        super().__init__(rect, label, callback, theme, style, font_size, enabled)
        self.rect = pygame.Rect(rect)
