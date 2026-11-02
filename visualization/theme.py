"""Central dark theme and drawing helpers for the Pygame client."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import pygame


class Theme:
    BG_TOP = (8, 12, 21)
    BG_BOTTOM = (17, 25, 40)
    SURFACE = (21, 29, 44)
    SURFACE_ALT = (27, 37, 55)
    SURFACE_LIGHT = (35, 48, 68)
    BORDER = (47, 63, 87)
    BORDER_SOFT = (36, 49, 69)
    TEXT = (236, 242, 250)
    TEXT_DIM = (151, 166, 188)
    TEXT_FAINT = (94, 110, 132)
    ACCENT = (58, 202, 232)
    ACCENT_DARK = (27, 129, 158)
    ACCENT_SOFT = (37, 79, 101)
    GOLD = (246, 190, 92)
    GREEN = (75, 205, 139)
    RED = (240, 86, 88)
    PURPLE = (151, 126, 240)
    PLAYER_ONE = (232, 66, 72)
    PLAYER_TWO = (244, 195, 57)
    BOARD_SHADOW = (3, 8, 17)
    BOARD_BLUE = (24, 86, 139)
    BOARD_BLUE_DARK = (15, 52, 91)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __init__(self) -> None:
        self._fonts: dict[tuple[int, bool, bool], pygame.font.Font] = {}
        self._font_name = self._find_font()

    @staticmethod
    def _find_font() -> str:
        candidates = ["segoeui", "segoeuib", "helveticaneue", "dejavusans", "arial", "freesans"]
        for candidate in candidates:
            if pygame.font.match_font(candidate):
                return candidate
        return pygame.font.get_default_font()

    def font(self, size: int, bold: bool = False, italic: bool = False) -> pygame.font.Font:
        key = (int(size), bool(bold), bool(italic))
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont(self._font_name, int(size), bold=bold, italic=italic)
        return self._fonts[key]


def gradient_surface(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> pygame.Surface:
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    for y in range(height):
        t = y / max(1, height - 1)
        color = (
            int(top[0] * (1 - t) + bottom[0] * t),
            int(top[1] * (1 - t) + bottom[1] * t),
            int(top[2] * (1 - t) + bottom[2] * t),
        )
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface


def draw_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    position: tuple[int, int],
    anchor: str = "topleft",
    alpha: int = 255,
) -> pygame.Rect:
    image = font.render(str(text), True, color)
    if alpha < 255:
        image.set_alpha(alpha)
    rect = image.get_rect()
    setattr(rect, anchor, position)
    surface.blit(image, rect)
    return rect


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    position: tuple[int, int],
    max_width: int,
    line_height: int | None = None,
) -> int:
    line_height = line_height or font.get_linesize()
    x, y = position
    lines = wrap_text(text, font, max_width)
    for index, line in enumerate(lines):
        draw_text(surface, line, font, color, (x, y + index * line_height))
    return len(lines) * line_height


def rounded_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, ...],
    radius: int = 14,
    border: int = 0,
    border_color: tuple[int, ...] | None = None,
) -> None:
    if len(color) == 4 and color[3] < 255:
        temp = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp, color, temp.get_rect(), border_radius=radius)
        if border and border_color:
            rgba = border_color if len(border_color) == 4 else (*border_color, color[3])
            pygame.draw.rect(temp, rgba, temp.get_rect(), width=border, border_radius=radius)
        surface.blit(temp, rect.topleft)
        return
    pygame.draw.rect(surface, color[:3], rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surface, border_color[:3], rect, width=border, border_radius=radius)


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    title: str | None = None,
    subtitle: str | None = None,
    accent: tuple[int, int, int] | None = None,
    padding: int = 20,
) -> pygame.Rect:
    rounded_rect(surface, rect, (*theme.SURFACE, 245), radius=18, border=1, border_color=theme.BORDER_SOFT)
    inner = rect.inflate(-padding * 2, -padding * 2)
    y = rect.top + padding
    if accent:
        pygame.draw.rect(surface, accent, pygame.Rect(rect.left, rect.top + 18, 4, 28), border_radius=2)
    if title:
        draw_text(surface, title, theme.font(19, bold=True), theme.TEXT, (rect.left + padding, y))
        y += 29
    if subtitle:
        draw_text(surface, subtitle, theme.font(13), theme.TEXT_DIM, (rect.left + padding, y))
        y += 23
    return pygame.Rect(rect.left + padding, y, rect.width - padding * 2, rect.bottom - padding - y)


def draw_stat_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    label: str,
    value: str,
    accent: tuple[int, int, int] | None = None,
    detail: str | None = None,
) -> None:
    rounded_rect(surface, rect, theme.SURFACE_ALT, radius=14, border=1, border_color=theme.BORDER_SOFT)
    accent = accent or theme.ACCENT
    pygame.draw.rect(surface, accent, pygame.Rect(rect.left + 14, rect.top + 14, 6, 6), border_radius=3)
    draw_text(surface, label.upper(), theme.font(11, bold=True), theme.TEXT_DIM, (rect.left + 28, rect.top + 9))
    draw_text(surface, value, theme.font(24, bold=True), theme.TEXT, (rect.left + 16, rect.top + 29))
    if detail:
        draw_text(surface, detail, theme.font(12), theme.TEXT_FAINT, (rect.left + 16, rect.bottom - 22))


def draw_pill(
    surface: pygame.Surface,
    text: str,
    theme: Theme,
    position: tuple[int, int],
    color: tuple[int, int, int] = Theme.ACCENT,
    font_size: int = 12,
) -> pygame.Rect:
    font = theme.font(font_size, bold=True)
    width = font.size(text)[0] + 22
    rect = pygame.Rect(position[0], position[1], width, font_size + 13)
    rounded_rect(surface, rect, (*color, 24), radius=rect.height // 2, border=1, border_color=(*color, 80))
    draw_text(surface, text, font, color, rect.center, anchor="center")
    return rect


def draw_token(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    player: int,
    theme: Theme,
    alpha: int = 255,
    highlight: bool = False,
) -> None:
    color = theme.PLAYER_ONE if player == 1 else theme.PLAYER_TWO
    dark = tuple(max(0, c - 45) for c in color)
    light = tuple(min(255, c + 40) for c in color)
    token = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
    center_local = (radius + 2, radius + 2)
    pygame.draw.circle(token, (*dark, alpha), center_local, radius)
    pygame.draw.circle(token, (*color, alpha), (center_local[0] - 2, center_local[1] - 2), max(1, radius - 3))
    pygame.draw.circle(
        token,
        (*light, alpha),
        (center_local[0] - radius // 3, center_local[1] - radius // 3),
        max(2, radius // 4),
    )
    if highlight:
        pygame.draw.circle(token, (255, 255, 255, min(220, alpha)), center_local, radius, width=3)
    surface.blit(token, (center[0] - radius - 2, center[1] - radius - 2))


def draw_spinner(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    angle: float,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    start = angle
    end = angle + math.pi * 1.35
    points = []
    steps = 24
    for index in range(steps + 1):
        t = start + (end - start) * index / steps
        points.append((center[0] + math.cos(t) * radius, center[1] + math.sin(t) * radius))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, width)


def draw_orb(surface: pygame.Surface, position: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: int = 25) -> None:
    orb = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for r in range(radius, 0, -max(1, radius // 12)):
        a = int(alpha * (r / radius) ** 2)
        pygame.draw.circle(orb, (*color, a), (radius, radius), r)
    surface.blit(orb, (position[0] - radius, position[1] - radius))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {int(seconds % 60):02d}s"


def format_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "--"
    if decimals <= 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.{decimals}f}%"
