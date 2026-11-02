"""Lightweight Pygame charts driven by real metrics."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import pygame

from visualization.theme import Theme, draw_text, rounded_rect


class LineChart:
    def __init__(
        self,
        rect: pygame.Rect,
        theme: Theme,
        title: str,
        series: Sequence[tuple[str, tuple[int, int, int], Sequence[tuple[float, float]]]],
        y_range: tuple[float, float] | None = None,
        y_format: str = "{:.2f}",
        x_format: str = "{:.0f}",
        empty_message: str = "No data yet",
        zero_floor: bool = False,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        self.title = title
        self.series = series
        self.y_range = y_range
        self.y_format = y_format
        self.x_format = x_format
        self.empty_message = empty_message
        self.zero_floor = zero_floor

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        rounded_rect(surface, self.rect, theme.SURFACE, radius=15, border=1, border_color=theme.BORDER_SOFT)
        draw_text(surface, self.title, theme.font(14, bold=True), theme.TEXT, (self.rect.left + 18, self.rect.top + 14))
        legend_x = self.rect.left + 18
        legend_y = self.rect.top + 38
        for label, color, _points in self.series:
            pygame.draw.circle(surface, color, (legend_x + 4, legend_y + 6), 4)
            draw_text(surface, label, theme.font(10), theme.TEXT_DIM, (legend_x + 12, legend_y))
            legend_x += theme.font(10).size(label)[0] + 32
        plot = pygame.Rect(self.rect.left + 52, self.rect.top + 66, self.rect.width - 72, self.rect.height - 98)
        if plot.width < 20 or plot.height < 20:
            return
        valid_series = [s for s in self.series if len(s[2]) > 0]
        if not valid_series:
            draw_text(surface, self.empty_message, theme.font(12), theme.TEXT_FAINT, plot.center, anchor="center")
            return
        all_points = [point for _, _, points in valid_series for point in points]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        x_min, x_max = min(xs), max(xs)
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        if self.y_range is None:
            y_min, y_max = min(ys), max(ys)
            if y_min == y_max:
                padding = max(0.1, abs(y_min) * 0.1)
                y_min -= padding
                y_max += padding
            else:
                padding = (y_max - y_min) * 0.12
                y_min -= padding
                y_max += padding
        else:
            y_min, y_max = self.y_range
        if self.zero_floor:
            y_min = max(0.0, y_min)
        if y_min == y_max:
            y_max = y_min + 1
        for grid_index in range(5):
            t = grid_index / 4
            y = int(plot.bottom - t * plot.height)
            pygame.draw.line(surface, theme.BORDER_SOFT, (plot.left, y), (plot.right, y), 1)
            value = y_min + t * (y_max - y_min)
            draw_text(
                surface,
                self.y_format.format(value),
                theme.font(9),
                theme.TEXT_FAINT,
                (plot.left - 8, y),
                anchor="midright",
            )
        for label_index in range(5):
            t = label_index / 4
            x = int(plot.left + t * plot.width)
            pygame.draw.line(surface, theme.BORDER_SOFT, (x, plot.top), (x, plot.bottom), 1)
            value = x_min + t * (x_max - x_min)
            draw_text(
                surface,
                self.x_format.format(value),
                theme.font(9),
                theme.TEXT_FAINT,
                (x, plot.bottom + 6),
                anchor="midtop",
            )
        for label, color, points in valid_series:
            if not points:
                continue
            screen_points = []
            for x_value, y_value in points:
                x = plot.left + (x_value - x_min) / (x_max - x_min) * plot.width
                y = plot.bottom - (y_value - y_min) / (y_max - y_min) * plot.height
                screen_points.append((int(x), int(y)))
            if len(screen_points) > 1:
                pygame.draw.lines(surface, color, False, screen_points, 2)
            for point in screen_points:
                pygame.draw.circle(surface, color, point, 3)
                pygame.draw.circle(surface, theme.SURFACE, point, 1)


class Sparkline:
    def __init__(self, rect: pygame.Rect, theme: Theme, color: tuple[int, int, int]) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        self.color = color

    def draw(self, surface: pygame.Surface, values: Sequence[float]) -> None:
        rounded_rect(surface, self.rect, (12, 18, 29), radius=8)
        clean = [v for v in values if v is not None and math.isfinite(v)]
        if len(clean) < 2:
            return
        low, high = min(clean), max(clean)
        if low == high:
            high += 1
        points = []
        for index, value in enumerate(clean):
            x = self.rect.left + int(index / (len(clean) - 1) * self.rect.width)
            y = self.rect.bottom - int((value - low) / (high - low) * self.rect.height)
            points.append((x, y))
        pygame.draw.lines(surface, self.color, False, points, 2)


class BarChart:
    def __init__(
        self,
        rect: pygame.Rect,
        theme: Theme,
        title: str,
        labels: Sequence[str],
        values: Sequence[float],
        colors: Sequence[tuple[int, int, int]],
        max_value: float | None = None,
        value_format: str = "{:.1f}",
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.theme = theme
        self.title = title
        self.labels = labels
        self.values = values
        self.colors = colors
        self.max_value = max_value
        self.value_format = value_format

    def draw(self, surface: pygame.Surface) -> None:
        theme = self.theme
        rounded_rect(surface, self.rect, theme.SURFACE, radius=15, border=1, border_color=theme.BORDER_SOFT)
        draw_text(surface, self.title, theme.font(14, bold=True), theme.TEXT, (self.rect.left + 18, self.rect.top + 14))
        if not self.values:
            draw_text(surface, "No data", theme.font(12), theme.TEXT_FAINT, self.rect.center, anchor="center")
            return
        chart = pygame.Rect(self.rect.left + 24, self.rect.top + 52, self.rect.width - 48, self.rect.height - 82)
        max_value = self.max_value or max(1.0, max(self.values))
        bar_width = chart.width / (len(self.values) * 1.5)
        for index, (label, value, color) in enumerate(zip(self.labels, self.values, self.colors)):
            x = chart.left + int((index + 0.25) * chart.width / len(self.values))
            height = int(max(1, value / max_value * chart.height))
            rect = pygame.Rect(x, chart.bottom - height, int(bar_width), height)
            rounded_rect(surface, rect, color, radius=5)
            draw_text(
                surface,
                self.value_format.format(value),
                theme.font(10, bold=True),
                theme.TEXT,
                (rect.centerx, rect.top - 6),
                anchor="midbottom",
            )
            draw_text(surface, label, theme.font(10), theme.TEXT_DIM, (rect.centerx, chart.bottom + 6), anchor="midtop")


def points_from(records: Iterable[dict], y_key: str, x_key: str = "iteration") -> list[tuple[float, float]]:
    points = []
    for record in records:
        x = record.get(x_key)
        y = record.get(y_key)
        if x is None or y is None:
            continue
        try:
            if math.isfinite(float(y)):
                points.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return points
