#!/usr/bin/env python3

"""
Improved dependency-free plotting to PNG.

This module builds on `simple_png_plot.py` but adds:
- tick labels
- thicker lines
- multi-panel line plots

It intentionally stays stdlib-only to keep the repo dependency-free.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple, List

from simple_png_plot import Canvas, draw_text, _nice_limits


def _fmt_tick(v: float) -> str:
    if not math.isfinite(v):
        return "0"
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _line_thick(c: Canvas, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], w: int) -> None:
    if w <= 1:
        c.line(x0, y0, x1, y1, color)
        return
    r = max(1, w // 2)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            c.line(x0 + dx, y0 + dy, x1 + dx, y1 + dy, color)


def _polyline_thick(c: Canvas, pts: Sequence[Tuple[int, int]], color: Tuple[int, int, int], w: int) -> None:
    if len(pts) < 2:
        return
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        _line_thick(c, x0, y0, x1, y1, color, w)


def _draw_panel(
    c: Canvas,
    x: Sequence[float],
    series: Dict[str, Sequence[float]],
    title: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    show_x_ticks: bool,
    y_label: str = "",
) -> None:
    fg = (15, 15, 15)
    grid = (225, 225, 225)

    left = x0 + 90
    right = x1 - 20
    top = y0 + 40
    bottom = y1 - 55

    xmin, xmax = min(x), max(x)
    yvals = [v for s in series.values() for v in s]
    ymin, ymax = _nice_limits(min(yvals), max(yvals))

    def tx(xv: float) -> int:
        if xmax == xmin:
            return left
        return left + int((xv - xmin) / (xmax - xmin) * (right - left))

    def ty(yv: float) -> int:
        if ymax == ymin:
            return bottom
        return top + int((ymax - yv) / (ymax - ymin) * (bottom - top))

    # Grid + ticks (5).
    for i in range(6):
        xx = left + int(i / 5 * (right - left))
        yy = top + int(i / 5 * (bottom - top))
        c.vline(xx, top, bottom, grid)
        c.hline(left, right, yy, grid)

        # y tick label
        ytick = ymax - (ymax - ymin) * (i / 5)
        draw_text(c, x0 + 10, yy - 8, _fmt_tick(ytick), fg, scale=2)

        if show_x_ticks:
            xtick = xmin + (xmax - xmin) * (i / 5)
            draw_text(c, xx - 10, bottom + 10, _fmt_tick(xtick), fg, scale=2)

    # Axes.
    c.vline(left, top, bottom, fg)
    c.hline(left, right, bottom, fg)

    # Title + y label (as simple text).
    draw_text(c, left, y0 + 10, title, fg, scale=2)
    if y_label:
        draw_text(c, x0 + 10, y0 + 10, y_label, fg, scale=2)

    colors = [
        (0, 102, 204),   # blue
        (220, 0, 0),     # red
        (0, 140, 0),     # green
        (180, 0, 180),   # purple
        (255, 140, 0),   # orange
    ]

    for idx, (name, y) in enumerate(series.items()):
        pts = [(tx(xv), ty(yv)) for xv, yv in zip(x, y)]
        col = colors[idx % len(colors)]
        _polyline_thick(c, pts, col, w=3)
        # Legend row at bottom-left of panel.
        lx = left + idx * 240
        ly = bottom + 22
        c.hline(lx, lx + 40, ly + 7, col)
        draw_text(c, lx + 55, ly, name, fg, scale=2)


def plot_lines_png(path: Path, x: Sequence[float], series: Dict[str, Sequence[float]], title: str) -> None:
    width, height = 1200, 700
    c = Canvas.new(width, height)
    _draw_panel(c, x, series, title, 0, 0, width - 1, height - 1, show_x_ticks=True)
    c.save(path)


def plot_panels_lines_png(path: Path, x: Sequence[float], panels: Sequence[Tuple[str, Dict[str, Sequence[float]]]], title: str) -> None:
    n = len(panels)
    if n <= 0:
        raise ValueError("panels empty")

    width, height = 1200, 980
    top = 10
    bottom = 10
    gap = 30
    panel_h = int((height - top - bottom - gap * (n - 1)) / n)

    c = Canvas.new(width, height)
    draw_text(c, 90, 10, title, (15, 15, 15), scale=2)

    for idx, (ylabel, series) in enumerate(panels):
        y0 = top + idx * (panel_h + gap) + 30
        y1 = y0 + panel_h
        _draw_panel(
            c,
            x,
            series,
            title="",
            x0=0,
            y0=y0,
            x1=width - 1,
            y1=y1,
            show_x_ticks=(idx == n - 1),
            y_label=ylabel,
        )

    c.save(path)


def plot_hist_png(path: Path, series: Dict[str, Sequence[float]], title: str, bins: int = 18) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 90, 20, 50, 70
    fg = (15, 15, 15)
    grid = (225, 225, 225)

    c = Canvas.new(width, height)
    draw_text(c, left, 10, title, fg, scale=2)

    allv = [v for s in series.values() for v in s]
    vmin, vmax = min(allv), max(allv)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]

    counts_by: Dict[str, List[int]] = {}
    maxc = 1
    for name, vals in series.items():
        counts = [0] * bins
        for v in vals:
            j = min(bins - 1, max(0, int((v - vmin) / (vmax - vmin) * bins)))
            counts[j] += 1
        counts_by[name] = counts
        maxc = max(maxc, max(counts))

    plot_w = width - left - right
    plot_h = height - top - bottom

    # Grid + tick labels.
    for i in range(6):
        xx = left + int(i / 5 * plot_w)
        yy = top + int(i / 5 * plot_h)
        c.vline(xx, top, top + plot_h, grid)
        c.hline(left, left + plot_w, yy, grid)

        ytick = maxc - (maxc) * (i / 5)
        draw_text(c, 10, yy - 8, _fmt_tick(ytick), fg, scale=2)

    c.vline(left, top, top + plot_h, fg)
    c.hline(left, left + plot_w, top + plot_h, fg)

    colors = [
        (0, 102, 204),
        (220, 0, 0),
        (0, 140, 0),
        (180, 0, 180),
    ]

    m = len(counts_by)
    if m == 0:
        c.save(path)
        return

    bar_w = max(1, int(plot_w / bins))
    group_w = bar_w
    if m > 1:
        group_w = max(1, int(bar_w * 0.9 / m))

    for mi, (name, counts) in enumerate(counts_by.items()):
        col = colors[mi % len(colors)]
        for j, cnt in enumerate(counts):
            x_base = left + j * bar_w + mi * group_w
            h = int((cnt / maxc) * plot_h) if maxc > 0 else 0
            c.rect(x_base, top + plot_h - h, x_base + group_w - 1, top + plot_h, col)
        # Legend
        lx = left + mi * 240
        ly = height - bottom + 22
        c.hline(lx, lx + 40, ly + 7, col)
        draw_text(c, lx + 55, ly, name, fg, scale=2)

    # x tick labels (min/mid/max)
    draw_text(c, left, top + plot_h + 10, _fmt_tick(vmin), fg, scale=2)
    draw_text(c, left + int(plot_w / 2) - 10, top + plot_h + 10, _fmt_tick(0.5 * (vmin + vmax)), fg, scale=2)
    draw_text(c, left + plot_w - 30, top + plot_h + 10, _fmt_tick(vmax), fg, scale=2)

    c.save(path)

