#!/usr/bin/env python3

"""
Minimal plotting to PNG using only the Python standard library.

This is a fallback for environments where matplotlib is unavailable.
It supports basic line charts and histograms with a tiny built-in bitmap font.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def save_png_rgb(path: Path, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != width * height * 3:
        raise ValueError("rgb buffer has wrong size")
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    comp = zlib.compress(raw, level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


@dataclass
class Canvas:
    width: int
    height: int
    rgb: bytearray

    @classmethod
    def new(cls, width: int, height: int, bg: Tuple[int, int, int] = (255, 255, 255)) -> "Canvas":
        rgb = bytearray(width * height * 3)
        c = cls(width, height, rgb)
        c.fill(bg)
        return c

    def fill(self, color: Tuple[int, int, int]) -> None:
        r, g, b = color
        for i in range(0, len(self.rgb), 3):
            self.rgb[i] = r
            self.rgb[i + 1] = g
            self.rgb[i + 2] = b

    def set_px(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        i = (y * self.width + x) * 3
        self.rgb[i] = color[0]
        self.rgb[i + 1] = color[1]
        self.rgb[i + 2] = color[2]

    def hline(self, x0: int, x1: int, y: int, color: Tuple[int, int, int]) -> None:
        if y < 0 or y >= self.height:
            return
        if x0 > x1:
            x0, x1 = x1, x0
        x0 = max(x0, 0)
        x1 = min(x1, self.width - 1)
        for x in range(x0, x1 + 1):
            self.set_px(x, y, color)

    def vline(self, x: int, y0: int, y1: int, color: Tuple[int, int, int]) -> None:
        if x < 0 or x >= self.width:
            return
        if y0 > y1:
            y0, y1 = y1, y0
        y0 = max(y0, 0)
        y1 = min(y1, self.height - 1)
        for y in range(y0, y1 + 1):
            self.set_px(x, y, color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(y0, y1 + 1):
            self.hline(x0, x1, y, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        # Bresenham.
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            self.set_px(x, y, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def polyline(self, pts: Sequence[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
        if len(pts) < 2:
            return
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            self.line(x0, y0, x1, y1, color)

    def save(self, path: Path) -> None:
        save_png_rgb(path, self.width, self.height, bytes(self.rgb))


# 5x7 bitmap font for ASCII uppercase letters, digits, space, dash, underscore, period.
_FONT_5X7: Dict[str, List[str]] = {
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def draw_text(canvas: Canvas, x: int, y: int, text: str, color: Tuple[int, int, int] = (0, 0, 0), scale: int = 2) -> None:
    cx = x
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row, bits in enumerate(glyph):
            for col, b in enumerate(bits):
                if b == "1":
                    canvas.rect(cx + col * scale, y + row * scale, cx + (col + 1) * scale - 1, y + (row + 1) * scale - 1, color)
        cx += (5 * scale) + scale


def _nice_limits(vmin: float, vmax: float) -> Tuple[float, float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        pad = 1.0 if vmin == 0.0 else abs(vmin) * 0.1
        return vmin - pad, vmax + pad
    pad = 0.05 * (vmax - vmin)
    return vmin - pad, vmax + pad


def plot_lines_png(path: Path, x: Sequence[float], series: Dict[str, Sequence[float]], title: str) -> None:
    width, height = 900, 500
    left, right, top, bottom = 70, 20, 40, 60
    c = Canvas.new(width, height)
    fg = (0, 0, 0)
    grid = (220, 220, 220)

    # Range.
    xmin, xmax = min(x), max(x)
    yvals = [v for s in series.values() for v in s]
    ymin, ymax = _nice_limits(min(yvals), max(yvals))

    def tx(xv: float) -> int:
        if xmax == xmin:
            return left
        return left + int((xv - xmin) / (xmax - xmin) * (width - left - right))

    def ty(yv: float) -> int:
        if ymax == ymin:
            return height - bottom
        return top + int((ymax - yv) / (ymax - ymin) * (height - top - bottom))

    # Grid (5 ticks).
    for i in range(6):
        xx = left + int(i / 5 * (width - left - right))
        yy = top + int(i / 5 * (height - top - bottom))
        c.vline(xx, top, height - bottom, grid)
        c.hline(left, width - right, yy, grid)

    # Axes.
    c.vline(left, top, height - bottom, fg)
    c.hline(left, width - right, height - bottom, fg)

    # Title.
    draw_text(c, left, 10, title, fg, scale=2)

    colors = [(220, 0, 0), (0, 70, 200), (0, 140, 0), (180, 0, 180)]
    for idx, (name, y) in enumerate(series.items()):
        pts = [(tx(xv), ty(yv)) for xv, yv in zip(x, y)]
        col = colors[idx % len(colors)]
        c.polyline(pts, col)
        # Legend.
        lx = left + idx * 220
        ly = height - bottom + 15
        c.hline(lx, lx + 30, ly + 7, col)
        draw_text(c, lx + 40, ly, name, fg, scale=2)

    c.save(path)


def plot_hist_png(path: Path, series: Dict[str, Sequence[float]], title: str, bins: int = 15) -> None:
    width, height = 900, 500
    left, right, top, bottom = 70, 20, 40, 60
    c = Canvas.new(width, height)
    fg = (0, 0, 0)
    grid = (220, 220, 220)

    allv = [v for s in series.values() for v in s]
    vmin, vmax = min(allv), max(allv)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]

    counts_by = {}
    maxc = 1
    for name, vals in series.items():
        counts = [0] * bins
        for v in vals:
            j = min(bins - 1, max(0, int((v - vmin) / (vmax - vmin) * bins)))
            counts[j] += 1
        counts_by[name] = counts
        maxc = max(maxc, max(counts))

    # Grid.
    for i in range(6):
        xx = left + int(i / 5 * (width - left - right))
        yy = top + int(i / 5 * (height - top - bottom))
        c.vline(xx, top, height - bottom, grid)
        c.hline(left, width - right, yy, grid)

    # Axes.
    c.vline(left, top, height - bottom, fg)
    c.hline(left, width - right, height - bottom, fg)

    draw_text(c, left, 10, title, fg, scale=2)

    colors = [(220, 0, 0), (0, 70, 200), (0, 140, 0), (180, 0, 180)]
    names = list(counts_by.keys())
    nseries = max(1, len(names))
    bin_w = (width - left - right) / bins
    bar_w = bin_w / (nseries + 1)

    for si, name in enumerate(names):
        col = colors[si % len(colors)]
        counts = counts_by[name]
        for bi, cnt in enumerate(counts):
            x0 = left + int(bi * bin_w + si * bar_w + 2)
            x1 = left + int(bi * bin_w + (si + 1) * bar_w - 2)
            y1 = height - bottom
            y0 = top + int((1.0 - cnt / maxc) * (height - top - bottom))
            c.rect(x0, y0, x1, y1, col)
        lx = left + si * 260
        ly = height - bottom + 15
        c.rect(lx, ly + 4, lx + 18, ly + 12, col)
        draw_text(c, lx + 30, ly, name, fg, scale=2)

    c.save(path)

