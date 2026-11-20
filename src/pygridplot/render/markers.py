"""Drawing point markers from cached, sub-pixel-positioned stamps.

Every marker of a :class:`~pygridplot.core.primitives.Markers` primitive
looks the same, so each one is rasterised once into a small translucent
"stamp" and then blitted wherever it is needed, many at a time with
``Surface.fblits``. Thousands of points are then cheap to draw.

Positions are kept to a quarter of a pixel: a stamp is cached for each of
the 4 × 4 possible sub-pixel phases, so markers glide smoothly while zooming
instead of jumping from pixel to pixel.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Final

import numpy as np
import pygame

from pygridplot.core.primitives import Markers
from pygridplot.render.colors import RGBA, ColorResolver
from pygridplot.render.raster import marker_coverage

PHASES: Final = 4
"""Sub-pixel positions per axis that markers are rendered at."""

_StampKey = tuple[object, ...]


class MarkerPainter:
    """Draws markers, caching one stamp per style and sub-pixel phase.

    Args:
        colors: Resolves colour values.
        cache_size: Most stamps kept in memory.
    """

    def __init__(self, colors: ColorResolver, cache_size: int = 512) -> None:
        self._colors = colors
        self._cache_size = max(16, cache_size)
        self._stamps: OrderedDict[_StampKey, tuple[pygame.Surface, int]] = OrderedDict()

    def draw(self, surface: pygame.Surface, markers: Markers) -> None:
        """Draw every marker of ``markers`` onto ``surface``.

        Args:
            surface: The surface to draw on (its clip area is respected).
            markers: The markers to draw.
        """
        positions = markers.positions
        if positions.shape[0] == 0:
            return
        whole, phase = _split_positions(positions)
        fill = self._colors.rgba(markers.color)
        edge = (
            None
            if markers.edge_color is None
            else self._colors.rgba(markers.edge_color)
        )
        phase_ids = phase[:, 0] * PHASES + phase[:, 1]
        for pid in np.unique(phase_ids).tolist():
            px, py = divmod(int(pid), PHASES)
            stamp, half = self._stamp(markers, fill, edge, (px, py))
            chosen = whole[phase_ids == pid] - half
            surface.fblits([(stamp, (int(x), int(y))) for x, y in chosen.tolist()])

    def clear(self) -> None:
        """Drop all cached stamps."""
        self._stamps.clear()

    def _stamp(
        self,
        markers: Markers,
        fill: RGBA,
        edge: RGBA | None,
        phase: tuple[int, int],
    ) -> tuple[pygame.Surface, int]:
        """Return the stamp for a marker style at one sub-pixel phase (cached).

        Args:
            markers: Supplies shape, size, outline width and opacity.
            fill: Resolved fill colour.
            edge: Resolved outline colour, or ``None``.
            phase: Sub-pixel phase in units of ``1 / PHASES`` pixels.

        Returns:
            ``(stamp, half)``, where ``half`` is the offset from the stamp's
            top-left pixel to the pixel holding the marker centre.
        """
        edge_width = markers.edge_width if edge is not None else 0.0
        key = (
            markers.shape,
            markers.radius,
            edge_width,
            fill,
            edge,
            markers.opacity,
            phase,
        )
        cached = self._stamps.get(key)
        if cached is not None:
            self._stamps.move_to_end(key)
            return cached
        offset = (phase[0] / PHASES, phase[1] / PHASES)
        body, ring, half = marker_coverage(
            markers.shape, markers.radius, edge_width, offset
        )
        stamp = _paint_stamp(body, ring, fill, edge, markers.opacity)
        self._stamps[key] = (stamp, half)
        if len(self._stamps) > self._cache_size:
            self._stamps.popitem(last=False)
        return stamp, half


def _split_positions(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split continuous positions into whole pixels and quarter-pixel phases.

    Args:
        positions: Marker centres, shape ``(N, 2)``.

    Returns:
        ``(whole, phase)``: integer pixel of each centre and its phase in
        ``0 .. PHASES - 1`` (both of shape ``(N, 2)``).
    """
    base = np.floor(positions)
    phase = np.floor((positions - base) * PHASES + 0.5).astype(np.intp)
    carry = phase == PHASES  # rounded up to the next whole pixel
    phase[carry] = 0
    return (base + carry).astype(np.intp), phase


def _paint_stamp(
    body: np.ndarray,
    ring: np.ndarray,
    fill: RGBA,
    edge: RGBA | None,
    opacity: float,
) -> pygame.Surface:
    """Colour a marker's coverage into a translucent stamp surface.

    The body and the outline ring do not overlap, so their colours are simply
    added, weighted by coverage.

    Args:
        body: Coverage of the marker body, shape ``(h, w)``.
        ring: Coverage of the outline, shape ``(h, w)``.
        fill: Body colour.
        edge: Outline colour, or ``None``.
        opacity: Overall opacity.

    Returns:
        The stamp, with per-pixel alpha.
    """
    a_fill = body * (fill[3] / 255.0) * opacity
    rgb = np.broadcast_to(np.array(fill[:3], dtype=np.float64), (*body.shape, 3)).copy()
    alpha = a_fill
    if edge is not None:
        a_edge = ring * (edge[3] / 255.0) * opacity
        alpha = a_fill + a_edge
        mix = np.array(edge[:3], dtype=np.float64) * a_edge[..., None]
        mix += np.array(fill[:3], dtype=np.float64) * a_fill[..., None]
        rgb = np.where(
            alpha[..., None] > 0, mix / np.maximum(alpha, 1e-9)[..., None], rgb
        )
    height, width = body.shape
    stamp = pygame.Surface((width, height), pygame.SRCALPHA)
    pixels = pygame.surfarray.pixels3d(stamp)
    channel = pygame.surfarray.pixels_alpha(stamp)
    try:
        pixels[...] = (rgb + 0.5).astype(np.uint8).transpose(1, 0, 2)
        channel[...] = (alpha * 255.0 + 0.5).astype(np.uint8).T
    finally:
        del pixels, channel
    if pygame.display.get_init() and pygame.display.get_surface() is not None:
        stamp = stamp.convert_alpha()
    return stamp
