"""Drawing point markers from cached, sub-pixel-positioned stamps.

Every marker of a :class:`~pygridplot.core.primitives.Markers` primitive
looks the same, so each one is rasterised once into a small translucent
"stamp" and then blitted wherever it is needed, many at a time with
``Surface.fblits``. Thousands of points are then cheap to draw.

Positions are kept to a quarter of a pixel: a stamp is cached for each of
the 4 × 4 possible sub-pixel phases, so markers glide smoothly while zooming
instead of jumping from pixel to pixel.

Markers may also each have their own size and colour (a scatter plot sized
or coloured by a value). Sizes are rounded to a quarter of a pixel too, so
the stamps stay few. Colours that vary from marker to marker, often
continuously, would make a stamp per marker; instead a colourless stamp is
cached and tinted for each marker with pygame's multiplying blend (about a
microsecond each). With an outline, which keeps its own colour, the stamp
is split into the share of each pixel that is fill (tinted) and a fixed
image of the outline's colour (added), which gives exactly the colours of
a stamp painted in that fill colour.
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

    def __init__(self, colors: ColorResolver, cache_size: int = 2048) -> None:
        self._colors = colors
        self._cache_size = max(16, cache_size)
        self._stamps: OrderedDict[_StampKey, tuple[pygame.Surface, int]] = OrderedDict()
        self._shares: OrderedDict[
            _StampKey, tuple[pygame.Surface, pygame.Surface | None, int]
        ] = OrderedDict()

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
        edge = (
            None
            if markers.edge_color is None
            else self._colors.rgba(markers.edge_color)
        )
        if markers.colors is not None or markers.radii is not None:
            self._draw_each(surface, markers, whole, phase, edge)
            return
        fill = self._colors.rgba(markers.color)
        phase_ids = phase[:, 0] * PHASES + phase[:, 1]
        for pid in np.unique(phase_ids).tolist():
            px, py = divmod(int(pid), PHASES)
            stamp, half = self._stamp(markers, markers.radius, fill, edge, (px, py))
            chosen = whole[phase_ids == pid] - half
            surface.fblits([(stamp, (int(x), int(y))) for x, y in chosen.tolist()])

    def _draw_each(
        self,
        surface: pygame.Surface,
        markers: Markers,
        whole: np.ndarray,
        phase: np.ndarray,
        edge: RGBA | None,
    ) -> None:
        """Draw markers that each have their own size or colour, in order.

        Args:
            surface: The surface to draw on.
            markers: The markers.
            whole: The whole pixel of each centre.
            phase: The sub-pixel phase of each centre.
            edge: The resolved outline colour, or ``None``.
        """
        count = whole.shape[0]
        if markers.radii is None:
            radii = [markers.radius] * count
        else:
            radii = (np.floor(markers.radii * PHASES + 0.5) / PHASES).tolist()
        phases = phase.tolist()
        corners = whole.tolist()
        blits: list[tuple[pygame.Surface, tuple[int, int]]] = []
        if markers.colors is None:
            fill = self._colors.rgba(markers.color)
            for radius, (px, py), (x, y) in zip(radii, phases, corners, strict=True):
                if radius <= 0:
                    continue
                stamp, half = self._stamp(markers, radius, fill, edge, (px, py))
                blits.append((stamp, (x - half, y - half)))
        else:
            fills: list[RGBA]
            if isinstance(markers.colors, np.ndarray):  # RGBA bytes already
                fills = [(r, g, b, a) for r, g, b, a in markers.colors.tolist()]
            else:
                fills = [self._colors.rgba(color) for color in markers.colors]
            for fill, radius, (px, py), (x, y) in zip(
                fills, radii, phases, corners, strict=True
            ):
                if radius <= 0:
                    continue
                if edge is not None and fill[3] < 255:
                    # A translucent fill changes how fill and outline share
                    # the pixels between them: paint (and cache) its stamp.
                    stamp, half = self._stamp(markers, radius, fill, edge, (px, py))
                    blits.append((stamp, (x - half, y - half)))
                    continue
                share, outline, half = self._colorless(markers, radius, edge, (px, py))
                stamp = share.copy()
                if outline is None:
                    stamp.fill(fill, special_flags=pygame.BLEND_RGBA_MULT)
                else:
                    stamp.fill(fill, special_flags=pygame.BLEND_RGB_MULT)
                    stamp.blit(outline, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
                blits.append((stamp, (x - half, y - half)))
        if blits:
            surface.fblits(blits)

    def _colorless(
        self,
        markers: Markers,
        radius: float,
        edge: RGBA | None,
        phase: tuple[int, int],
    ) -> tuple[pygame.Surface, pygame.Surface | None, int]:
        """Return a marker's colourless stamp, to be tinted (cached).

        Args:
            markers: Supplies shape, outline width and opacity.
            radius: The marker's size.
            edge: Resolved outline colour, or ``None``.
            phase: Sub-pixel phase in units of ``1 / PHASES`` pixels.

        Returns:
            ``(share, outline, half)``. Without an outline, ``share`` is
            white with the marker's coverage as alpha: multiplying it by a
            colour (RGBA) gives the marker in that colour. With one, ``share``
            holds in each pixel the part of its colour that is fill (in
            white) and the marker's whole coverage as alpha, and ``outline``
            the part that is outline (in the outline's colour), to be added
            after tinting ``share``'s colour. ``half`` is as for
            :meth:`_stamp`.
        """
        edge_width = markers.edge_width if edge is not None else 0.0
        key = (
            "colorless",
            markers.shape,
            radius,
            edge_width,
            edge,
            markers.opacity,
            phase,
        )
        cached = self._shares.get(key)
        if cached is not None:
            self._shares.move_to_end(key)
            return cached
        offset = (phase[0] / PHASES, phase[1] / PHASES)
        body, ring, half = marker_coverage(markers.shape, radius, edge_width, offset)
        result: tuple[pygame.Surface, pygame.Surface | None, int]
        if edge is None:
            white = np.full((*body.shape, 3), 255.0)
            share = _surface(white, body * markers.opacity)
            result = (share, None, half)
        else:
            a_edge = ring * (edge[3] / 255.0)
            total = body + a_edge
            part = np.divide(body, total, out=np.ones_like(total), where=total > 0)
            share = _surface(np.repeat(255.0 * part[..., None], 3, axis=2),
                             total * markers.opacity)  # fmt: skip
            rest = (1.0 - part)[..., None] * np.array(edge[:3], dtype=np.float64)
            result = (share, _surface(rest, np.zeros_like(total)), half)
        self._shares[key] = result
        if len(self._shares) > self._cache_size:
            self._shares.popitem(last=False)
        return result

    def clear(self) -> None:
        """Drop all cached stamps."""
        self._stamps.clear()
        self._shares.clear()

    def _stamp(
        self,
        markers: Markers,
        radius: float,
        fill: RGBA,
        edge: RGBA | None,
        phase: tuple[int, int],
    ) -> tuple[pygame.Surface, int]:
        """Return the stamp for a marker style at one sub-pixel phase (cached).

        Args:
            markers: Supplies shape, outline width and opacity.
            radius: The marker's size.
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
            radius,
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
        body, ring, half = marker_coverage(markers.shape, radius, edge_width, offset)
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
    return _surface(rgb, alpha)


def _surface(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    """Make a surface with per-pixel alpha from colour and alpha arrays.

    Args:
        rgb: Colours in ``[0, 255]``, shape ``(h, w, 3)``.
        alpha: Alpha in ``[0, 1]``, shape ``(h, w)``.

    Returns:
        The surface.
    """
    height, width = alpha.shape
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
