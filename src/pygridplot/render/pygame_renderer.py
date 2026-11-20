"""The pygame (pygame-ce) rendering backend.

:class:`PygameRenderer` turns the backend-neutral primitives from
:mod:`pygridplot.core.primitives` into pygame drawing calls:

* Axis-aligned lines are drawn as filled 1-pixel-wide rectangles, the
  fastest and crispest way to draw them in pygame (no antialiasing blur).
  When a line's paint is translucent and the background colour is unknown,
  it is alpha-blended using a cached strip surface instead.
* Labels are blitted from cached, pre-rendered surfaces (see
  :mod:`pygridplot.render.text`).
* Strokes (curves, polylines, outlines) and fills are rasterised with
  antialiasing by :mod:`pygridplot.render.raster` and blended by
  :mod:`pygridplot.render.composite`.
* Point markers are blitted from cached stamps (see
  :mod:`pygridplot.render.markers`).
* All drawing is clipped to the plane's rectangle, and any clip area the
  caller had already set is respected and restored.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

import pygame

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.grid.scene import GridScene
from pygridplot.core.pixels import line_starts
from pygridplot.core.primitives import (
    Fill,
    FontSpec,
    LabelText,
    LineBatch,
    Markers,
    Orientation,
    Stroke,
    TextItem,
    TextSize,
)
from pygridplot.render.colors import RGBA, ColorResolver
from pygridplot.render.composite import blend_dense, blend_sparse
from pygridplot.render.fonts import FontLibrary
from pygridplot.render.markers import MarkerPainter
from pygridplot.render.raster import PixelBox, fill_coverage, stroke_coverage
from pygridplot.render.text import TextRenderer

_STRIP_CACHE_SIZE = 64


class PygameRenderer:
    """Draws PyGridPlot primitives onto pygame surfaces.

    One renderer can be shared by several planes. It holds the font and
    label caches, so sharing it saves memory and work.

    Args:
        text_cache_size: Maximum number of rendered labels to keep cached.
    """

    def __init__(self, *, text_cache_size: int = 2048) -> None:
        self._colors = ColorResolver()
        self._fonts = FontLibrary()
        self._text = TextRenderer(self._fonts, self._colors, text_cache_size)
        self._markers = MarkerPainter(self._colors)
        self._strips: OrderedDict[tuple[object, ...], pygame.Surface] = OrderedDict()

    # ------------------------------------------------------------------
    # TextMeasurer protocol
    # ------------------------------------------------------------------
    def measure_text(self, text: LabelText, font: FontSpec) -> TextSize:
        """Return the size ``text`` will have when drawn with ``font``.

        Args:
            text: The label to measure.
            font: The font it will be drawn with.

        Returns:
            The label's digit-box size.
        """
        return self._text.measure(text, font)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_grid(self, surface: pygame.Surface, scene: GridScene) -> None:
        """Draw a complete grid scene: background, lines, then labels.

        Args:
            surface: The surface to draw on.
            scene: The grid built by :func:`pygridplot.core.grid.scene.build_grid`.
        """
        with self.clipped(surface, scene.rect) as visible:
            if visible is None:
                return
            if scene.background is not None:
                surface.fill(self._colors.rgba(scene.background), visible)
            for batch in scene.lines:
                self.draw_line_batch(surface, batch, scene.background)
            for item in scene.labels:
                self.draw_text(surface, item)

    def draw_line_batch(
        self,
        surface: pygame.Surface,
        batch: LineBatch,
        background: ColorLike | None = None,
    ) -> None:
        """Draw a batch of parallel axis-aligned lines with crisp edges.

        Args:
            surface: The surface to draw on.
            batch: The lines to draw.
            background: The colour already under the lines, if known. It lets
                translucent paints be flattened into opaque colours.
        """
        if len(batch) == 0:
            return
        color = self._colors.flatten(batch.layers, background)
        if color[3] == 0:
            return
        starts = line_starts(batch.positions, batch.width)
        span_start = math.floor(batch.span[0])
        length = math.ceil(batch.span[1]) - span_start
        if length <= 0:
            return
        vertical = batch.orientation is Orientation.VERTICAL
        width = batch.width
        if color[3] == 255:
            fill = surface.fill
            if vertical:
                for x in starts.tolist():
                    fill(color, (x, span_start, width, length))
            else:
                for y in starts.tolist():
                    fill(color, (span_start, y, length, width))
            return
        strip = self._strip(color, width, length, vertical=vertical)
        if vertical:
            surface.fblits([(strip, (x, span_start)) for x in starts.tolist()])
        else:
            surface.fblits([(strip, (span_start, y)) for y in starts.tolist()])

    def draw_text(self, surface: pygame.Surface, item: TextItem) -> None:
        """Draw one positioned label.

        Args:
            surface: The surface to draw on.
            item: The label to draw.
        """
        if item.opacity <= 0.0:
            return
        rendered = self._text.render(
            item.text, item.font, item.color, item.halo_color, item.halo_width
        )
        image = rendered.surface
        image.set_alpha(round(255 * min(item.opacity, 1.0)))
        ox, oy = rendered.offset
        surface.blit(image, (item.left - ox, item.top - oy))

    def draw_stroke(self, surface: pygame.Surface, stroke: Stroke) -> None:
        """Draw polylines with an antialiased round pen.

        Args:
            surface: The surface to draw on (its clip area is respected).
            stroke: The polylines and pen.
        """
        if stroke.opacity <= 0.0:
            return
        coverage = stroke_coverage(stroke.paths, stroke.width, _clip_box(surface))
        if coverage is not None:
            blend_sparse(
                surface, coverage, self._colors.rgba(stroke.color), stroke.opacity
            )

    def draw_fill(self, surface: pygame.Surface, fill: Fill) -> None:
        """Fill polygons with antialiased edges.

        Args:
            surface: The surface to draw on (its clip area is respected).
            fill: The rings and paint.
        """
        if fill.opacity <= 0.0:
            return
        coverage = fill_coverage(fill.rings, fill.rule, _clip_box(surface))
        if coverage is not None:
            blend_dense(surface, coverage, self._colors.rgba(fill.color), fill.opacity)

    def draw_markers(self, surface: pygame.Surface, markers: Markers) -> None:
        """Draw point markers.

        Args:
            surface: The surface to draw on (its clip area is respected).
            markers: The markers.
        """
        if markers.opacity > 0.0:
            self._markers.draw(surface, markers)

    @contextmanager
    def clipped(
        self, surface: pygame.Surface, rect: ScreenRect
    ) -> Iterator[pygame.Rect | None]:
        """Temporarily restrict drawing on ``surface`` to ``rect``.

        The new clip area is intersected with any clip area already set on the
        surface, and the previous clip area is restored afterwards.

        Args:
            surface: The surface to clip.
            rect: The plane's screen rectangle.

        Yields:
            The visible part of ``rect`` as a ``pygame.Rect``, or ``None`` if
            nothing of it is visible.
        """
        previous = surface.get_clip()
        visible = previous.clip(pygame.Rect(rect.x, rect.y, rect.width, rect.height))
        if visible.width == 0 or visible.height == 0:
            yield None
            return
        surface.set_clip(visible)
        try:
            yield visible
        finally:
            surface.set_clip(previous)

    def clear_caches(self) -> None:
        """Drop all cached text layouts, labels, marker stamps and line strips."""
        self._text.clear()
        self._markers.clear()
        self._strips.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _strip(
        self, color: RGBA, width: int, length: int, *, vertical: bool
    ) -> pygame.Surface:
        """Return a cached translucent line surface for alpha-blended lines.

        Args:
            color: The (translucent) line colour.
            width: Line thickness in pixels.
            length: Line length in pixels.
            vertical: Orientation of the line.

        Returns:
            A surface filled with ``color`` that blends when blitted.
        """
        key = (color, width, length, vertical)
        strip = self._strips.get(key)
        if strip is None:
            size = (width, length) if vertical else (length, width)
            strip = pygame.Surface(size, pygame.SRCALPHA)
            strip.fill(color)
            self._strips[key] = strip
            if len(self._strips) > _STRIP_CACHE_SIZE:
                self._strips.popitem(last=False)
        else:
            self._strips.move_to_end(key)
        return strip


def _clip_box(surface: pygame.Surface) -> PixelBox:
    """Return the surface's current clip area as a pixel box.

    Args:
        surface: The surface.

    Returns:
        ``(x0, y0, x1, y1)`` with exclusive ends.
    """
    clip = surface.get_clip()
    return (clip.left, clip.top, clip.right, clip.bottom)
