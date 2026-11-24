"""Drawing a plane's frame again onto a picture of its own, at any scale.

The plane keeps a list of what it drew in the current frame (see
:attr:`Frame`): the grid scene, then each call's primitives and texts.
:class:`Replayer` draws that list again onto a new surface the size of
the plane, or ``scale`` times bigger, with every primitive scaled (see
:mod:`pygridplot.core.scaling`). This is how a frame is saved as a picture
and recorded into a video at any resolution, without the program drawing
it again.

A replayer has its own renderer, so its text and coverage caches never
push out the screen's. It also keeps the scaled copy of each primitive of
the previous frame: the plane draws unchanged plots with the same
primitive objects, so their scaled copies are the same objects too, and
the renderer reuses their coverage while a video is recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, TypeAlias, cast

import pygame

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.grid.scene import GridScene
from pygridplot.core.primitives import (
    Fill,
    Image,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
)
from pygridplot.core.scaling import (
    Scaling,
    scale_fill,
    scale_image,
    scale_line_batch,
    scale_markers,
    scale_stroke,
    scale_text,
)
from pygridplot.render.colors import ColorResolver
from pygridplot.render.pygame_renderer import PygameRenderer

Shape: TypeAlias = Stroke | Fill | Markers | Image
"""A primitive drawn by a plotting call."""

Drawn: TypeAlias = (
    tuple[Literal["scene"], GridScene]
    | tuple[Literal["shapes"], tuple[Shape, ...]]
    | tuple[Literal["texts"], tuple[TextItem, ...]]
)
"""One thing the plane drew: its grid, a call's shapes, or texts."""

Frame: TypeAlias = Sequence[Drawn]
"""Everything a plane drew in one frame, in order."""


class Replayer:
    """Draws frames a plane recorded onto pictures of their own, at a scale."""

    def __init__(self) -> None:
        self.renderer = PygameRenderer()
        """The renderer pictures are drawn with (its caches are its own)."""
        self._colors = ColorResolver()
        self._scaling: Scaling | None = None
        self._previous: dict[int, tuple[object, object]] = {}
        self._current: dict[int, tuple[object, object]] = {}

    def render(
        self,
        frame: Frame,
        rect: ScreenRect,
        measurer: TextMeasurer,
        scale: float,
        background: ColorLike | None = None,
        *,
        transparent: bool = False,
    ) -> pygame.Surface:
        """Draw a recorded frame onto a new surface.

        Args:
            frame: What the plane drew.
            rect: The plane's rectangle on screen (where it drew it).
            measurer: Measures text as it was drawn on screen.
            scale: How many picture pixels a screen pixel becomes, along
                each side.
            background: A colour to fill the picture with first, or
                ``None``.
            transparent: Make the picture a surface with per-pixel alpha,
                transparent where nothing is drawn (for a plane without a
                background). An opaque picture is drawn on a surface like the
                screen, so at scale 1 it is the screen's pixels exactly
                (translucent fills are blended a little differently, to a
                level, on surfaces with alpha).

        Returns:
            The picture.

        Raises:
            ValueError: If the picture would be too large.
        """
        scaling = Scaling(scale, rect.left, rect.top)
        width, height = scaling.size(rect.width, rect.height)
        surface = pygame.Surface((width, height), pygame.SRCALPHA if transparent else 0)
        if background is not None:
            surface.fill(self._colors.rgba(background))
        if scaling != self._scaling:  # copies made for another place or size
            self._scaling, self._previous = scaling, {}
        self._current = {}
        whole = ScreenRect(0, 0, width, height)
        renderer = self.renderer
        for kind, item in frame:
            if isinstance(item, GridScene):
                self._draw_scene(surface, item, whole, scaling, measurer)
            elif kind == "shapes":
                with renderer.clipped(surface, whole) as visible:
                    if visible is not None:
                        for shape in cast(tuple[Shape, ...], item):
                            self._draw_shape(surface, self._scaled(shape, scaling))
            else:
                for text in cast(tuple[TextItem, ...], item):
                    renderer.draw_text(surface, self._text(text, scaling, measurer))
        self._previous = self._current
        return surface

    def _draw_scene(
        self,
        surface: pygame.Surface,
        scene: GridScene,
        whole: ScreenRect,
        scaling: Scaling,
        measurer: TextMeasurer,
    ) -> None:
        """Draw a grid scene as the renderer does, every part scaled."""
        renderer = self.renderer
        with renderer.clipped(surface, whole) as visible:
            if visible is None:
                return
            if scene.background is not None:
                surface.fill(self._colors.rgba(scene.background), visible)
            for batch in scene.lines:
                lines = self._reuse(batch, partial(scale_line_batch, batch, scaling))
                renderer.draw_line_batch(surface, lines, scene.background)
            for head in scene.arrows:
                renderer.draw_fill(surface, self._scaled(head, scaling))
            for label in scene.labels:
                renderer.draw_text(surface, self._text(label, scaling, measurer))

    def _draw_shape(self, surface: pygame.Surface, shape: Shape) -> None:
        """Draw one scaled primitive."""
        if isinstance(shape, Stroke):
            self.renderer.draw_stroke(surface, shape)
        elif isinstance(shape, Fill):
            self.renderer.draw_fill(surface, shape)
        elif isinstance(shape, Image):
            self.renderer.draw_image(surface, shape)
        else:
            self.renderer.draw_markers(surface, shape)

    def _scaled(self, shape: Shape, scaling: Scaling) -> Any:
        """The scaled copy of a primitive (the previous frame's, if the same)."""
        if isinstance(shape, Stroke):
            return self._reuse(shape, partial(scale_stroke, shape, scaling))
        if isinstance(shape, Fill):
            return self._reuse(shape, partial(scale_fill, shape, scaling))
        if isinstance(shape, Image):
            return self._reuse(shape, partial(scale_image, shape, scaling))
        return self._reuse(shape, partial(scale_markers, shape, scaling))

    def _text(
        self, item: TextItem, scaling: Scaling, measurer: TextMeasurer
    ) -> TextItem:
        """The scaled copy of a text item (texts are compared by value)."""
        return scale_text(item, scaling, measurer, self.renderer)

    def _reuse(self, source: object, make: Callable[[], Any]) -> Any:
        """Return the previous frame's scaled copy of ``source``, or make one.

        Copies are matched by the identity of what they copy, which the
        plane keeps for unchanged plots. Only a copy of an object drawn in
        the previous frame (and still alive, held here) is reused, and only
        for the same scaling (the same scale and place on screen).
        """
        key = id(source)
        found = self._previous.get(key)
        copy = found[1] if found is not None and found[0] is source else make()
        self._current[key] = (source, copy)
        return copy
