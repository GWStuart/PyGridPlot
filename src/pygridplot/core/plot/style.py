"""Styles for plotted objects: lines, points, polygons, areas, arrows, bars, text.

Every drawing call accepts keyword arguments (``color=``, ``width=``, ...)
and, optionally, a reusable style dataclass. Keyword arguments override the
style's fields, so a shared style can be tweaked per call::

    thick = LineStyle(width=4)
    grid.plot(screen, f, style=thick, color="crimson")

A colour of ``None`` means "take the next colour from the theme's palette"
(see :attr:`pygridplot.core.grid.theme.GridTheme.palette`).

Sizes are in screen pixels, so lines and points look the same at every zoom
level, and so do dash patterns (see :data:`~pygridplot.core.plot.dashes.Dash`).
Polygons themselves are in world units and scale with the zoom; only their
outline width is in pixels. Shaded areas (between graphs, or where an
inequality holds) have no outline: draw their boundaries with the curve
methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, TypeVar

from pygridplot.core.coords import ColorLike
from pygridplot.core.plot.dashes import Dash, check_dash
from pygridplot.core.primitives import (
    FILL_RULES,
    MARKER_SHAPES,
    FillRule,
    MarkerShape,
)

StyleT = TypeVar(
    "StyleT",
    "LineStyle",
    "PointStyle",
    "PolygonStyle",
    "FillStyle",
    "ArrowStyle",
    "SequenceStyle",
    "BarStyle",
    "TextStyle",
)


def _check_opacity(name: str, value: float) -> None:
    """Raise unless ``value`` is a valid opacity.

    Args:
        name: Field name used in the error message.
        value: The value to check.

    Raises:
        ValueError: If ``value`` is not within ``[0, 1]``.
    """
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be within [0, 1], got {value!r}"
        raise ValueError(msg)


def _check_size(name: str, value: float, *, allow_zero: bool) -> None:
    """Raise unless ``value`` is a valid size in pixels.

    Args:
        name: Field name used in the error message.
        value: The value to check.
        allow_zero: Whether zero is allowed (it switches a feature off).

    Raises:
        ValueError: If ``value`` is negative, not finite, or zero when not allowed.
    """
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        limit = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be {limit} and finite, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LineStyle:
    """How a curve or polyline is drawn.

    Attributes:
        color: Line colour, or ``None`` for the next palette colour.
        width: Line thickness in pixels (fractional widths are fine; lines
            thinner than 1 pixel are drawn 1 pixel wide but fainter).
        opacity: Opacity in ``[0, 1]``.
        dash: ``"solid"``, ``"dashed"``, ``"dotted"``, or the lengths of the
            dashes and gaps in pixels, alternately: ``(12, 6)``; a dash of
            ``0`` is a dot. Named patterns grow with the width. The pattern
            is anchored to the plane, so the dashes stay put while panning.
    """

    color: ColorLike | None = None
    width: float = 2.5
    opacity: float = 1.0
    dash: Dash = "solid"

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=False)
        _check_opacity("opacity", self.opacity)
        object.__setattr__(self, "dash", check_dash(self.dash))


@dataclass(frozen=True, slots=True)
class PointStyle:
    """How points are drawn. Points keep the same size in pixels at every zoom.

    Attributes:
        color: Fill colour, or ``None`` for the next palette colour.
        radius: Size of the marker in pixels (the radius of a circle; other
            shapes have about the same area as that circle). ``0`` draws no
            marker, for error bars alone.
        marker: The marker shape.
        edge_color: Colour of an outline drawn just outside the marker, or
            ``None`` for no outline. A background-coloured outline separates
            points from the curves they sit on.
        edge_width: Thickness of the outline in pixels.
        opacity: Opacity in ``[0, 1]``.
        error_width: Thickness of the error bars in pixels (see
            :meth:`~pygridplot.Grid.points`).
        error_cap: Width of the caps across the ends of the error bars, in
            pixels (``0`` for none).
    """

    color: ColorLike | None = None
    radius: float = 4.5
    marker: MarkerShape = "circle"
    edge_color: ColorLike | None = None
    edge_width: float = 1.5
    opacity: float = 1.0
    error_width: float = 1.5
    error_cap: float = 8.0

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("radius", self.radius, allow_zero=True)
        _check_size("edge_width", self.edge_width, allow_zero=True)
        _check_size("error_width", self.error_width, allow_zero=False)
        _check_size("error_cap", self.error_cap, allow_zero=True)
        _check_opacity("opacity", self.opacity)
        if self.marker not in MARKER_SHAPES:
            msg = f"marker must be one of {MARKER_SHAPES}, got {self.marker!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PolygonStyle:
    """How a polygon is drawn: a translucent fill and an outline.

    Attributes:
        color: Outline colour, also used for the fill unless ``fill_color``
            is given. ``None`` takes the next palette colour.
        width: Outline thickness in pixels (``0`` for no outline).
        opacity: Opacity of the outline in ``[0, 1]``.
        fill: Whether to fill the inside.
        fill_color: Fill colour, or ``None`` to use ``color``.
        fill_opacity: Opacity of the fill in ``[0, 1]``.
        fill_rule: Which parts of a self-overlapping polygon are inside.
        dash: How the outline is drawn: ``"solid"``, ``"dashed"``,
            ``"dotted"``, or dash and gap lengths in pixels (as for
            :attr:`LineStyle.dash`).
    """

    color: ColorLike | None = None
    width: float = 2.5
    opacity: float = 1.0
    fill: bool = True
    fill_color: ColorLike | None = None
    fill_opacity: float = 0.25
    fill_rule: FillRule = "nonzero"
    dash: Dash = "solid"

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=True)
        _check_opacity("opacity", self.opacity)
        _check_opacity("fill_opacity", self.fill_opacity)
        object.__setattr__(self, "dash", check_dash(self.dash))
        if self.fill_rule not in FILL_RULES:
            msg = f"fill_rule must be one of {FILL_RULES}, got {self.fill_rule!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FillStyle:
    """How a shaded area is drawn: a translucent fill, without an outline.

    Used by :meth:`~pygridplot.Grid.fill_between` and
    :meth:`~pygridplot.Grid.inequality`. Their boundaries are curves of
    their own, drawn with :meth:`~pygridplot.Grid.plot` or
    :meth:`~pygridplot.Grid.implicit` when wanted.

    Attributes:
        color: Fill colour. ``None`` takes the next palette colour.
        opacity: Opacity of the fill in ``[0, 1]``.
    """

    color: ColorLike | None = None
    opacity: float = 0.25

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_opacity("opacity", self.opacity)


@dataclass(frozen=True, slots=True)
class ArrowStyle:
    """How arrows are drawn: a straight shaft with a triangular head.

    Used by :meth:`~pygridplot.Grid.arrow` and
    :meth:`~pygridplot.Grid.vector_field`. Sizes are in pixels, so arrows
    look the same at every zoom level; only where they start and end is in
    world units.

    Attributes:
        color: Arrow colour, or ``None`` for the next palette colour.
        width: Thickness of the shaft in pixels.
        opacity: Opacity in ``[0, 1]``.
        head_size: Length of the head in pixels, or ``None`` for a head in
            proportion to the shaft (about ``3 × width + 4``). ``0`` draws no
            head: a plain segment with square ends. An arrow shorter than its
            head is drawn as a smaller head alone.
    """

    color: ColorLike | None = None
    width: float = 2.5
    opacity: float = 1.0
    head_size: float | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=False)
        _check_opacity("opacity", self.opacity)
        if self.head_size is not None:
            _check_size("head_size", self.head_size, allow_zero=True)


@dataclass(frozen=True, slots=True)
class SequenceStyle:
    """How the terms of a sequence are drawn: a stem and a marker each.

    Used by :meth:`~pygridplot.Grid.sequence`. Each term ``aₙ`` gets a
    vertical stem from the x axis up (or down) to ``(n, aₙ)``, and a marker
    there. Sizes are in pixels, so they look the same at every zoom level.

    Attributes:
        color: Colour of the stems and markers, or ``None`` for the next
            palette colour.
        width: Thickness of the stems in pixels (``0`` for markers alone).
        radius: Size of the markers in pixels, as for
            :class:`PointStyle` (``0`` for stems alone).
        marker: The marker shape.
        opacity: Opacity in ``[0, 1]``.
    """

    color: ColorLike | None = None
    width: float = 2.0
    radius: float = 3.5
    marker: MarkerShape = "circle"
    opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=True)
        _check_size("radius", self.radius, allow_zero=True)
        _check_opacity("opacity", self.opacity)
        if self.width == 0 and self.radius == 0:
            msg = "width and radius cannot both be 0: nothing would be drawn"
            raise ValueError(msg)
        if self.marker not in MARKER_SHAPES:
            msg = f"marker must be one of {MARKER_SHAPES}, got {self.marker!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BarStyle:
    """How bars are drawn: a translucent fill with an outline.

    Used by :meth:`~pygridplot.Grid.bars` and
    :meth:`~pygridplot.Grid.histogram`. The bars themselves are in world
    units (they grow and shrink with the zoom); only the outline is in
    pixels.

    Attributes:
        color: Colour of the fill and the outline, or ``None`` for the next
            palette colour.
        opacity: Opacity of the fill in ``[0, 1]``; the outline is opaque.
        edge_width: Thickness of the outline in pixels (``0`` for none).
    """

    color: ColorLike | None = None
    opacity: float = 0.5
    edge_width: float = 1.5

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_opacity("opacity", self.opacity)
        _check_size("edge_width", self.edge_width, allow_zero=True)


@dataclass(frozen=True, slots=True)
class TextStyle:
    """How text is drawn by :meth:`~pygridplot.Grid.text`.

    Text uses the theme's label font. Unlike plots, it never takes palette
    colours, so labelling a plot does not change the colours of the plots
    drawn after it: pass the plot's colour to match it.

    Attributes:
        color: Text colour, or ``None`` for the theme's label colour.
        size: Font size in points (a whole number), or ``None`` for the
            theme's label size.
        opacity: Opacity in ``[0, 1]``.
        halo: Whether to outline the glyphs in the background colour (the
            theme's label halo), so the text stays readable over grid lines
            and curves.
    """

    color: ColorLike | None = None
    size: int | None = None
    opacity: float = 1.0
    halo: bool = True

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_opacity("opacity", self.opacity)
        if self.size is not None and (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 1
        ):
            msg = f"size must be a positive whole number of points, got {self.size!r}"
            raise ValueError(msg)


def with_overrides(style: StyleT, **overrides: object) -> StyleT:
    """Return ``style`` with every override that is not ``None`` applied.

    This is how drawing calls combine a reusable style with per-call keyword
    arguments: ``None`` means "keep the style's value".

    Args:
        style: The base style.
        **overrides: Field values to replace. ``None`` values are ignored.

    Returns:
        The combined style (validated again).

    Raises:
        TypeError: If an override names a field the style does not have.
    """
    names = {f.name for f in fields(style)}
    unknown = set(overrides) - names
    if unknown:
        msg = f"{type(style).__name__} has no field(s) {sorted(unknown)}"
        raise TypeError(msg)
    changes: dict[str, Any] = {k: v for k, v in overrides.items() if v is not None}
    return replace(style, **changes) if changes else style


class ColorCycle:
    """Hands out palette colours in order, for objects drawn without a colour.

    In immediate mode everything is drawn again every frame, so the cycle
    must restart every frame for each object to keep its colour.
    :meth:`pygridplot.Grid.draw` restarts it automatically.

    Args:
        palette: The colours to cycle through (at least one).

    Raises:
        ValueError: If the palette is empty.
    """

    def __init__(self, palette: tuple[ColorLike, ...]) -> None:
        if not palette:
            msg = "the palette needs at least one colour"
            raise ValueError(msg)
        self._palette = palette
        self._index = 0

    @property
    def palette(self) -> tuple[ColorLike, ...]:
        """The colours being cycled through."""
        return self._palette

    @palette.setter
    def palette(self, palette: tuple[ColorLike, ...]) -> None:
        if not palette:
            msg = "the palette needs at least one colour"
            raise ValueError(msg)
        self._palette = palette

    def next(self) -> ColorLike:
        """Return the next colour, wrapping around at the end of the palette.

        Returns:
            A palette colour.
        """
        color = self._palette[self._index % len(self._palette)]
        self._index += 1
        return color

    def resolve(self, color: ColorLike | None) -> ColorLike:
        """Return ``color``, or the next palette colour if it is ``None``.

        Explicit colours do not use up a palette slot.

        Args:
            color: A colour or ``None``.

        Returns:
            The colour to draw with.
        """
        return self.next() if color is None else color

    def reset(self) -> None:
        """Start again from the first palette colour."""
        self._index = 0
