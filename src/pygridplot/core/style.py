"""Styles for plotted objects: lines, points and polygons.

Every drawing call accepts keyword arguments (``color=``, ``width=``, ...)
and, optionally, a reusable style dataclass. Keyword arguments override the
style's fields, so a shared style can be tweaked per call::

    thick = LineStyle(width=4)
    plane.plot(screen, f, style=thick, color="crimson")

A colour of ``None`` means "take the next colour from the theme's palette"
(see :attr:`pygridplot.core.theme.GridTheme.palette`).

Sizes are in screen pixels, so lines and points look the same at every zoom
level. Polygons themselves are in world units and scale with the zoom; only
their outline width is in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, TypeAlias, TypeVar

from pygridplot.core.coords import ColorLike

MarkerShape: TypeAlias = Literal["circle", "square", "diamond", "cross", "plus"]
"""The shape drawn for each point."""

FillRule: TypeAlias = Literal["nonzero", "evenodd"]
"""How overlapping parts of a polygon are filled (as in SVG and the HTML canvas).

``"nonzero"`` fills everything enclosed by the outline, and ``"evenodd"``
leaves areas that are enclosed an even number of times empty (for example
the middle of a five-pointed star).
"""

MARKER_SHAPES: tuple[MarkerShape, ...] = (
    "circle",
    "square",
    "diamond",
    "cross",
    "plus",
)
"""All supported marker shapes."""

FILL_RULES: tuple[FillRule, ...] = ("nonzero", "evenodd")
"""All supported fill rules."""

StyleT = TypeVar("StyleT", "LineStyle", "PointStyle", "PolygonStyle")


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
    """

    color: ColorLike | None = None
    width: float = 2.5
    opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=False)
        _check_opacity("opacity", self.opacity)


@dataclass(frozen=True, slots=True)
class PointStyle:
    """How points are drawn. Points keep the same size in pixels at every zoom.

    Attributes:
        color: Fill colour, or ``None`` for the next palette colour.
        radius: Size of the marker in pixels (the radius of a circle; other
            shapes have about the same area as that circle).
        marker: The marker shape.
        edge_color: Colour of an outline drawn just outside the marker, or
            ``None`` for no outline. A background-coloured outline separates
            points from the curves they sit on.
        edge_width: Thickness of the outline in pixels.
        opacity: Opacity in ``[0, 1]``.
    """

    color: ColorLike | None = None
    radius: float = 4.5
    marker: MarkerShape = "circle"
    edge_color: ColorLike | None = None
    edge_width: float = 1.5
    opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("radius", self.radius, allow_zero=False)
        _check_size("edge_width", self.edge_width, allow_zero=True)
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
    """

    color: ColorLike | None = None
    width: float = 2.5
    opacity: float = 1.0
    fill: bool = True
    fill_color: ColorLike | None = None
    fill_opacity: float = 0.25
    fill_rule: FillRule = "nonzero"

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_size("width", self.width, allow_zero=True)
        _check_opacity("opacity", self.opacity)
        _check_opacity("fill_opacity", self.fill_opacity)
        if self.fill_rule not in FILL_RULES:
            msg = f"fill_rule must be one of {FILL_RULES}, got {self.fill_rule!r}"
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
    :meth:`pygridplot.Plane.draw_grid` restarts it automatically.

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
