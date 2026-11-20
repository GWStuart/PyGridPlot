"""Small immutable value types describing regions of the screen and the world.

This module is part of the pure maths layer, so it must never import pygame.
A ``pygame.Rect`` can still be converted with :meth:`ScreenRect.from_any`,
which only relies on the rectangle's ``x``, ``y``, ``w`` and ``h``
attributes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

Axis: TypeAlias = Literal["x", "y"]
"""Name of a Cartesian axis."""

ColorLike: TypeAlias = "str | tuple[int, int, int] | tuple[int, int, int, int]"
"""Any colour value accepted by the rendering backend.

The core never interprets colours itself; it only passes them on to the
renderer. With the pygame backend any value accepted by ``pygame.Color``
works (names such as ``"crimson"``, hex strings such as ``"#ff8800"``, and
RGB or RGBA tuples).
"""


@dataclass(frozen=True, slots=True)
class ScreenRect:
    """An axis-aligned rectangle of whole pixels on a surface.

    The edges follow pygame's conventions: ``left``/``top`` are the first
    pixel row/column inside the rectangle, while ``right``/``bottom`` are
    *exclusive* (one past the last pixel inside it).

    Attributes:
        x: Left edge in pixels.
        y: Top edge in pixels.
        width: Width in pixels (non-negative).
        height: Height in pixels (non-negative).
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate that the size is non-negative."""
        if self.width < 0 or self.height < 0:
            msg = (
                f"ScreenRect size must be non-negative, got {self.width}x{self.height}"
            )
            raise ValueError(msg)

    @classmethod
    def from_any(cls, rect: object) -> ScreenRect:
        """Convert a rectangle-like object into a :class:`ScreenRect`.

        Args:
            rect: A :class:`ScreenRect`, any object with ``x``, ``y``, ``w`` and
                ``h`` attributes (such as ``pygame.Rect`` or ``pygame.FRect``),
                or a 4-item sequence ``(x, y, width, height)``.

        Returns:
            The equivalent :class:`ScreenRect`. Fractional values are rounded
            down to whole pixels.

        Raises:
            TypeError: If ``rect`` cannot be interpreted as a rectangle.
        """
        if isinstance(rect, ScreenRect):
            return rect
        if all(hasattr(rect, name) for name in ("x", "y", "w", "h")):
            values = (rect.x, rect.y, rect.w, rect.h)  # type: ignore[attr-defined]
        elif isinstance(rect, Sequence) and len(rect) == 4:
            values = tuple(rect)
        else:
            msg = f"cannot interpret {rect!r} as a rectangle"
            raise TypeError(msg)
        x, y, w, h = (math.floor(v) for v in values)
        return cls(x, y, w, h)

    @property
    def left(self) -> int:
        """First pixel column inside the rectangle."""
        return self.x

    @property
    def top(self) -> int:
        """First pixel row inside the rectangle."""
        return self.y

    @property
    def right(self) -> int:
        """One past the last pixel column inside the rectangle (exclusive)."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """One past the last pixel row inside the rectangle (exclusive)."""
        return self.y + self.height

    @property
    def size(self) -> tuple[int, int]:
        """The ``(width, height)`` of the rectangle."""
        return (self.width, self.height)

    @property
    def is_empty(self) -> bool:
        """Whether the rectangle covers no pixels at all."""
        return self.width == 0 or self.height == 0

    def contains_point(self, px: float, py: float) -> bool:
        """Return whether a screen point lies inside the rectangle.

        Args:
            px: Horizontal screen coordinate in pixels.
            py: Vertical screen coordinate in pixels.

        Returns:
            ``True`` if ``left <= px < right`` and ``top <= py < bottom``.
        """
        return self.left <= px < self.right and self.top <= py < self.bottom


@dataclass(frozen=True, slots=True)
class Bounds:
    """An axis-aligned region of the world (mathematical) plane.

    Attributes:
        x_min: Smallest visible x value.
        x_max: Largest visible x value.
        y_min: Smallest visible y value.
        y_max: Largest visible y value.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        """Extent of the region along x, in world units."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Extent of the region along y, in world units."""
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        """The world point at the middle of the region."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def range(self, axis: Axis) -> tuple[float, float]:
        """Return the ``(min, max)`` range of the region along one axis.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            The extent of the region along ``axis``.
        """
        if axis == "x":
            return (self.x_min, self.x_max)
        return (self.y_min, self.y_max)

    def contains_point(self, x: float, y: float) -> bool:
        """Return whether a world point lies inside the region (edges included).

        Args:
            x: World x coordinate.
            y: World y coordinate.

        Returns:
            ``True`` if the point is inside or on the boundary.
        """
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
