"""Colouring values: the public :func:`color_map`.

The mapping itself is pure and lives in
:mod:`pygridplot.core.plot.color_maps`; this module adds what needs pygame:
colour maps given as colour *names* or hex strings (``["navy", "white",
"#b2182b"]``) are resolved with :class:`pygame.Color` first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pygame
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike
from pygridplot.core.plot.color_maps import (
    DEFAULT_COLOR_MAP,
    ColorStops,
    check_range,
    check_stops,
    map_values,
)

ColorMapLike = str | Sequence[ColorLike]
"""A colour map: the name of a built-in map, or a sequence of colours."""


def resolve_stops(colors: object) -> ColorStops:
    """Return a colour map's colours as numbers, resolving colour names.

    Args:
        colors: A built-in map's name, or a sequence of at least two colours
            (names, hex strings or ``(r, g, b)`` tuples).

    Returns:
        The colours, an ``(M, 4)`` array.

    Raises:
        ValueError: If the map is unknown or a colour is not one.
    """
    if isinstance(colors, str) or colors is None:
        return check_stops(DEFAULT_COLOR_MAP if colors is None else colors)
    try:
        items = list(colors)  # type: ignore[call-overload]
    except TypeError as error:
        msg = "a colour map is a name or a sequence of colours"
        raise ValueError(msg) from error
    resolved: list[Any] = []
    for item in items:
        if isinstance(item, str):
            try:
                resolved.append(tuple(pygame.Color(item)))
            except ValueError as error:
                msg = f"unknown colour {item!r} in a colour map"
                raise ValueError(msg) from error
        else:
            resolved.append(item)
    return check_stops(resolved)


def color_map(
    values: ArrayLike,
    colors: ColorMapLike = DEFAULT_COLOR_MAP,
    value_range: tuple[float, float] | None = None,
    *,
    log: bool = False,
) -> NDArray[np.uint8]:
    """Return a colour for each value, along a colour map.

    For colouring points by a value, with
    :meth:`Grid.points(color=...) <pygridplot.Grid.points>`, and saying
    which colour is which with :meth:`~pygridplot.Grid.color_bar`::

        colors = pgp.color_map(speed, "plasma", (0, 5))
        grid.points(screen, positions, color=colors)
        grid.color_bar(screen, "plasma", (0, 5), title="speed")

    Values are mapped linearly from ``value_range[0]`` (the map's first
    colour) to ``value_range[1]`` (its last), or on a log scale with
    ``log=True``. Values beyond the range take the colour of its end; NaN
    (and with ``log``, values not positive) are transparent, so their points
    are not drawn.
    Without a range, the values' own smallest and largest are used; give
    one for values that change from frame to frame, so a colour keeps its
    meaning.

    Args:
        values: The values: a number, a sequence or an array, of any shape.
        colors: A built-in map, ``"viridis"`` (default), ``"plasma"``,
            ``"inferno"``, ``"magma"``, ``"coolwarm"`` (diverging, for values
            either side of a middle) or ``"gray"``; or a sequence of two or
            more colours, evenly spaced from low to high (names, hex strings
            or ``(r, g, b)`` tuples).
        value_range: ``(low, high)``, the values at the map's ends (default:
            the values' own).
        log: Map the values on a log scale (they and the range must be
            positive).

    Returns:
        RGBA bytes, of shape ``values.shape + (4,)``: ready for ``color=``.

    Raises:
        ValueError: If the map or the range is not valid, or the values are
            not numbers.
    """
    stops = resolve_stops(colors)
    bounds = check_range(value_range, log=log)
    try:
        data = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "values must be numbers"
        raise ValueError(msg) from error
    return map_values(data, stops, bounds, log=bool(log))
