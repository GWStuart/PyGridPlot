"""Colour maps: colouring by value.

A scatter plot can show a third value by colour (``points(color=...)``), and
a colour bar says which colour is which value. :func:`map_values` turns
values into colours along a **colour map**: a sequence of colours from low
to high, evenly spaced, between which colours are interpolated linearly.

The built-in maps (:data:`COLOR_MAPS`) are the perceptually even maps of
matplotlib (``viridis``, ``plasma``, ``inferno``, ``magma``), sampled at
eleven points each, which keeps their look to within a few levels of each
channel: equal steps of value look like equal steps of colour, they read
the same in grey, and colour-blind readers can read them. ``coolwarm``
diverges from blue through grey to red, for values on both sides of a
middle (zero, say), and ``gray`` runs from black to white. Any sequence of
two or more colours is a map too.

Values are mapped linearly between the ends of a range, or on a log scale
(``log=True``, for values spanning decades); values beyond the range take
the colour of its end, and NaN (or, on a log scale, values not positive)
is transparent, so a point without a value is not drawn. Without a range,
the values' own smallest and largest are used: right for a picture of
fixed data, but a colour then changes meaning when the data changes, so
moving data should be given a fixed range.

This module is pure: colour *names* (``"navy"``) are resolved by the plane
before they get here, so maps arrive as numbers.
"""

from __future__ import annotations

import math
import numbers
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

ColorStops: TypeAlias = NDArray[np.float64]
"""A colour map's colours, low to high: an ``(M, 4)`` array of RGBA in [0, 255]."""


def _hex(*codes: str) -> tuple[tuple[int, int, int], ...]:
    """Return colours written as ``"#rrggbb"``."""
    return tuple(
        (int(code[1:3], 16), int(code[3:5], 16), int(code[5:7], 16)) for code in codes
    )


COLOR_MAPS: Final[dict[str, tuple[tuple[int, int, int], ...]]] = {
    "viridis": _hex(
        "#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c",
        "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725",
    ),
    "plasma": _hex(
        "#0d0887", "#41049d", "#6a00a8", "#8f0da4", "#b12a90", "#cc4778",
        "#e16462", "#f2844b", "#fca636", "#fcce25", "#f0f921",
    ),
    "inferno": _hex(
        "#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754",
        "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4",
    ),
    "magma": _hex(
        "#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", "#b73779",
        "#de4968", "#f7705c", "#fe9f6d", "#fecf92", "#fcfdbf",
    ),
    "coolwarm": (
        (59, 76, 192), (98, 130, 234), (141, 176, 254), (184, 208, 249),
        (221, 221, 221), (245, 196, 173), (244, 154, 123), (222, 96, 77),
        (180, 4, 38),
    ),
    "gray": ((0, 0, 0), (255, 255, 255)),
}  # fmt: skip
"""The built-in colour maps, by name: their colours from low to high."""

DEFAULT_COLOR_MAP: Final = "viridis"
"""The map used when none is named."""


def check_stops(colors: object) -> ColorStops:
    """Return a colour map's colours as an array.

    Args:
        colors: The name of a built-in map, or a sequence of at least two
            colours given as numbers: ``(r, g, b)`` or ``(r, g, b, a)`` in
            ``[0, 255]``.

    Returns:
        The colours, an ``(M, 4)`` array.

    Raises:
        ValueError: If the name is unknown, or the colours are not at least
            two colours of three or four numbers in ``[0, 255]``.
    """
    if isinstance(colors, str):
        if colors not in COLOR_MAPS:
            known = ", ".join(sorted(COLOR_MAPS))
            msg = f"unknown colour map {colors!r}; the built-in maps are {known}"
            raise ValueError(msg)
        colors = COLOR_MAPS[colors]
    try:
        rows = [tuple(float(channel) for channel in color) for color in colors]  # type: ignore[attr-defined]
    except (TypeError, ValueError) as error:
        msg = "a colour map is a name or a sequence of colours, (r, g, b) in [0, 255]"
        raise ValueError(msg) from error
    if len(rows) < 2 or any(len(row) not in (3, 4) for row in rows):
        msg = "a colour map needs at least two colours, each (r, g, b) or (r, g, b, a)"
        raise ValueError(msg)
    stops = np.array([(*row, 255.0)[:4] for row in rows], dtype=np.float64)
    if not (np.all(np.isfinite(stops)) and np.min(stops) >= 0 and np.max(stops) <= 255):
        msg = "colour map values must be within [0, 255]"
        raise ValueError(msg)
    return stops


def check_range(
    value_range: object, *, log: bool = False
) -> tuple[float, float] | None:
    """Return a valid value range for a colour map, or ``None`` for automatic.

    Args:
        value_range: ``None``, or ``(low, high)`` with ``low < high``, finite
            (and positive with ``log``).
        log: Whether values are mapped on a log scale.

    Returns:
        The range, or ``None``.

    Raises:
        ValueError: If the range is not valid.
    """
    if value_range is None:
        return None
    try:
        pair: tuple[object, ...] = tuple(value_range)  # type: ignore[arg-type]
    except TypeError as error:
        msg = f"range must be (low, high), got {value_range!r}"
        raise ValueError(msg) from error
    if len(pair) != 2 or not all(
        isinstance(v, numbers.Real) and not isinstance(v, bool) for v in pair
    ):
        msg = f"range must be two numbers, (low, high), got {value_range!r}"
        raise ValueError(msg)
    first, second = pair
    low, high = float(first), float(second)  # type: ignore[arg-type]
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        msg = f"range must be finite, with low < high, got {value_range!r}"
        raise ValueError(msg)
    if log and low <= 0:
        msg = f"a log range must be positive, got {value_range!r}"
        raise ValueError(msg)
    return low, high


def value_range(
    values: NDArray[np.float64], *, log: bool = False
) -> tuple[float, float] | None:
    """Return the smallest and largest of values that can be mapped.

    Args:
        values: The values (NaN, infinities and, with ``log``, values not
            positive are ignored).
        log: Whether values are mapped on a log scale.

    Returns:
        ``(low, high)``, equal when all values are, or ``None`` if there is no
        value to map.
    """
    with np.errstate(invalid="ignore"):
        usable = values[np.isfinite(values) & ((values > 0) | (not log))]
    if usable.size == 0:
        return None
    return float(np.min(usable)), float(np.max(usable))


def map_values(
    values: ArrayLike,
    stops: ColorStops,
    value_range: tuple[float, float] | None = None,
    *,
    log: bool = False,
) -> NDArray[np.uint8]:
    """Return the colour of each value along a colour map.

    Args:
        values: The values, of any shape.
        stops: The colour map's colours (see :func:`check_stops`).
        value_range: ``(low, high)`` mapped onto the map's ends, or ``None``
            for the values' own smallest and largest. When they are equal,
            every value gets the middle colour.
        log: Map ``log10`` of the values (and of the range).

    Returns:
        RGBA bytes, shape ``values.shape + (4,)``. NaN (and values that are
        not positive, with ``log``) are transparent black.
    """
    data = np.asarray(values, dtype=np.float64)
    bounds = value_range if value_range is not None else _own_range(data, log)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = _fractions(data, bounds, log)
    known = np.isfinite(t)
    t = np.clip(np.where(known, t, 0.0), 0.0, 1.0)
    position = t * (stops.shape[0] - 1)
    index = np.minimum(np.floor(position).astype(np.intp), stops.shape[0] - 2)
    part = (position - index)[..., None]
    colors = stops[index] * (1.0 - part) + stops[index + 1] * part
    colors[~known] = 0.0
    rgba: NDArray[np.uint8] = np.floor(colors + 0.5).astype(np.uint8)
    return rgba


def _own_range(data: NDArray[np.float64], log: bool) -> tuple[float, float]:
    """The values' own range, or ``(0, 1)`` when none can be mapped."""
    found = value_range(data, log=log)
    return found if found is not None else (1.0, 10.0) if log else (0.0, 1.0)


def _fractions(
    data: NDArray[np.float64], bounds: tuple[float, float], log: bool
) -> NDArray[np.float64]:
    """Where each value lies between the range's ends: 0 at low, 1 at high.

    Values on a log scale that are not positive are NaN. When the ends are
    equal, every value is in the middle.
    """
    low, high = bounds
    if log:
        data = np.where(data > 0, np.log10(np.where(data > 0, data, 1.0)), np.nan)
        low, high = math.log10(low), math.log10(high)
    if high == low:
        return np.where(np.isnan(data), np.nan, 0.5)
    return (data - low) / (high - low)
