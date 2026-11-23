"""Data plots: colours and sizes per point, error bars, steps and bars.

Plotting *data* (measurements, samples, the state of a running simulation)
needs a few things beyond plotting functions, and each of them goes into the
``Grid`` method it improves rather than a parallel API:

* :meth:`~pygridplot.Grid.points` takes a colour and a size **per point**
  (a scatter plot coloured or sized by a value) and **error bars**
  (:func:`per_point_colors`, :func:`per_point_radii`, :func:`as_errors`);
* :meth:`~pygridplot.Grid.polyline` draws **steps** between its vertices
  (:func:`step_vertices`), for counts, sampled-and-held signals and
  empirical distribution functions;
* :meth:`~pygridplot.Grid.bars` draws **bars** from a baseline
  (:func:`as_bars`, :func:`auto_bar_width`).

Histograms, which bin raw samples, are in
:mod:`pygridplot.core.plot.histograms`.

Everything here validates and normalises the program's data; drawing is in
:mod:`pygridplot.core.plot.builders`. Bars and steps are laid out in
*transformed* coordinates (``log10`` on a log axis), like every sampler, so
they look the same on log axes: a bar's width is in decades there.
"""

from __future__ import annotations

import math
import numbers
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import ColorLike

Step: TypeAlias = Literal["pre", "post", "mid"]
"""Where a step plot steps between two vertices.

* ``"post"``: each value holds from its x until the next x (a count, a
  sampled-and-held signal, an empirical distribution function).
* ``"pre"``: each value holds from the previous x up to its own.
* ``"mid"``: the steps are half way between the xs (each value holds
  around its x, as in a bar chart without the bars).
"""

STEPS: tuple[Step, ...] = ("pre", "post", "mid")
"""All step kinds."""

PointColors: TypeAlias = "tuple[ColorLike, ...] | NDArray[np.uint8]"
"""A colour per point: colour values, or an ``(N, 4)`` array of RGBA bytes."""

AUTO_BAR_FRACTION = 0.8
"""An automatic bar is this fraction of the gap between neighbouring bars."""


def is_one_color(value: object) -> bool:
    """Return whether a value is a single colour rather than one per point.

    A string, or a sequence of three or four numbers (such as ``(255, 0, 0)``
    or a ``pygame.Color``), is one colour.

    Args:
        value: The ``color=`` given.

    Returns:
        Whether it is one colour.
    """
    if isinstance(value, str):
        return True
    if isinstance(value, (bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.shape[0] in (3, 4) and _numeric(value)
    try:
        items = list(value)  # type: ignore[call-overload]
    except TypeError:
        return False
    return len(items) in (3, 4) and all(
        isinstance(item, numbers.Real) and not isinstance(item, bool) for item in items
    )


def per_point_colors(color: object, count: int) -> PointColors | None:
    """Return the colour of each point, if ``color`` gives one per point.

    Args:
        color: The ``color=`` given: ``None`` or one colour (then ``None``
            is returned), or a colour for each point: a sequence of colours
            (names or ``(r, g, b)`` tuples), or an ``(N, 3)`` or ``(N, 4)``
            array of values in ``[0, 255]``.
        count: The number of points.

    Returns:
        ``None`` for one colour; otherwise the colours, as an ``(N, 4)``
        array of bytes when they are numbers (fast to compare and draw) or
        as a tuple of the values given.

    Raises:
        ValueError: If there is not one colour per point, or numbers are
            outside ``[0, 255]``.
        TypeError: If ``color`` is neither a colour nor a sequence of them.
    """
    if color is None or is_one_color(color):
        return None
    if isinstance(color, (bytes, bytearray)):
        msg = f"color must be a colour or one colour per point, got {color!r}"
        raise TypeError(msg)
    try:
        items = list(color)  # type: ignore[call-overload]
    except TypeError as error:
        msg = f"color must be a colour or one colour per point, got {color!r}"
        raise TypeError(msg) from error
    if len(items) != count:
        msg = f"color gives {len(items)} colours for {count} points"
        raise ValueError(msg)
    if all(not isinstance(item, str) and is_one_color(item) for item in items):
        if count == 0:
            return np.zeros((0, 4), dtype=np.uint8)
        try:
            values = np.array(items, dtype=np.float64)
        except ValueError:  # three- and four-number colours mixed
            values = np.array([(*item, 255)[:4] for item in items], dtype=np.float64)
        if not (
            np.all(np.isfinite(values))
            and np.min(values) >= 0
            and np.max(values) <= 255
        ):
            msg = "colour values must be within [0, 255]"
            raise ValueError(msg)
        if values.shape[1] == 3:
            values = np.column_stack((values, np.full(count, 255.0)))
        return np.floor(values + 0.5).astype(np.uint8)
    for item in items:
        if not (isinstance(item, str) or is_one_color(item)):
            msg = f"color must be a colour or one colour per point, got {item!r}"
            raise TypeError(msg)
    return tuple(items)


def per_point_radii(radius: object, count: int) -> NDArray[np.float64] | None:
    """Return the size of each point, if ``radius`` gives one per point.

    Args:
        radius: The ``radius=`` given: ``None`` or a number (then ``None`` is
            returned), or a sequence of ``count`` sizes in pixels. A NaN size
            draws no marker for that point.
        count: The number of points.

    Returns:
        The sizes, or ``None``.

    Raises:
        ValueError: If there is not one size per point, or a size is
            negative or infinite.
    """
    if radius is None or isinstance(radius, numbers.Real):
        return None
    try:
        sizes = np.array(radius, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"radius must be a number or one number per point, got {radius!r}"
        raise ValueError(msg) from error
    if sizes.ndim != 1 or sizes.shape[0] != count:
        msg = f"radius gives {sizes.size} sizes for {count} points"
        raise ValueError(msg)
    with np.errstate(invalid="ignore"):
        if np.any(np.isinf(sizes)) or np.any(sizes < 0):
            msg = "sizes must be non-negative and finite (NaN for no marker)"
            raise ValueError(msg)
    return np.nan_to_num(sizes, nan=0.0)


def as_errors(error: object, count: int, name: str) -> NDArray[np.float64] | None:
    """Return the error bars' lengths below and above each point.

    Args:
        error: ``None``, one number for every point, ``count`` numbers (the
            same below and above), or two rows of ``count`` numbers (below,
            then above). A NaN draws no bar for that point.
        count: The number of points.
        name: The argument's name, for error messages.

    Returns:
        ``None``, or an array of shape ``(2, count)``: the lengths below and
        above.

    Raises:
        ValueError: If the shape does not fit, or a length is negative.
    """
    if error is None:
        return None
    try:
        values = np.array(error, dtype=np.float64)
    except (TypeError, ValueError) as error_:
        msg = f"{name} must be numbers, got {error!r}"
        raise ValueError(msg) from error_
    if values.ndim == 0:
        below = above = np.full(count, float(values))
    elif values.shape == (count,):
        below = above = values
    elif values.shape == (2, count):
        below, above = values[0], values[1]
    else:
        msg = (
            f"{name} must be a number, {count} numbers, or two rows of {count} "
            f"(below, above); got shape {values.shape}"
        )
        raise ValueError(msg)
    lengths = np.stack((below, above))
    with np.errstate(invalid="ignore"):
        if np.any(lengths < 0) or np.any(np.isinf(lengths)):
            msg = f"{name} must be non-negative and finite (NaN for no bar)"
            raise ValueError(msg)
    return lengths


def check_step(step: object) -> Step | None:
    """Return a valid step kind.

    Args:
        step: ``None`` (no steps), ``"pre"``, ``"post"`` or ``"mid"``.

    Returns:
        The step kind.

    Raises:
        ValueError: If it is not one of them.
    """
    if step is None:
        return None
    for known in STEPS:
        if step == known:
            return known
    msg = f"step must be one of {STEPS} or None, got {step!r}"
    raise ValueError(msg)


def step_vertices(
    u: NDArray[np.float64], w: NDArray[np.float64], step: Step
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the vertices of a step plot through ``(u, w)``.

    The stairs are horizontal and vertical, so ``u`` and ``w`` should be
    transformed (screen-linear) coordinates; a ``"mid"`` step is then half
    way on screen. A NaN vertex breaks the line, as in a polyline: each run
    between breaks gets its own stairs, and the breaks are kept.

    Args:
        u: Transformed x of each vertex.
        w: Transformed y of each vertex.
        step: Where to step (see :data:`Step`).

    Returns:
        The stairs' vertices ``(u, w)``, with NaN between runs.
    """
    broken = np.isnan(u) | np.isnan(w)
    edges = np.flatnonzero(np.diff(np.r_[True, broken, True].astype(np.int8)))
    us: list[NDArray[np.float64]] = []
    ws: list[NDArray[np.float64]] = []
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        ru, rw = u[first:last], w[first:last]
        if ru.shape[0] > 1:
            if step == "post":
                ru, rw = np.repeat(ru, 2)[1:], np.repeat(rw, 2)[:-1]
            elif step == "pre":
                ru, rw = np.repeat(ru, 2)[:-1], np.repeat(rw, 2)[1:]
            else:
                middle = np.repeat((ru[:-1] + ru[1:]) / 2, 2)
                ru, rw = np.r_[ru[0], middle, ru[-1]], np.repeat(rw, 2)
        if us:
            us.append(np.array([np.nan]))
            ws.append(np.array([np.nan]))
        us.append(ru)
        ws.append(rw)
    if not us:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(us), np.concatenate(ws)


def as_bars(data: object, start: object = 0) -> NDArray[np.float64]:
    """Return bars as ``(x, top)`` rows.

    Args:
        data: The bars' heights, a 1-D sequence (the bars stand at ``start``,
            ``start + 1``, ...), or their ``(x, top)`` pairs: a sequence of
            pairs or an ``(N, 2)`` array. A NaN leaves a bar out.
        start: Where the first of a 1-D sequence's bars stands.

    Returns:
        A new array of shape ``(N, 2)``.

    Raises:
        ValueError: If ``data`` is not numbers in one of those shapes, or
            ``start`` is not a finite number.
    """
    if (
        isinstance(start, bool)
        or not isinstance(start, numbers.Real)
        or not math.isfinite(start)
    ):
        msg = f"start must be a finite number, got {start!r}"
        raise ValueError(msg)
    try:
        values = np.array(data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "bars must be numbers: heights, or (x, top) pairs"
        raise ValueError(msg) from error
    if values.size == 0:
        return np.zeros((0, 2))
    if values.ndim == 1:
        x = float(start) + np.arange(values.shape[0], dtype=np.float64)
        return np.column_stack((x, values))
    if values.ndim == 2 and values.shape[1] == 2:
        return values
    msg = f"bars must be heights or (x, top) pairs, got shape {values.shape}"
    raise ValueError(msg)


def auto_bar_width(u: NDArray[np.float64]) -> float:
    """Return a bar width for bars at transformed xs ``u``.

    Args:
        u: The bars' transformed x (NaN and infinities are ignored).

    Returns:
        :data:`AUTO_BAR_FRACTION` of the smallest gap between neighbouring
        bars, so bars never touch; or of 1 for a single bar.
    """
    places = np.unique(u[np.isfinite(u)])
    gaps = np.diff(places)
    gap = float(np.min(gaps)) if gaps.size else 1.0
    return AUTO_BAR_FRACTION * gap


def check_width(width: object, name: str = "width") -> float | None:
    """Return a valid width, or ``None`` for an automatic one.

    Args:
        width: The width given.
        name: The argument's name, for error messages.

    Returns:
        The width, or ``None``.

    Raises:
        ValueError: If it is not a positive, finite number.
    """
    if width is None:
        return None
    if isinstance(width, bool) or not isinstance(width, numbers.Real):
        msg = f"{name} must be a positive, finite number, got {width!r}"
        raise ValueError(msg)
    value = float(width)
    if not (math.isfinite(value) and value > 0):
        msg = f"{name} must be a positive, finite number, got {width!r}"
        raise ValueError(msg)
    return value


def _numeric(values: NDArray[np.generic]) -> bool:
    """Return whether an array holds real numbers (not strings or objects)."""
    return values.dtype.kind in "biuf"


DEFAULT_MARGIN = 0.05
"""The space :func:`data_range` leaves beyond the data, as a fraction of its spread."""


def data_range(
    values: object,
    *,
    margin: float = DEFAULT_MARGIN,
    include: object = None,
    log: bool = False,
) -> tuple[float, float] | None:
    """Return a range that shows all of some values, with a margin.

    For setting up the view from data, on one axis at a time::

        grid = pgp.Grid(x_range=pgp.data_range(xs), y_range=pgp.data_range(ys))
        grid.set_view(y_range=pgp.data_range(heights, include=0))  # bars from 0

    Args:
        values: Numbers of any shape (a sequence, an array, a column of
            points). NaN and infinities are ignored, and with ``log``, values
            that are not positive.
        margin: Space beyond the values on each side, as a fraction of their
            spread (on a log scale, of their spread in decades).
        include: A number, or numbers, the range must also show (``0`` for
            the baseline of bars, say).
        log: Whether the range is for a log axis: the margin is then a
            fraction of the decades the values span.

    Returns:
        ``(low, high)``, or ``None`` when there is no value to show (an empty
        stream, say), which :meth:`~pygridplot.Grid.set_view` and
        :class:`~pygridplot.Grid` take as "leave this axis as it is". Values
        all equal are given room around them: ±10% (±1 at zero), or half a
        decade on a log scale.

    Raises:
        ValueError: If ``values`` or ``include`` are not numbers, or
            ``margin`` is negative or not finite.
    """
    try:
        room_fraction = float(margin)
    except (TypeError, ValueError) as error:
        msg = f"margin must be a non-negative, finite number, got {margin!r}"
        raise ValueError(msg) from error
    if not (math.isfinite(room_fraction) and room_fraction >= 0):
        msg = f"margin must be a non-negative, finite number, got {margin!r}"
        raise ValueError(msg)
    try:
        data = np.array(values, dtype=np.float64).reshape(-1)
        extra = np.zeros(0) if include is None else np.array(include, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "values and include must be numbers"
        raise ValueError(msg) from error
    data = np.concatenate((data, extra.reshape(-1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = data[np.isfinite(data) & ((data > 0) | (not log))]
        if log:
            usable = np.log10(usable)
    if usable.size == 0:
        return None
    low, high = float(np.min(usable)), float(np.max(usable))
    if high == low:
        if log:
            low, high = low - 0.5, high + 0.5
        else:
            room = 0.1 * abs(low) if low != 0 else 1.0
            low, high = low - room, high + room
    else:
        room = room_fraction * (high - low)
        low, high = low - room, high + room
    if log:
        return 10.0**low, 10.0**high
    return low, high


def as_baselines(baseline: object, count: int) -> float | NDArray[np.float64]:
    """Return where bars stand: one value for all, or one per bar.

    Args:
        baseline: A finite number, or ``count`` numbers (NaN leaves that bar
            out).
        count: The number of bars.

    Returns:
        The number, or a new array of ``count`` numbers.

    Raises:
        ValueError: If it is neither, or a number is infinite.
    """
    if not isinstance(baseline, bool) and isinstance(baseline, numbers.Real):
        if not math.isfinite(baseline):
            msg = f"baseline must be a finite number, got {baseline!r}"
            raise ValueError(msg)
        return float(baseline)
    try:
        bases = np.array(baseline, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"baseline must be a number or one per bar, got {baseline!r}"
        raise ValueError(msg) from error
    if bases.shape != (count,):
        msg = f"baseline gives {bases.size} values for {count} bars"
        raise ValueError(msg)
    if np.any(np.isinf(bases)):
        msg = "baselines must be finite (NaN for no bar)"
        raise ValueError(msg)
    return bases
