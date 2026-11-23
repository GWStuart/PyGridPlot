"""Summaries of samples: box plots and violin plots.

:meth:`pygridplot.Grid.box_plot` and :meth:`pygridplot.Grid.violin_plot`
draw groups of samples side by side, one summary each:

* a **box plot** (:func:`box_stats`) shows the quartiles as a box, the
  median as a line across it, whiskers out to the furthest samples within
  ``whisker`` (1.5, Tukey's rule) interquartile ranges of the box, and the
  samples beyond them as points (outliers);
* a **violin plot** (:func:`violin_shape`) shows the density of the samples,
  estimated with a Gaussian kernel, as a shape mirrored about the group's
  position, from the smallest sample to the largest, with the median as a
  line across.

Quartiles are NumPy's (linear interpolation). The kernel's bandwidth is
Silverman's rule, ``0.9 min(s, IQR / 1.34) n^(-1/5)`` with ``s`` the
standard deviation, which suits most distributions and is not thrown by
outliers, or as given. The density is
estimated on a fine histogram of the samples (:data:`KDE_BINS` bins)
smoothed by the kernel, so a hundred thousand samples cost a few
milliseconds; the shape is then sampled at :data:`VIOLIN_POINTS` points.

Both are computed in *transformed* coordinates (``log10`` on a log axis),
like every sampler: on a log axis they summarise the logarithms of the
samples, which is what the axis shows.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

KDE_BINS: Final = 512
"""Bins of the fine histogram a violin's density is estimated on."""

VIOLIN_POINTS: Final = 96
"""Points along a violin's outline, on each side."""

DEFAULT_WHISKER: Final = 1.5
"""Tukey's whisker reach, in interquartile ranges beyond the box."""


@dataclass(frozen=True, slots=True, eq=False)
class BoxStats:
    """What a box plot shows of one group of samples.

    Attributes:
        low: The lower whisker's end: the smallest sample within reach.
        q1: The first quartile (the box's lower edge).
        median: The median.
        q3: The third quartile (the box's upper edge).
        high: The upper whisker's end: the largest sample within reach.
        outliers: The samples beyond the whiskers.
    """

    low: float
    q1: float
    median: float
    q3: float
    high: float
    outliers: NDArray[np.float64]


def as_groups(data: object) -> list[NDArray[np.float64]]:
    """Return groups of samples, each a flat array of floats (new arrays).

    Args:
        data: One group (a sequence or 1-D array of numbers), or several (a
            sequence of such sequences, which may differ in length, or a 2-D
            array, one group per row).

    Returns:
        The groups, in order.

    Raises:
        ValueError: If the data are not numbers in one of those shapes.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [_numbers(data)]
        if data.ndim == 2:
            return [_numbers(row) for row in data]
        msg = f"samples must be one group or a list of groups, got shape {data.shape}"
        raise ValueError(msg)
    try:
        items = list(data)  # type: ignore[call-overload]
    except TypeError as error:
        msg = "samples must be a sequence of numbers, or of groups of numbers"
        raise ValueError(msg) from error
    if all(isinstance(item, numbers.Real) for item in items):
        return [_numbers(items)]
    return [_numbers(item) for item in items]


def _numbers(values: object) -> NDArray[np.float64]:
    """Return one group of samples as a flat array of floats."""
    try:
        return np.array(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        msg = "samples must be numbers"
        raise ValueError(msg) from error


def check_positions(
    positions: object, start: object, count: int
) -> NDArray[np.float64]:
    """Return where each group's summary stands.

    Args:
        positions: ``None`` (then ``start``, ``start + 1``, ...), or one
            position per group.
        start: The first position, when ``positions`` is ``None``.
        count: The number of groups.

    Returns:
        The positions, a new array.

    Raises:
        ValueError: If there is not one finite position per group, or
            ``start`` is not a finite number.
    """
    if positions is None:
        if (
            isinstance(start, bool)
            or not isinstance(start, numbers.Real)
            or not math.isfinite(start)
        ):
            msg = f"start must be a finite number, got {start!r}"
            raise ValueError(msg)
        return float(start) + np.arange(count, dtype=np.float64)
    try:
        places = np.array(positions, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        msg = "positions must be numbers, one per group"
        raise ValueError(msg) from error
    if places.shape != (count,) or not np.all(np.isfinite(places)):
        msg = f"positions must be {count} finite numbers, one per group"
        raise ValueError(msg)
    return places


def check_whisker(whisker: object) -> float:
    """Return a valid whisker reach, in interquartile ranges.

    Args:
        whisker: A non-negative, finite number (``inf`` reaches every sample).

    Returns:
        It, as a float.

    Raises:
        ValueError: If it is negative or NaN.
    """
    try:
        reach = float(whisker)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        msg = f"whisker must be a number, got {whisker!r}"
        raise ValueError(msg) from error
    if not reach >= 0:
        msg = f"whisker must be non-negative, got {whisker!r}"
        raise ValueError(msg)
    return reach


def box_stats(
    samples: NDArray[np.float64], whisker: float = DEFAULT_WHISKER
) -> BoxStats | None:
    """Return what a box plot shows of some samples.

    Args:
        samples: The samples (finite ones are used).
        whisker: How far the whiskers may reach beyond the box, in
            interquartile ranges.

    Returns:
        The statistics, or ``None`` without a finite sample.
    """
    data = samples[np.isfinite(samples)]
    if data.size == 0:
        return None
    q1, median, q3 = (float(q) for q in np.percentile(data, [25.0, 50.0, 75.0]))
    reach = whisker * (q3 - q1)
    inside = data[(data >= q1 - reach) & (data <= q3 + reach)]
    low = float(np.min(inside)) if inside.size else q1
    high = float(np.max(inside)) if inside.size else q3
    outliers = data[(data < low) | (data > high)]
    return BoxStats(min(low, q1), q1, median, q3, max(high, q3), outliers)


def silverman_bandwidth(data: NDArray[np.float64]) -> float:
    """Return Silverman's rule-of-thumb bandwidth for a Gaussian kernel.

    Args:
        data: The samples (finite).

    Returns:
        ``0.9 min(s, IQR / 1.34) n^(-1/5)``, ``s`` the standard deviation,
        or 0 if the samples do not spread.
    """
    if data.size < 2:
        return 0.0
    sigma = float(np.std(data))
    q1, q3 = np.percentile(data, [25.0, 75.0])
    spread: float = min(sigma, float(q3 - q1) / 1.34) or sigma
    return 0.9 * spread * math.pow(data.size, -0.2)


def check_bandwidth(bandwidth: object) -> float | None:
    """Return a valid kernel bandwidth, or ``None`` for an automatic one.

    Args:
        bandwidth: ``None`` or a positive, finite number.

    Returns:
        It, or ``None``.

    Raises:
        ValueError: If it is not a positive, finite number.
    """
    if bandwidth is None:
        return None
    try:
        value = float(bandwidth)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        msg = f"bandwidth must be a positive number, got {bandwidth!r}"
        raise ValueError(msg) from error
    if not (math.isfinite(value) and value > 0):
        msg = f"bandwidth must be positive and finite, got {bandwidth!r}"
        raise ValueError(msg)
    return value


def violin_shape(
    samples: NDArray[np.float64], bandwidth: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the density of samples, from the smallest to the largest.

    Args:
        samples: The samples (finite ones are used).
        bandwidth: The Gaussian kernel's standard deviation, or ``None`` for
            :func:`silverman_bandwidth`.

    Returns:
        ``(values, density)``, :data:`VIOLIN_POINTS` points each, the density
        integrating to (about) 1 over all values; or ``None`` when the
        samples do not spread (fewer than two different values).
    """
    data = samples[np.isfinite(samples)]
    if data.size < 2:
        return None
    low, high = float(np.min(data)), float(np.max(data))
    if high == low:
        return None
    h = bandwidth if bandwidth is not None else silverman_bandwidth(data)
    if not h > 0:
        h = (high - low) / 10
    # A fine histogram, reaching four bandwidths beyond the samples, smoothed
    # by the kernel sampled at the bins' spacing.
    edge_low, edge_high = low - 4 * h, high + 4 * h
    counts, edges = np.histogram(data, bins=KDE_BINS, range=(edge_low, edge_high))
    step = edges[1] - edges[0]
    reach = min(math.ceil(4 * h / step), (KDE_BINS - 1) // 2)  # shorter than the bins
    offsets = np.arange(-reach, reach + 1) * step
    kernel = np.exp(-0.5 * (offsets / h) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same") / (data.size * step)
    centres = (edges[:-1] + edges[1:]) / 2
    values = np.linspace(low, high, VIOLIN_POINTS)
    return values, np.interp(values, centres, smooth)
