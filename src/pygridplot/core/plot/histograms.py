"""Binning samples for histograms: anchored bins and an automatic width.

:meth:`pygridplot.Grid.histogram` draws the distribution of raw samples as
bars, one per bin, as high as the number of samples in it (or as the
density they estimate). In immediate mode the samples are often changing
(a simulation adding samples every frame), so the bins must not move with
them: a bin of width ``h`` is always ``[k h, (k + 1) h)`` for a whole
number ``k``, wherever the samples lie, and a sample far from the others
simply gets a bin of its own far away. Panning and zooming never change the
bins either: they depend on the samples alone.

Without a width, :func:`auto_bin_width` chooses one by the
Freedman-Diaconis rule (twice the interquartile range over the cube root
of the number of samples, which suits most distributions), rounded to the
nearest 1, 2 or 5 times a power of ten, so that bin edges fall on round
numbers and the width changes only now and then as samples arrive.

Bins are found in *transformed* coordinates (``log10`` on a log x axis),
so on a log axis they are equal on screen (a width there is in decades),
like every sampler's grid. Explicit bin edges are used as given, with
NumPy's rule that the last bin includes its right edge.

Binning is sparse: only bins with samples are kept, so a width far smaller
than the spread of the samples costs no more than the samples themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.transforms import AxisTransform

DENSE_FACTOR: Final = 8
"""Bins are counted with a dense array when their span is at most this many
times the number of samples, and by sorting otherwise."""

LARGEST_INDEX: Final = 2.0**52
"""Samples further than this many bins from zero cannot be binned exactly
(their bin number would not be a whole float) and are left out."""


@dataclass(frozen=True, slots=True, eq=False)
class Bins:
    """The bins that have samples in them, in order.

    Attributes:
        index: Each bin's number; neighbouring bins have consecutive numbers.
        left: Each bin's left edge, in transformed coordinates.
        right: Each bin's right edge, likewise.
        counts: How many samples each bin holds.
    """

    index: NDArray[np.int64]
    left: NDArray[np.float64]
    right: NDArray[np.float64]
    counts: NDArray[np.float64]


def as_samples(samples: ArrayLike) -> NDArray[np.float64]:
    """Return samples as a flat array of floats (a new array).

    Args:
        samples: The samples: a sequence or array of numbers. NaN and
            infinite samples are left out when binned.

    Returns:
        The samples.

    Raises:
        ValueError: If they are not numbers.
    """
    try:
        values = np.array(samples, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "samples must be numbers: a sequence or an array"
        raise ValueError(msg) from error
    return values.reshape(-1)


def check_edges(edges: object) -> NDArray[np.float64] | None:
    """Return valid bin edges, or ``None``.

    Args:
        edges: ``None``, or at least two finite, increasing numbers.

    Returns:
        The edges as a new array, or ``None``.

    Raises:
        ValueError: If they are not finite, increasing numbers, at least two.
    """
    if edges is None:
        return None
    try:
        values = np.array(edges, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "edges must be numbers"
        raise ValueError(msg) from error
    if (
        values.ndim != 1
        or values.shape[0] < 2
        or not np.all(np.isfinite(values))
        or not np.all(np.diff(values) > 0)
    ):
        msg = "edges must be at least two finite numbers, increasing"
        raise ValueError(msg)
    return values


def nice_width(width: float) -> float:
    """Round a positive width to the nearest 1, 2 or 5 times a power of ten.

    Args:
        width: The width.

    Returns:
        The nearest of 1, 2, 5 and 10 times ``10 ** floor(log10(width))``,
        nearest on a log scale.
    """
    power = 10.0 ** math.floor(math.log10(width))
    best = min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(math.log(width / (m * power))))
    return best * power


def auto_bin_width(u: NDArray[np.float64]) -> float:
    """Return a bin width for samples, by the Freedman-Diaconis rule.

    Args:
        u: The samples (finite), in transformed coordinates.

    Returns:
        ``2 × IQR / n^(1/3)`` rounded by :func:`nice_width`. Where the
        interquartile range is zero (most samples equal), the range over
        ``log2(n) + 1`` (Sturges' rule) is used instead, and for samples all
        equal, a width of 1.
    """
    count = u.shape[0]
    if count == 0:
        return 1.0
    low, high = np.percentile(u, [25.0, 75.0])
    width = 2.0 * float(high - low) / count ** (1.0 / 3.0)
    if not width > 0:
        spread = float(np.max(u) - np.min(u))
        width = spread / (math.log2(count) + 1.0) if count > 1 else 0.0
    if not (width > 0 and math.isfinite(width)):
        return 1.0
    return nice_width(width)


def bin_samples(u: NDArray[np.float64], width: float) -> Bins:
    """Count samples in the anchored bins ``[k w, (k + 1) w)``.

    Args:
        u: The samples (finite), in transformed coordinates.
        width: The bins' width ``w``, in transformed units.

    Returns:
        The bins that have samples, in order.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        k = np.floor(u / width)
    k = k[np.abs(k) < LARGEST_INDEX].astype(np.int64)
    if k.shape[0] == 0:
        empty = np.zeros(0)
        return Bins(np.zeros(0, dtype=np.int64), empty, empty, empty)
    lowest, highest = int(k.min()), int(k.max())
    if highest - lowest <= DENSE_FACTOR * k.shape[0]:
        counts = np.bincount(k - lowest)
        index = np.flatnonzero(counts)
        found = counts[index].astype(np.float64)
        index = index.astype(np.int64) + lowest
    else:
        index, tallies = np.unique(k, return_counts=True)
        found = tallies.astype(np.float64)
    left = index.astype(np.float64) * width
    return Bins(index, left, left + width, found)


def bin_by_edges(
    samples: NDArray[np.float64],
    edges: NDArray[np.float64],
    transform: AxisTransform,
) -> Bins:
    """Count samples in bins with explicit edges (NumPy's rule).

    Args:
        samples: The samples (finite), in world coordinates.
        edges: The bin edges, in world coordinates, increasing.
        transform: The x axis' transform, from world to transformed x.

    Returns:
        The bins that have samples, in order, their edges transformed. Bins
        whose edges cannot be shown (not positive on a log axis) are left
        out.
    """
    counts, _ = np.histogram(samples, edges)
    index = np.flatnonzero(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        placed = np.asarray(transform.forward(edges), dtype=np.float64)
    left, right = placed[index], placed[index + 1]
    keep = np.isfinite(left) & np.isfinite(right)
    index = index[keep]
    return Bins(
        index.astype(np.int64),
        left[keep],
        right[keep],
        counts[index].astype(np.float64),
    )
