"""The area between two graphs, as polygons.

Each graph comes from the function sampler as *pieces*: polylines in order
of increasing x, broken wherever the function is undefined, jumps or has a
pole. The area between two graphs exists only where both are defined, so
it is built interval by interval:

1. **Overlaps.** Two sorted lists of pieces are swept together: wherever a
   piece of one graph and a piece of the other cover the same x interval,
   there is a part of the area. This takes one pass over both lists.
2. **Rings.** On each overlap, both pieces are cut to the interval (their
   ends interpolated exactly at its bounds), and the ring is the first
   piece forwards followed by the second backwards. Where the graphs cross,
   the ring crosses itself, and the nonzero rule fills both lobes.

So the area has a straight vertical edge wherever either graph stops (a
jump, a domain edge, a pole), and none elsewhere.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rings_between(
    first: list[NDArray[np.float64]], second: list[NDArray[np.float64]]
) -> list[NDArray[np.float64]]:
    """Return the rings enclosing the area between two graphs.

    Args:
        first: The first graph's pieces, each of shape ``(N, 2)`` with
            ``N >= 2`` and strictly increasing x, in order of x and not
            overlapping each other.
        second: The second graph's pieces, likewise.

    Returns:
        One ring per x interval where both graphs are defined, in order of
        x, each of shape ``(M, 2)``, to fill with the nonzero rule.
    """
    rings = []
    i = j = 0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        low = max(a[0, 0], b[0, 0])
        high = min(a[-1, 0], b[-1, 0])
        if high > low:
            rings.append(
                np.concatenate((section(a, low, high), section(b, low, high)[::-1]))
            )
        # Move on from whichever piece ends first.
        if a[-1, 0] <= b[-1, 0]:
            i += 1
        else:
            j += 1
    return rings


def section(piece: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """Cut a piece of a graph to ``low <= x <= high``, interpolating its ends.

    Args:
        piece: The piece, shape ``(N, 2)``, x strictly increasing and
            covering ``[low, high]``.
        low: Start of the interval.
        high: End of the interval (greater than ``low``).

    Returns:
        The points of the piece inside the interval, with points added
        exactly at ``low`` and ``high``.
    """
    xs, ys = piece[:, 0], piece[:, 1]
    inside = (xs > low) & (xs < high)
    ends = np.interp([low, high], xs, ys)
    return np.vstack(
        ([low, ends[0]], piece[inside], [high, ends[1]]),
    ).astype(np.float64)
