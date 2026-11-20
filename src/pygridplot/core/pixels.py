"""The library's single source of truth for snapping to whole pixels.

Screen coordinates follow pygame's drawing convention: **integer coordinates
are pixel centres**, so pixel column ``k`` covers the continuous interval
``[k - 0.5, k + 0.5)``. A continuous coordinate ``p`` produced by
:class:`~pygridplot.core.view.View` therefore belongs to pixel
``floor(p + 0.5)``.

Every part of the library that needs to land on whole pixels (the renderer
drawing crisp grid lines, label layout aligning text to those lines, and so
on) calls these helpers. That keeps things consistent: a label centred on a
grid line stays exactly centred on the pixels that line was drawn with, so
labels never wobble relative to their lines while zooming.

Rounding is always *half up* (``floor(p + 0.5)``). Python's built-in
:func:`round` uses banker's rounding, which would snap neighbouring
half-pixel positions in opposite directions and make grid spacing uneven.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def snap(value: float) -> int:
    """Return the pixel that contains a continuous screen coordinate.

    Args:
        value: Continuous screen coordinate in pixels.

    Returns:
        The index of the pixel whose centre is nearest, rounding halves up.
    """
    return math.floor(value + 0.5)


def snap_array(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Vectorised version of :func:`snap`.

    Args:
        values: Continuous screen coordinates in pixels.

    Returns:
        The pixel index of each coordinate.
    """
    return np.floor(values + 0.5).astype(np.int64)


def line_start(center: float, width: int) -> int:
    """Return the first pixel of a ``width``-pixel line centred on ``center``.

    The line covers ``width`` consecutive pixels, chosen so that their middle
    is as close to ``center`` as possible. For a 1-pixel line this is the
    same as :func:`snap`. For a 2-pixel line centred exactly on a pixel centre
    the extra pixel goes to the right (or below).

    Args:
        center: Continuous coordinate of the line's centre in pixels.
        width: Line thickness in whole pixels (at least 1).

    Returns:
        Index of the first (left-most or top-most) pixel the line covers.
    """
    return math.floor(center + 1.0 - width / 2.0)


def line_starts(centers: NDArray[np.float64], width: int) -> NDArray[np.int64]:
    """Vectorised version of :func:`line_start`.

    Args:
        centers: Continuous coordinates of the line centres in pixels.
        width: Line thickness in whole pixels (at least 1).

    Returns:
        Index of the first pixel covered by each line.
    """
    return np.floor(centers + (1.0 - width / 2.0)).astype(np.int64)
