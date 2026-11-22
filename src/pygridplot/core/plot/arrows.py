"""Arrows as polygons: a shaft and a triangular head, in screen pixels.

An arrow is drawn as **one polygon**, shaft and head together, rather than
a stroke with a head on top. With a translucent colour, a separate shaft
and head would overlap where they meet and darken there; one polygon is
filled once. All the arrows of a call become rings of one
:class:`~pygridplot.core.primitives.Fill` (nonzero rule), so a vector field
of a thousand arrows is rasterised in one pass, and arrows that overlap
merge instead of darkening each other.

Each ring has seven vertices, in order round the outline: a corner of the
tail, the corner of the shaft on the same side where the head begins, the
barb on that side, the tip, the other barb, the shaft's other corner at
the head, and the tail's other corner. The tail is square. An arrow
shorter than its head is drawn as a smaller head alone, with the same
proportions, so short vectors in a field still show their direction.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

HEAD_ASPECT: Final = 0.8
"""The width of an arrowhead across its barbs, as a fraction of its length."""

AUTO_HEAD_PER_WIDTH: Final = 3.0
"""An automatic head is this many shaft widths long, plus :data:`AUTO_HEAD_BASE`."""

AUTO_HEAD_BASE: Final = 4.0
"""Pixels added to every automatic head, so heads of thin arrows stay visible."""


def head_shape(width: float, head_size: float | None) -> tuple[float, float]:
    """Return the length and width of an arrowhead, in pixels.

    Args:
        width: The shaft's width in pixels.
        head_size: The head's length in pixels, or ``None`` for one in
            proportion to the shaft.

    Returns:
        ``(length, width)`` of the head (both 0 for no head).
    """
    length = (
        head_size
        if head_size is not None
        else (AUTO_HEAD_PER_WIDTH * width + AUTO_HEAD_BASE)
    )
    return length, HEAD_ASPECT * length


def arrow_rings(
    tails: NDArray[np.float64],
    tips: NDArray[np.float64],
    width: float,
    head: tuple[float, float],
) -> NDArray[np.float64]:
    """Return the outline of each arrow, for all the arrows at once.

    Args:
        tails: Where each arrow starts, in pixels, shape ``(N, 2)``.
        tips: Where each arrow points to, likewise.
        width: Shaft width in pixels.
        head: ``(length, width)`` of the head in pixels (see
            :func:`head_shape`).

    Returns:
        The rings, shape ``(M, 7, 2)``: one per arrow with finite ends that
        are not in the same place (the others have no direction and are
        left out), in the order given.
    """
    delta = tips - tails
    length = np.hypot(delta[:, 0], delta[:, 1])
    with np.errstate(invalid="ignore"):
        keep = np.isfinite(length) & (length > 0)
    if not keep.all():
        tails, tips, delta, length = tails[keep], tips[keep], delta[keep], length[keep]
    if length.shape[0] == 0:
        return np.zeros((0, 7, 2))
    along = delta / length[:, None]  # unit vectors from tail to tip
    across = np.column_stack((-along[:, 1], along[:, 0]))
    head_length, head_width = head
    if head_length > 0:
        # A head longer than its arrow shrinks to fit, keeping its shape.
        shrink = np.minimum(1.0, length / head_length)[:, None]
        reach = head_length * shrink * along
        barb = head_width / 2 * shrink * across
    else:
        reach = barb = np.zeros_like(along)
    base = tips - reach
    shaft = width / 2 * across
    return np.stack(
        (
            tails + shaft,
            base + shaft,
            base + barb,
            tips,
            base - barb,
            base - shaft,
            tails - shaft,
        ),
        axis=1,
    )
