"""Where the arrows of a vector or slope field go, and which way they point.

A field is shown by sampling it on a lattice of points and drawing a short
arrow (or segment) at each. Three things make the lattice look right while
the view moves:

* **It is anchored to the world.** The nodes are multiples of a step in
  transformed coordinates, so panning moves every arrow rigidly and keeps
  sampling the same points (like the grids of implicit curves).
* **Its density stays about the same on screen.** The step is a power of
  two, chosen per axis so that neighbouring nodes are between half the
  requested spacing and the spacing apart in pixels.
* **Nothing pops while zooming.** Powers of two nest: zooming in until the
  step halves keeps every node and adds one between each pair. Those new
  nodes (the ones at odd multiples) *fade in* over the second half of the
  octave of zoom before the halving (:data:`FADE_START`): their opacity
  and length grow smoothly from 0 to 1, so they grow into place without
  touching their neighbours. Zooming out reverses it. So the fully drawn
  arrows are always between the spacing and twice that apart, the picture
  changes continuously, and over half of all zoom levels there are no
  fading arrows at all. A node new along both axes fades with the later
  of the two, so a frame has at most three levels of fade (two when the
  axes have the same scale), each drawn as one primitive.

Directions are mapped through the axis transforms' derivatives (see
:meth:`~pygridplot.core.transforms.AxisTransform.derivative`): a vector
``(u, v)`` at ``(x, y)`` points along ``(u T_x'(x), v T_y'(y))`` in
transformed coordinates. So arrows stay tangent to the field's flow lines,
and slope segments to the solution curves, on log axes too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.view import View

DEFAULT_SPACING: Final = 40.0
"""Default distance between neighbouring arrows, in pixels."""

MIN_SPACING: Final = 10.0
"""Smallest spacing allowed: it bounds the number of arrows in a frame."""

ARROW_FRACTION: Final = 0.8
"""The longest arrow (and every arrow, when lengths do not show magnitude),
as a fraction of the spacing."""

SLOPE_FRACTION: Final = 0.6
"""The length of a slope field's segments, as a fraction of the spacing."""

FADE_START: Final = 0.5
"""How far through an octave of zoom (as ``log2`` of the spacing) new nodes
start to fade in; they are fully in at the end of it."""

_MAX_INDEX: Final = 2.0**52
"""Beyond this many steps from the origin, nodes cannot be placed exactly."""


@dataclass(frozen=True, slots=True, eq=False)
class FieldLattice:
    """The nodes of a field's lattice that are (at least partly) visible.

    Attributes:
        x: World x coordinate of each node.
        y: World y coordinate of each node.
        px: Screen x of each node.
        py: Screen y of each node.
        fade: How far each node has faded in, in ``(0, 1]``: 1 for nodes of
            the coarser lattice, less for nodes still appearing.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    px: NDArray[np.float64]
    py: NDArray[np.float64]
    fade: NDArray[np.float64]


def check_spacing(spacing: float) -> float:
    """Validate a field's spacing.

    Args:
        spacing: Distance between neighbouring arrows in pixels.

    Returns:
        The spacing as a float.

    Raises:
        ValueError: If it is not a finite number of at least
            :data:`MIN_SPACING` pixels.
    """
    value = float(spacing)
    if not (math.isfinite(value) and value >= MIN_SPACING):
        msg = f"spacing must be at least {MIN_SPACING:g} pixels, got {spacing!r}"
        raise ValueError(msg)
    return value


def field_lattice(view: View, spacing: float, margin: float) -> FieldLattice | None:
    """Return the nodes of the lattice for a field drawn ``spacing`` px apart.

    Args:
        view: The view being drawn.
        spacing: The requested distance between neighbouring arrows, in
            pixels (see :func:`check_spacing`).
        margin: Pixels beyond the plane to cover, so that arrows centred
            just outside it still show their part inside.

    Returns:
        The nodes, or ``None`` if there are none (or the view is so far
        from the origin that nodes cannot be placed exactly).
    """
    if view.rect.is_empty:
        return None
    axes = [_axis_nodes(view, axis, spacing, margin) for axis in ("x", "y")]
    if axes[0] is None or axes[1] is None:
        return None
    (x, px, fade_x), (y, py, fade_y) = axes[0], axes[1]
    # Every combination of a column and a row, flattened.
    xx, yy = np.meshgrid(x, y)
    pxx, pyy = np.meshgrid(px, py)
    fx, fy = np.meshgrid(fade_x, fade_y)
    return FieldLattice(
        xx.ravel(), yy.ravel(), pxx.ravel(), pyy.ravel(), np.minimum(fx, fy).ravel()
    )


def lattice_step(scale: float, spacing: float) -> tuple[float, float]:
    """Return the lattice step along an axis, and how far its new nodes have faded in.

    Args:
        scale: Pixels per transformed unit along the axis.
        spacing: The requested spacing in pixels.

    Returns:
        ``(step, fade)``: the step, a power of two in transformed units with
        ``spacing / 2 <= step × scale < spacing``, and the fade of the nodes
        at odd multiples of it, in ``[0, 1)``: 0 for the first half of the
        octave (where ``log2(2 × step × scale / spacing)`` is below
        :data:`FADE_START`), then rising smoothly to 1.
    """
    exponent = math.ceil(math.log2(spacing / 2 / scale))
    step = math.ldexp(1.0, exponent)
    octave = math.log2(2 * step * scale / spacing)
    t = min(max((octave - FADE_START) / (1.0 - FADE_START), 0.0), 1.0)
    return step, t * t * (3.0 - 2.0 * t)  # smoothstep


def _axis_nodes(
    view: View, axis: Literal["x", "y"], spacing: float, margin: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the world coordinates, pixels and fades of one axis's nodes.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"``.
        spacing: The requested spacing in pixels.
        margin: Pixels beyond the plane to cover.

    Returns:
        ``(world, pixels, fade)`` for the visible multiples of the step, or
        ``None`` if there are none.
    """
    scale = view.axis_scale(axis)
    step, fade = lattice_step(scale, spacing)
    bounds = view.transformed_bounds
    low, high = (
        (bounds.x_min, bounds.x_max) if axis == "x" else (bounds.y_min, bounds.y_max)
    )
    first = math.ceil((low - margin / scale) / step)
    last = math.floor((high + margin / scale) / step)
    if last < first or max(abs(first), abs(last)) > _MAX_INDEX:
        return None
    indices = np.arange(first, last + 1, dtype=np.int64)
    fades = np.where(indices % 2 == 1, fade, 1.0)
    shown = fades > 0
    indices, fades = indices[shown], fades[shown]
    transform = view.axis_transform(axis)
    world = transform.inverse(indices.astype(np.float64) * step)
    pixels = view.multiples_to_screen(axis, step, indices)
    return world, pixels, fades


def screen_directions(
    view: View,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map world vectors at world points to screen directions.

    Args:
        view: The view being drawn.
        x: World x of each point.
        y: World y of each point.
        u: The x component of the vector at each point (world units).
        v: The y component, likewise.

    Returns:
        Unit vectors on screen, shape ``(N, 2)`` (y pointing down), NaN where
        the vector is undefined, zero or infinite, as it then has no
        direction.
    """
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        # Scale each vector down by its larger component first, so that huge
        # vectors (next to a pole) do not overflow on the way to a direction.
        size = np.maximum(np.abs(u), np.abs(v))
        usable = np.isfinite(size) & (size > 0)
        size = np.where(usable, size, 1.0)
        dx = u / size * view.transform_x.derivative(x) * view.scale_x
        dy = -v / size * view.transform_y.derivative(y) * view.scale_y
        larger = np.maximum(np.abs(dx), np.abs(dy))
        usable &= np.isfinite(larger) & (larger > 0)
        larger = np.where(usable, larger, 1.0)
        dx, dy = dx / larger, dy / larger
        length = np.hypot(dx, dy)
        return np.where(
            usable[:, None], np.column_stack((dx / length, dy / length)), np.nan
        )
