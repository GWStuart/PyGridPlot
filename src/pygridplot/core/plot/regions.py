"""Finding regions on screen: where a condition ``inside(x, y)`` holds.

A region is the set of points where a condition is true: an inequality
such as ``x**2 + y**2 < 4`` or ``y < sin(x)``, or several combined with
NumPy's ``&``, ``|`` and ``~``. It is found on the same anchored grids as
implicit curves (:mod:`~pygridplot.core.plot.lattice`), so the work is
proportional to the length of its boundary, not to its area, and panning
moves it rigidly:

1. **A coarse grid** (8 px by default) covers the visible area; each node
   is inside (the condition holds and is defined) or outside.
2. **Candidate cells** are the coarse cells whose corners disagree: the
   boundary passes through them. They are widened by one cell on every
   side.
3. **A fine grid** (2 px by default) is evaluated only inside the candidate
   cells; elsewhere the nearest coarse node's answer stands in, which is
   right, as a cell that is not a candidate is inside or outside as a whole.
4. **Filled contours.** :mod:`contourpy` turns the grid of answers into
   polygons with holes (the region between the levels 0.5 and 1.5 of a
   grid of 0s and 1s), closed along the grid's border, which lies beyond
   the plane.
5. **Refining the boundary.** Each boundary vertex lies on a grid edge
   between an inside node and an outside one. Bisection along that edge
   (:attr:`~pygridplot.ImplicitSampling.refine_steps` halvings) moves it to
   within a fraction of a pixel of the true boundary, so edges are smooth
   once antialiased, not a staircase of grid cells.

Conditions only say *inside* or *outside*, so the boundary needs no sign
changes and no checks for poles: where ``y < tan(x)`` jumps from all to
nothing at an asymptote, the region really does end there. An undefined
condition (NaN, an error) counts as outside.

Limits: an island or a hole smaller than a coarse cell (8 px), which no
coarse node falls in, may be missed, and so may a band thinner than that.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import cast

import contourpy
import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot import lattice
from pygridplot.core.plot.implicit import DEFAULT_IMPLICIT_SAMPLING, ImplicitSampling
from pygridplot.core.plot.lattice import TransformedFunction


@dataclass(frozen=True, slots=True, eq=False)
class RegionRings:
    """A region as closed rings in grid coordinates, filled with the even-odd rule.

    A point ``(column, row)`` of a ring lies at the transformed coordinates
    ``((origin[0] + column) × step[0], (origin[1] + row) × step[1])``:
    columns count along the x axis and rows along the y axis, both in fine
    grid steps.

    Attributes:
        rings: The rings (outlines and holes), each of shape ``(N, 2)``
            with ``N >= 3``.
        origin: Grid indices ``(i, j)`` of the point ``(0, 0)``.
        step: Fine grid step ``(step_u, step_w)`` in transformed units.
        evaluations: How many times the condition was evaluated.
    """

    rings: list[NDArray[np.float64]]
    origin: tuple[int, int]
    step: tuple[float, float]
    evaluations: int


def sample_region(
    func: TransformedFunction,
    u_range: tuple[float, float],
    w_range: tuple[float, float],
    step: tuple[float, float],
    config: ImplicitSampling = DEFAULT_IMPLICIT_SAMPLING,
) -> RegionRings | None:
    """Find where ``func(u, w)`` holds over a region of transformed coordinates.

    Args:
        func: The condition in transformed coordinates: a value that is not
            zero (``True``) inside, zero (``False``) outside, NaN where
            undefined (see :data:`~pygridplot.core.plot.lattice.TransformedFunction`).
        u_range: ``(u_min, u_max)`` to cover.
        w_range: ``(w_min, w_max)`` to cover.
        step: The fine grid step ``(step_u, step_w)`` in transformed units,
            normally :attr:`ImplicitSampling.resolution_px` pixels.
        config: Sampling settings.

    Returns:
        The region, or ``None`` if none of it is in the area (or the area
        is so large that even the coarsest grid would overrun the budget).
    """
    if not all(math.isfinite(v) for v in (*u_range, *w_range)) or min(step) <= 0:
        return None
    chosen = lattice.coarse_lattice(u_range, w_range, step, config)
    if chosen is None:
        return None
    factor, coarse_step, start, shape = chosen
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = lattice.Grid(_inside(func), start, coarse_step)
        coarse = grid.evaluate_nodes(np.arange(shape[1]), np.arange(shape[0]), 1)
        if not coarse.any():
            return None
        cells = boundary_cells(coarse)
        if not cells.any():  # inside everywhere: one rectangle
            ring = _rectangle(0, 0, shape[1] - 1, shape[0] - 1)
            return RegionRings([ring], start, coarse_step, grid.evaluations)
        # Contour only the boundary's bounding box: contourpy's cost grows with
        # the size of the grid it is given. Around the box, each strip of
        # coarse cells is wholly inside or outside (see _frame).
        box_cells, box_coarse, (dc, dr) = lattice.crop(cells, coarse)
        box = grid.shifted((start[0] + dc, start[1] + dr))
        subdivision = lattice.subdivision(box_cells, factor, box.evaluations, config)
        values = box.fine_values(box_coarse, box_cells, subdivision)
        rings = _refine(_filled(values), values, box, subdivision, config)
        offset = np.array([dc * subdivision, dr * subdivision], dtype=np.float64)
        rings = [ring + offset for ring in rings]
        rings += _frame(coarse, (dc, dr), box_cells.shape, subdivision)
    if not rings:
        return None
    origin = (start[0] * subdivision, start[1] * subdivision)
    fine_step = (coarse_step[0] / subdivision, coarse_step[1] / subdivision)
    return RegionRings(rings, origin, fine_step, box.evaluations)


def _frame(
    coarse: NDArray[np.float64],
    corner: tuple[int, int],
    box_shape: tuple[int, int],
    subdivision: int,
) -> list[NDArray[np.float64]]:
    """Return rectangles for the parts of the region around the boundary's box.

    Outside the box of candidate cells, no cell has corners that disagree,
    so each of the four strips around the box (below, above, left and
    right of it) is inside or outside as a whole. The inside ones are
    added as rectangles; where they meet the box's rings, their shared
    edges cancel under the even-odd rule.

    Args:
        coarse: 1.0 inside, 0.0 outside, at the coarse nodes of the whole
            grid, shape ``(rows + 1, columns + 1)``.
        corner: ``(column, row)`` of the box's first cell.
        box_shape: ``(rows, columns)`` of the box, in cells.
        subdivision: Fine steps per coarse step.

    Returns:
        The rectangles, in fine grid coordinates of the whole grid.
    """
    rows, columns = coarse.shape[0] - 1, coarse.shape[1] - 1
    c0, r0 = corner
    r1, c1 = r0 + box_shape[0], c0 + box_shape[1]
    strips = (
        (0, 0, columns, r0, (0, 0)),  # below the box, full width
        (0, r1, columns, rows, (rows, 0)),  # above it
        (0, r0, c0, r1, (r0, 0)),  # left of it
        (c1, r0, columns, r1, (r0, columns)),  # right of it
    )
    s = subdivision
    return [
        _rectangle(x0 * s, y0 * s, x1 * s, y1 * s)
        for x0, y0, x1, y1, (node_row, node_column) in strips
        if x1 > x0 and y1 > y0 and coarse[node_row, node_column] > 0.5
    ]


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> NDArray[np.float64]:
    """Return the ring of a rectangle with corners ``(x0, y0)`` and ``(x1, y1)``."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def boundary_cells(inside: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the grid cells the region's boundary could pass through.

    A cell is a candidate if its corners disagree (some inside, some
    outside). Candidates are then widened by one cell on every side, for
    boundaries that run close to cell borders.

    Args:
        inside: 1.0 at the nodes inside the region, 0.0 elsewhere, shape
            ``(rows, columns)``.

    Returns:
        A mask of shape ``(rows - 1, columns - 1)``.
    """
    a, b, c, d = inside[:-1, :-1], inside[:-1, 1:], inside[1:, :-1], inside[1:, 1:]
    total = a + b + c + d
    return lattice.widen((total > 0) & (total < 4))


def _inside(func: TransformedFunction) -> TransformedFunction:
    """Turn a condition into 1.0 inside and 0.0 outside (undefined: outside)."""

    def inside(u: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(func(u, w), dtype=np.float64)
        return np.where(np.isfinite(values) & (values != 0.0), 1.0, 0.0)

    return inside


def _filled(values: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Return the rings around the nodes with value 1 (outlines and holes).

    Args:
        values: 1.0 inside, 0.0 outside, at the fine nodes.

    Returns:
        The rings in grid coordinates ``(column, row)``. Every vertex lies on
        a grid edge between an inside and an outside node (halfway along),
        or on a node of the grid's border.
    """
    generator = contourpy.contour_generator(
        z=values, name="serial", fill_type=contourpy.FillType.OuterOffset
    )
    points_list, offsets_list = cast(
        tuple[list[NDArray[np.float64]], list[NDArray[np.intp]]],
        generator.filled(0.5, 1.5),
    )
    rings = []
    for points, offsets in zip(points_list, offsets_list, strict=True):
        for start, end in itertools.pairwise(offsets):
            if end - start >= 3:
                rings.append(np.asarray(points[start:end], dtype=np.float64))
    return rings


def _refine(
    rings: list[NDArray[np.float64]],
    values: NDArray[np.float64],
    grid: lattice.Grid,
    subdivision: int,
    config: ImplicitSampling,
) -> list[NDArray[np.float64]]:
    """Move the boundary vertices onto the true boundary, by bisection.

    A vertex halfway along a grid edge has an inside node at one end and an
    outside node at the other. Each step evaluates the condition at the
    middle of the bracket and keeps the half that still has one end of each
    kind; the vertex ends in the middle of the last bracket. All vertices
    are refined together, one evaluation of the condition per step. (A
    ring's closing vertex repeats its first, and is refined to the same
    place.)

    Args:
        rings: The rings in grid coordinates.
        values: 1.0 inside, 0.0 outside, at the fine nodes.
        grid: The grid (to evaluate the condition, and count).
        subdivision: Fine steps per coarse step.
        config: Sampling settings (the steps and the budget).

    Returns:
        The rings with their boundary vertices moved.
    """
    if not rings:
        return rings
    points = np.concatenate(rings)
    columns, rows = points[:, 0], points[:, 1]
    across = columns != np.floor(columns)  # on an edge along a row
    down = rows != np.floor(rows)  # on an edge along a column
    on_edge = np.flatnonzero(across ^ down)
    if on_edge.size == 0:
        return rings
    steps = min(config.refine_steps, lattice.remaining(grid, config) // on_edge.size)
    if steps <= 0:
        return rings
    start = np.floor(points[on_edge])
    end = start + np.column_stack((across[on_edge], down[on_edge]))
    first_inside = (
        values[start[:, 1].astype(np.intp), start[:, 0].astype(np.intp)] > 0.5
    )
    inner = np.where(first_inside[:, None], start, end)
    outer = np.where(first_inside[:, None], end, start)
    for _ in range(steps):
        middle = (inner + outer) / 2
        holds = grid.at(middle[:, 0], middle[:, 1], subdivision) > 0.5
        inner = np.where(holds[:, None], middle, inner)
        outer = np.where(holds[:, None], outer, middle)
    moved = points.copy()
    moved[on_edge] = (inner + outer) / 2
    bounds = np.cumsum([len(ring) for ring in rings])[:-1]
    return list(np.split(moved, bounds))
