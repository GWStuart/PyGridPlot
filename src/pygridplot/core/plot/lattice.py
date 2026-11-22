"""Anchored sampling grids: the lattice shared by implicit curves and regions.

Implicit curves (:mod:`~pygridplot.core.plot.implicit`) and regions
(:mod:`~pygridplot.core.plot.regions`) are both found from samples of a
function of two variables on a grid, in two passes:

1. **A coarse grid** covers the visible area. Its nodes are multiples of the
   coarse step, so panning reuses exactly the same samples and the result
   moves rigidly (:func:`coarse_lattice`).
2. **A fine grid** subdivides only the *candidate* cells (those the curve,
   or the region's boundary, can pass through), within the evaluation
   budget (:func:`subdivision`); elsewhere the nearest coarse value stands
   in (:meth:`Grid.fine_values`).

What counts as a candidate, and what is done with the fine values
(contour lines, or filled regions), is up to each sampler; this module
evaluates the grids and counts the evaluations against the budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

TransformedFunction: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
]
"""A function of transformed coordinates ``(u, w)``, never raising.

Undefined values are NaN (or infinite).
"""

VERTICES_PER_CELL: Final = 0.5
"""Boundary vertices per candidate cell per fine step across it, for planning
the budget: measured at 0.06 (``y - floor x``) to 0.43 (a contour map of
nine levels). If there are more, fewer bisection steps are taken."""


class LatticeSettings(Protocol):
    """What the lattice needs to know (:class:`~pygridplot.ImplicitSampling` has it)."""

    @property
    def coarse_factor(self) -> int:
        """Fine cells per coarse cell (a power of two)."""
        ...

    @property
    def refine_steps(self) -> int:
        """Bisection steps per boundary vertex."""
        ...

    @property
    def max_samples(self) -> int:
        """Most evaluations per call."""
        ...


DENSE_FACTOR: Final = 16
"""The fine grid's array has at most this many times ``max_samples`` nodes
(most of them not evaluated), which bounds memory and contouring time."""


MAX_FACTOR: Final = 1024
"""The largest coarse cell, in fine steps, before a region counts as too big."""


def coarse_lattice(
    u_range: tuple[float, float],
    w_range: tuple[float, float],
    step: tuple[float, float],
    config: LatticeSettings,
) -> tuple[int, tuple[float, float], tuple[int, int], tuple[int, int]] | None:
    """Choose the coarse grid that covers a region, with one cell to spare.

    The coarse grid normally takes a quarter of the budget at most. On a huge
    region (or with a very fine setting) its cells are made larger, by powers
    of two, so that the lattices stay nested and panning stays rigid.

    Args:
        u_range: ``(u_min, u_max)`` to cover.
        w_range: ``(w_min, w_max)`` to cover.
        step: The fine grid step.
        config: Sampling settings.

    Returns:
        ``(factor, coarse_step, start, shape)``: fine steps per coarse step,
        the coarse step, the coarse indices ``(a0, b0)`` of the first node,
        and the number of nodes as ``(rows, columns)``; or ``None`` if no
        coarse grid up to ``MAX_FACTOR`` fits.
    """
    factor = config.coarse_factor
    while True:
        coarse_step = (step[0] * factor, step[1] * factor)
        a0 = math.floor(u_range[0] / coarse_step[0]) - 1
        a1 = math.ceil(u_range[1] / coarse_step[0]) + 1
        b0 = math.floor(w_range[0] / coarse_step[1]) - 1
        b1 = math.ceil(w_range[1] / coarse_step[1]) + 1
        shape = (b1 - b0 + 1, a1 - a0 + 1)
        if shape[0] * shape[1] <= config.max_samples // 4:
            return factor, coarse_step, (a0, b0), shape
        if factor >= MAX_FACTOR:
            return None
        factor *= 2


def crop(
    cells: NDArray[np.bool_], coarse: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.float64], tuple[int, int]]:
    """Cut the cells and coarse nodes down to the candidates' bounding box.

    Args:
        cells: Candidate cells (at least one), shape ``(rows, columns)``.
        coarse: Values at the coarse nodes, shape ``(rows + 1, columns + 1)``.

    Returns:
        The cropped cells and nodes, and the ``(column, row)`` of the box's
        first cell.
    """
    rows, columns = np.nonzero(cells)
    r0, r1 = int(np.min(rows)), int(np.max(rows)) + 1
    c0, c1 = int(np.min(columns)), int(np.max(columns)) + 1
    return cells[r0:r1, c0:c1], coarse[r0 : r1 + 1, c0 : c1 + 1], (c0, r0)


class Grid:
    """Evaluates ``F`` on grids of nodes that are multiples of a step.

    Node ``(i, j)`` of a grid subdivided ``s`` times lies at the transformed
    coordinates ``((a0 × s + i) × step_u / s, (b0 × s + j) × step_w / s)``,
    where ``(a0, b0)`` is the coarse grid's first node.

    Args:
        func: ``F`` in transformed coordinates.
        start: Coarse indices ``(a0, b0)`` of the first node.
        coarse_step: The coarse step ``(step_u, step_w)``.

    Attributes:
        start: Coarse indices ``(a0, b0)`` of the first node.
        evaluations: How many times ``F`` has been evaluated, which the
            budget is checked against.
    """

    def __init__(
        self,
        func: TransformedFunction,
        start: tuple[int, int],
        coarse_step: tuple[float, float],
    ) -> None:
        self._func = func
        self.start = start
        self._step = coarse_step
        self.evaluations = 0

    def shifted(self, start: tuple[int, int]) -> Grid:
        """Return the same grid with another first node, keeping the count.

        Args:
            start: Coarse indices ``(a0, b0)`` of the new first node.

        Returns:
            The shifted grid.
        """
        grid = Grid(self._func, start, self._step)
        grid.evaluations = self.evaluations
        return grid

    def at(
        self, columns: NDArray[np.float64], rows: NDArray[np.float64], subdivision: int
    ) -> NDArray[np.float64]:
        """Evaluate ``F`` at (fractional) positions of a subdivided grid.

        Args:
            columns: Column positions, in fine steps from the first node.
            rows: Row positions, likewise.
            subdivision: Fine steps per coarse step.

        Returns:
            ``F`` at each position (NaN where undefined or infinite).
        """
        a0, b0 = self.start
        su, sw = self._step[0] / subdivision, self._step[1] / subdivision
        u = (a0 * subdivision + columns) * su
        w = (b0 * subdivision + rows) * sw
        self.evaluations += int(u.shape[0])
        values = np.asarray(self._func(u, w), dtype=np.float64)
        return np.where(np.isfinite(values), values, np.nan)

    def evaluate_nodes(
        self, columns: NDArray[np.int64], rows: NDArray[np.int64], subdivision: int
    ) -> NDArray[np.float64]:
        """Evaluate ``F`` at every node of a rectangle of the grid.

        Args:
            columns: Column indices.
            rows: Row indices.
            subdivision: Fine steps per coarse step.

        Returns:
            The values, shape ``(len(rows), len(columns))``.
        """
        cc, rr = np.meshgrid(columns.astype(np.float64), rows.astype(np.float64))
        return self.at(cc.ravel(), rr.ravel(), subdivision).reshape(cc.shape)

    def fine_values(
        self,
        coarse: NDArray[np.float64],
        cells: NDArray[np.bool_],
        subdivision: int,
    ) -> NDArray[np.float64]:
        """Evaluate the fine grid inside the candidate cells.

        Nodes outside the candidate cells are not evaluated: they take the
        value of the nearest coarse node. A cell that is not a candidate has
        corners of one sign, and neighbouring cells share corners, so these
        stand-in values can never create a contour; and unlike masking them,
        they leave contourpy a plain array, which it contours much faster.

        Args:
            coarse: Values at the coarse nodes, shape ``(NY + 1, NX + 1)``.
            cells: Candidate coarse cells, shape ``(NY, NX)``.
            subdivision: Fine steps per coarse step.

        Returns:
            Values at the fine nodes, shape ``(NY·s + 1, NX·s + 1)``.
        """
        s = subdivision
        if s == 1:
            return coarse
        ny, nx = cells.shape[0] * s + 1, cells.shape[1] * s + 1
        nearest_row = (np.arange(ny) + s // 2) // s
        nearest_column = (np.arange(nx) + s // 2) // s
        values = coarse[np.ix_(nearest_row, nearest_column)]
        need = node_mask(np.kron(cells, np.ones((s, s), dtype=bool)))
        need[::s, ::s] = False  # the coarse values are already exact
        rows, columns = np.nonzero(need)
        values[rows, columns] = self.at(
            columns.astype(np.float64), rows.astype(np.float64), s
        )
        return values


def widen_max(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the maximum over each cell and its eight neighbours.

    Args:
        values: A 2-D array (NaN-free).

    Returns:
        The 3×3 neighbourhood maximum, same shape.
    """
    padded = np.pad(values, 1, constant_values=-np.inf)
    rows, columns = values.shape
    result: NDArray[np.float64] = values.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.maximum(result, padded[dy : dy + rows, dx : dx + columns], out=result)
    return result


def widen(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Grow a mask by one cell in every direction (8-connected).

    Args:
        mask: A 2-D boolean array.

    Returns:
        The grown mask, same shape.
    """
    return np.asarray(widen_max(mask.astype(np.float64)) > 0.0, dtype=bool)


def node_mask(cells: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the nodes that are a corner of at least one of the cells.

    Args:
        cells: A cell mask, shape ``(rows, columns)``.

    Returns:
        A node mask, shape ``(rows + 1, columns + 1)``.
    """
    nodes = np.zeros((cells.shape[0] + 1, cells.shape[1] + 1), dtype=bool)
    nodes[:-1, :-1] |= cells
    nodes[:-1, 1:] |= cells
    nodes[1:, :-1] |= cells
    nodes[1:, 1:] |= cells
    return nodes


def subdivision(
    cells: NDArray[np.bool_], factor: int, used: int, config: LatticeSettings
) -> int:
    """Choose how finely to subdivide the candidate cells within the budget.

    Subdividing the candidate cells ``s`` times costs ``s - 1`` new
    evaluations along each of their edges and ``(s - 1)²`` inside each of
    them, counted exactly; checking the crossings then costs about
    ``VERTICES_PER_CELL × s × refine_steps`` per cell (a curve through a
    cell crosses about ``s`` fine cells, but many candidate cells are only
    neighbours of the curve). The fine grid's array, mostly not evaluated,
    is also limited to ``DENSE_FACTOR × max_samples`` nodes, which bounds
    memory and contouring time.

    Args:
        cells: Candidate coarse cells.
        factor: The preferred subdivision (fine steps per coarse step), a
            power of two.
        used: Evaluations already spent.
        config: The sampling settings (the budget and the steps).

    Returns:
        The largest power of two, at most ``factor``, that fits; 1 if none
        does.
    """
    count = int(cells.sum())
    rows, columns = cells.shape
    # The edges of the candidate cells, each counted once.
    across = np.zeros((rows + 1, columns), dtype=bool)
    across[:-1] |= cells
    across[1:] |= cells
    down = np.zeros((rows, columns + 1), dtype=bool)
    down[:, :-1] |= cells
    down[:, 1:] |= cells
    edges = int(across.sum()) + int(down.sum())
    budget = config.max_samples - used

    def cost(s: int) -> float:
        nodes = edges * (s - 1) + count * (s - 1) ** 2
        return nodes + VERTICES_PER_CELL * count * s * config.refine_steps

    s = factor
    while s > 1 and (
        cost(s) > budget
        or (rows * s + 1) * (columns * s + 1) > DENSE_FACTOR * config.max_samples
    ):
        s //= 2
    return s


def remaining(grid: Grid, config: LatticeSettings) -> int:
    """Return how many evaluations are left in the budget.

    Args:
        grid: The grid, which counts its evaluations.
        config: The sampling settings.

    Returns:
        The evaluations left, at least 0.
    """
    return max(config.max_samples - grid.evaluations, 0)
