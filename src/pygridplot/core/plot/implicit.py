"""Finding implicit curves ``F(x, y) = c`` on screen.

An implicit curve has no parameter to walk along, so it is found from
samples of ``F`` on a grid with **marching squares**: wherever ``F - c`` has
opposite signs at the two ends of a grid edge, the curve crosses that edge.
:mod:`contourpy` (the contouring library behind matplotlib) does the
marching and links the crossings into polylines. This module decides
*where* to sample and checks what marching squares finds:

1. **A coarse grid** of :attr:`ImplicitSampling.coarse_factor` ×
   :attr:`ImplicitSampling.resolution_px` pixels (8 px by default) covers
   the visible area. Its nodes are multiples of the grid step, so panning
   reuses exactly the same samples and curves move rigidly.
2. **Candidate cells** are the coarse cells the curve could pass through:
   ``F - c`` changes sign at their corners, or ``F`` is undefined at some
   corners and defined at others (a domain edge), or it comes close to the
   level compared with how fast it varies nearby, which catches small loops
   and curves that dip into a cell and out again without a sign change at
   the corners. They are widened by one cell on every side.
3. **A fine grid** (:attr:`ImplicitSampling.resolution_px`, 2 px by
   default) is evaluated only inside the candidate cells, so the work is
   proportional to the curve's length, not the plane's area. Elsewhere the
   nearest coarse value stands in, which cannot create a contour. If the
   fine grid would overrun the evaluation budget, it is made coarser
   instead.
4. **Checking crossings.** A sign change is not always a root: ``F`` also
   changes sign across a pole (``y - tan x``) and across a jump
   (``y - floor x``), and marching squares would draw a false line there.
   Each vertex of a contour lies on a grid edge between two samples of
   opposite sign, so bisection along that edge tells the cases apart: near
   a true root ``|F - c|`` shrinks, at a pole it grows, and at a jump it
   stays. False vertices are removed (breaking the line), and true ones are
   moved onto the refined root. A vertex on a node where ``F`` equals the
   level exactly counts only if ``F - c`` takes both signs around it.

Several levels share all the work but the contouring and the checks, so a
contour map costs little more than one curve (see :func:`sample_implicit`).
The sampler works on a function of *transformed* coordinates, so it needs
no knowledge of the view or of the kind of axes, and returns the curves in
grid coordinates, which the caller maps to the screen precisely.

Limits: ``F - c`` must change sign across the curve, so a curve where ``F``
only touches the level (``F = g²``) is not drawn; write ``g`` instead.
Features smaller than the fine grid (2 px) may be missed or merged, and a
curve stops within one fine cell of where ``F`` becomes undefined.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import contourpy
import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot import lattice
from pygridplot.core.plot.lattice import TransformedFunction

_ROOT_SHRINK: Final = 0.6
"""A crossing is a true root if bisection shrinks ``max(|F|)`` at its two ends
below this fraction of the original (by ``refine_steps = 3`` halvings a
smooth root shrinks it about 8 times, even a cube root 2 times; a jump
leaves it unchanged and a pole makes it grow)."""

_EXTRA_STEPS: Final = 8
"""Further bisection steps for crossings still undecided after
``refine_steps``. A root close to one end of its edge, where ``F`` first
moves away from zero (an edge nearly tangent to the curve), leaves that end,
and so ``max(|F|)``, unchanged until the bracket reaches it."""

_WHOLE: Final = 1e-7
"""A contour vertex coordinate this close to a whole number lies on a grid line."""

_SUBDIVISIONS: Final = (16, 8, 4, 2, 1)
"""Allowed values of :attr:`ImplicitSampling.coarse_factor`."""


@dataclass(frozen=True, slots=True)
class ImplicitSampling:
    """How implicit curves are sampled. The defaults suit most curves.

    Attributes:
        resolution_px: Spacing of the fine grid in pixels. Features smaller
            than this may be missed; halving it quadruples the work near the
            curve.
        coarse_factor: The coarse grid, used to find where the curve can be,
            is this many fine cells across (1, 2, 4, 8 or 16; 1 evaluates
            the fine grid everywhere).
        refine_steps: Bisection steps spent checking and refining each
            crossing of a grid edge. 0 skips the check, so poles and jumps
            may be drawn as lines.
        max_samples: Most evaluations of ``F`` per call (per frame). When
            the grids would need more (a large window, or a dense curve),
            they are made coarser, and if even then too little is left for
            ``refine_steps`` steps per crossing, fewer are taken.
    """

    resolution_px: float = 2.0
    coarse_factor: int = 4
    refine_steps: int = 3
    max_samples: int = 250_000

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 0.5 <= self.resolution_px <= 16.0:
            msg = "resolution_px must be within [0.5, 16]"
            raise ValueError(msg)
        if self.coarse_factor not in _SUBDIVISIONS:
            msg = f"coarse_factor must be one of {sorted(_SUBDIVISIONS)}"
            raise ValueError(msg)
        if not 0 <= self.refine_steps <= 30:
            msg = "refine_steps must be within [0, 30]"
            raise ValueError(msg)
        if self.max_samples < 1000:
            msg = "max_samples must be at least 1000"
            raise ValueError(msg)


DEFAULT_IMPLICIT_SAMPLING: Final = ImplicitSampling()
"""The implicit curve sampling settings used when none are given."""


@dataclass(frozen=True, slots=True, eq=False)
class ImplicitContours:
    """The curves ``F = c`` as polylines in grid coordinates.

    A point ``(column, row)`` of a line lies at the transformed coordinates
    ``((origin[0] + column) × step[0], (origin[1] + row) × step[1])``:
    columns count along the x axis and rows along the y axis, both in fine
    grid steps.

    Attributes:
        lines: The polylines, each of shape ``(N, 2)`` with ``N >= 2``.
        origin: Grid indices ``(i, j)`` of the point ``(0, 0)``.
        step: Fine grid step ``(step_u, step_w)`` in transformed units.
        evaluations: How many times ``F`` was evaluated.
    """

    lines: list[NDArray[np.float64]]
    origin: tuple[int, int]
    step: tuple[float, float]
    evaluations: int


def sample_implicit(
    func: TransformedFunction,
    u_range: tuple[float, float],
    w_range: tuple[float, float],
    step: tuple[float, float],
    config: ImplicitSampling = DEFAULT_IMPLICIT_SAMPLING,
    levels: Sequence[float] = (0.0,),
) -> ImplicitContours | None:
    """Find the curves ``F(u, w) = c`` over a region of transformed coordinates.

    Several levels ``c`` share the work: ``F`` is evaluated once, on the
    cells any of the curves could pass through, and contoured at each level,
    so a contour map costs far less than one call per level.

    Args:
        func: ``F`` in transformed coordinates (see :data:`TransformedFunction`).
        u_range: ``(u_min, u_max)`` to cover.
        w_range: ``(w_min, w_max)`` to cover.
        step: The fine grid step ``(step_u, step_w)`` in transformed units,
            normally :attr:`ImplicitSampling.resolution_px` pixels.
        config: Sampling settings.
        levels: The values ``c`` (finite numbers).

    Returns:
        The curves, or ``None`` if none passes through the region (or the
        region is so large that even coarse cells ``lattice.MAX_FACTOR`` fine
        steps across would overrun the budget).

    Raises:
        ValueError: If a level is not a finite number.
    """
    level_array = np.asarray(levels, dtype=np.float64).ravel()
    if not np.isfinite(level_array).all():
        msg = "levels must be finite numbers"
        raise ValueError(msg)
    if level_array.size == 0:
        return None
    if not all(math.isfinite(v) for v in (*u_range, *w_range)) or min(step) <= 0:
        return None
    chosen = lattice.coarse_lattice(u_range, w_range, step, config)
    if chosen is None:
        return None
    factor, coarse_step, start, shape = chosen
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = lattice.Grid(func, start, coarse_step)
        coarse = grid.evaluate_nodes(np.arange(shape[1]), np.arange(shape[0]), 1)
        cells = np.zeros((shape[0] - 1, shape[1] - 1), dtype=bool)
        for level in np.unique(level_array):
            cells |= candidate_cells(coarse - level)
        if not cells.any():
            return None
        # Work only on the candidates' bounding box: contourpy's cost grows
        # with the size of the grid it is given.
        cells, coarse, (dc, dr) = lattice.crop(cells, coarse)
        grid = grid.shifted((start[0] + dc, start[1] + dr))
        subdivision = lattice.subdivision(cells, factor, grid.evaluations, config)
        values = grid.fine_values(coarse, cells, subdivision)
        points, starts, point_levels = _contour(values, level_array)
        if points.shape[0] == 0:
            return None
        # Spend what is left of the budget on checking crossings, up to the
        # configured number of steps, evenly over all the levels.
        remaining = lattice.remaining(grid, config)
        steps = min(config.refine_steps, remaining // points.shape[0])
        if steps > 0:
            points = _check_crossings(
                _Brackets(points, values, point_levels),
                grid,
                subdivision,
                steps,
                config.max_samples,
            )
    lines = _runs(points, starts)
    if not lines:
        return None
    a0, b0 = grid.start
    origin = (a0 * subdivision, b0 * subdivision)
    fine_step = (coarse_step[0] / subdivision, coarse_step[1] / subdivision)
    return ImplicitContours(lines, origin, fine_step, grid.evaluations)


def candidate_cells(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the grid cells that the curve ``F = 0`` could pass through.

    A cell is a candidate if ``F`` changes sign at its corners (or is zero at
    one), if some corners are defined and others are not (a domain edge
    crosses it), or if the smallest ``|F|`` at its corners is no more than
    how much ``F`` varies across a cell without a sign change in its
    neighbourhood (so ``F`` could reach zero inside it, as at a small loop).
    Candidates are then widened by one cell on every side, for curves that
    run close to cell borders.

    Args:
        values: ``F`` at the grid nodes, shape ``(rows, columns)``, NaN where
            undefined.

    Returns:
        A mask of shape ``(rows - 1, columns - 1)``.
    """
    a, b, c, d = values[:-1, :-1], values[:-1, 1:], values[1:, :-1], values[1:, 1:]
    defined = np.isfinite(np.stack((a, b, c, d)))
    some, every = defined.any(axis=0), defined.all(axis=0)
    # fmin and fmax skip undefined (NaN) corners.
    low = np.fmin(np.fmin(a, b), np.fmin(c, d))
    high = np.fmax(np.fmax(a, b), np.fmax(c, d))
    crossing = (low <= 0.0) & (high >= 0.0)
    nearest = np.fmin(np.fmin(np.abs(a), np.abs(b)), np.fmin(np.abs(c), np.abs(d)))
    # How much F varies across a cell nearby. Cells with a sign change are
    # candidates anyway and are left out: next to a pole they would make the
    # whole neighbourhood look as if F could reach zero anywhere.
    variation = lattice.widen_max(np.where(some & ~crossing, high - low, 0.0))
    near = nearest <= variation
    candidates = crossing | (some & ~every) | near
    return lattice.widen(candidates)


def _contour(
    values: NDArray[np.float64], levels: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64]]:
    """Run marching squares on a grid of values (NaN nodes are masked).

    Args:
        values: ``F`` at the fine nodes, shape ``(rows, columns)``.
        levels: The values of ``F`` to contour.

    Returns:
        The vertices of all the lines where ``F`` equals a level, in grid
        coordinates ``(column, row)``, shape ``(N, 2)``; the index at which
        each line starts; and each vertex's level. Every vertex lies on a
        grid edge: corner masking is off, so lines never cut across a cell's
        diagonal.
    """
    rows, columns = values.shape
    undefined = np.isnan(values)
    generator = contourpy.contour_generator(
        x=np.arange(columns, dtype=np.float64),
        y=np.arange(rows, dtype=np.float64),
        # Masking is slow, so only mask where F is genuinely undefined.
        z=np.ma.masked_array(values, undefined) if undefined.any() else values,
        corner_mask=False,
        line_type=contourpy.LineType.ChunkCombinedOffset,
    )
    all_points, all_starts, all_levels = [], [], []
    count = 0
    for level in levels:
        points_list, offsets_list = cast(
            tuple[list[NDArray[np.float64] | None], list[NDArray[np.intp] | None]],
            generator.lines(float(level)),
        )
        # One chunk, as no chunk size was asked for.
        points, offsets = points_list[0], offsets_list[0]
        if points is None or offsets is None or len(points) == 0:
            continue
        all_points.append(np.asarray(points, dtype=np.float64))
        all_starts.append(np.asarray(offsets[:-1], dtype=np.intp) + count)
        all_levels.append(np.full(len(points), level))
        count += len(points)
    if not all_points:
        return np.empty((0, 2)), np.empty(0, dtype=np.intp), np.empty(0)
    return (
        np.concatenate(all_points),
        np.concatenate(all_starts),
        np.concatenate(all_levels),
    )


def _check_crossings(
    brackets: _Brackets,
    grid: lattice.Grid,
    subdivision: int,
    steps: int,
    max_samples: int,
) -> NDArray[np.float64]:
    """Remove false crossings (poles, jumps) and refine the true ones.

    Every vertex lies on a grid edge whose two end nodes have values of
    opposite sign. That bracket is bisected ``steps`` times (see
    :class:`_Brackets`). Near a true root the larger ``|F|`` at the
    bracket's ends shrinks; at a pole it grows and at a jump it stays the
    same. True vertices move to the root interpolated in the final bracket;
    false ones become NaN, which breaks their line.

    Args:
        brackets: The contour vertices' brackets.
        grid: Evaluates ``F`` at fractional grid positions.
        subdivision: The fine grid's subdivision.
        steps: Bisection steps per vertex.
        max_samples: The evaluation budget, which limits the extra steps.

    Returns:
        The refined vertices, NaN where false.
    """
    for _ in range(steps):
        if not brackets.bisect(grid, subdivision):
            break
    # Crossings that look false may be roots near one end of their edge:
    # bisect those further, as far as the budget allows.
    brackets.active = brackets.active[~brackets.is_root()[brackets.active]]
    if brackets.active.size:
        budget = max(max_samples - grid.evaluations, 0)
        extra = min(_EXTRA_STEPS, budget // brackets.active.size)
        for _ in range(extra):
            if not brackets.bisect(grid, subdivision):
                break
    return brackets.roots()


class _Brackets:
    """The grid edge of each contour vertex, as a bracket for bisection.

    Each vertex lies on an edge from node ``a`` to node ``b``; positions along
    it are ``a + t (b - a)`` with ``t`` in ``[0, 1]``. The bracket
    ``[t_low, t_high]`` always has ``F - c`` of opposite signs at its ends,
    where ``c`` is the vertex's level; the values kept are of ``F - c``.

    Args:
        points: Contour vertices in grid coordinates, shape ``(N, 2)``.
        values: ``F`` at the fine nodes.
        levels: Each vertex's level.
    """

    def __init__(
        self,
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        levels: NDArray[np.float64],
    ) -> None:
        column, row = points[:, 0], points[:, 1]
        # contourpy interpolates along the edge, so the coordinate that stays
        # constant on it is whole up to round-off (208.99999999999997).
        near_column, near_row = np.rint(column), np.rint(row)
        whole_column = np.abs(column - near_column) < _WHOLE
        whole_row = np.abs(row - near_row) < _WHOLE
        # A vertex on a node (both whole) is where F is zero to round-off;
        # otherwise exactly one coordinate is whole, which says which edge.
        on_node = whole_column & whole_row
        on_column = whole_column & ~whole_row  # a vertical edge: constant column
        column = np.where(whole_column, near_column, column)
        row = np.where(whole_row, near_row, row)
        self.c0 = np.where(on_column, column, np.floor(column)).astype(np.intp)
        self.r0 = np.where(on_column, np.floor(row), row).astype(np.intp)
        c1 = np.minimum(self.c0 + ~on_column, values.shape[1] - 1)
        r1 = np.minimum(self.r0 + on_column, values.shape[0] - 1)
        self.dc = (c1 - self.c0).astype(np.float64)
        self.dr = (r1 - self.r0).astype(np.float64)
        self.level = levels
        self.start = values[self.r0, self.c0] - levels
        self.end = values[r1, c1] - levels
        degenerate = (self.dc == 0) & (self.dr == 0)
        at_start = (self.start == 0.0) | on_node
        at_end = (self.end == 0.0) & ~at_start
        self.exact = at_start | at_end | degenerate
        # A vertex on a node where F equals the level exactly is a crossing
        # only if F takes both signs around that node: otherwise F merely
        # touches the level there, as (x - y)² does along x = y when the
        # nodes happen to lie on it, and nothing should be drawn (as nothing
        # is when they do not).
        node_row = np.where(at_end, r1, self.r0)
        node_column = np.where(at_end, c1, self.c0)
        on_zero_node = at_start | at_end
        self.touching = np.zeros(points.shape[0], dtype=bool)
        self.touching[on_zero_node] = ~_changes_sign_around(
            values,
            node_row[on_zero_node],
            node_column[on_zero_node],
            levels[on_zero_node],
        )
        self.defined = np.isfinite(self.start) & np.isfinite(self.end)
        self.t_low = np.zeros(points.shape[0])
        self.t_high = np.ones(points.shape[0])
        self.f_low, self.f_high = self.start.copy(), self.end.copy()
        self.active = np.flatnonzero(~self.exact & self.defined)

    def bisect(self, grid: lattice.Grid, subdivision: int) -> bool:
        """Halve every active bracket once.

        Args:
            grid: Evaluates ``F`` at fractional grid positions.
            subdivision: The fine grid's subdivision.

        Returns:
            Whether any bracket was active.
        """
        active = self.active
        if active.size == 0:
            return False
        t_mid = 0.5 * (self.t_low[active] + self.t_high[active])
        f_mid = (
            grid.at(
                self.c0[active] + t_mid * self.dc[active],
                self.r0[active] + t_mid * self.dr[active],
                subdivision,
            )
            - self.level[active]
        )
        usable = np.isfinite(f_mid)
        self.defined[active[~usable]] = False  # undefined between the nodes
        low_side = usable & (np.sign(f_mid) == np.sign(self.f_low[active]))
        high_side = usable & ~low_side
        self.t_low[active[low_side]] = t_mid[low_side]
        self.f_low[active[low_side]] = f_mid[low_side]
        self.t_high[active[high_side]] = t_mid[high_side]
        self.f_high[active[high_side]] = f_mid[high_side]
        self.active = active[usable]
        return True

    def is_root(self) -> NDArray[np.bool_]:
        """Return which crossings are true roots, judged so far.

        Returns:
            A mask: exact zeros around which ``F`` changes sign, and
            brackets over which ``F`` stayed defined and whose ``max(|F|)``
            shrank (see ``_ROOT_SHRINK``).
        """
        before = np.maximum(np.abs(self.start), np.abs(self.end))
        after = np.maximum(np.abs(self.f_low), np.abs(self.f_high))
        shrunk = after < _ROOT_SHRINK * before
        exact = self.exact & ~self.touching
        return np.asarray(exact | (~self.exact & self.defined & shrunk), dtype=bool)

    def roots(self) -> NDArray[np.float64]:
        """Return the refined vertices, with NaN for false crossings.

        Returns:
            Vertices in grid coordinates, shape ``(N, 2)``.
        """
        root = self.is_root()
        gap = self.f_low - self.f_high
        safe = np.where(gap != 0.0, gap, 1.0)
        fraction = np.where(gap != 0.0, self.f_low / safe, 0.5)
        t = self.t_low + (self.t_high - self.t_low) * fraction
        t = np.where(self.exact, np.where(self.end == 0.0, 1.0, 0.0), t)
        refined = np.column_stack((self.c0 + t * self.dc, self.r0 + t * self.dr))
        refined[~root] = np.nan
        return refined


def _changes_sign_around(
    values: NDArray[np.float64],
    rows: NDArray[np.intp],
    columns: NDArray[np.intp],
    levels: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return whether ``F - level`` takes both signs among each node's neighbours.

    The eight neighbours count, so that a node where two zero lines cross
    (the origin, for ``x·y``) still sees both signs on its diagonals.
    Undefined neighbours and those outside the grid are ignored.

    Args:
        values: ``F`` at the fine nodes.
        rows: Row of each node.
        columns: Column of each node.
        levels: The level for each node.

    Returns:
        A mask, one entry per node.
    """
    padded = np.pad(values, 1, constant_values=np.nan)
    positive = np.zeros(rows.shape[0], dtype=bool)
    negative = np.zeros(rows.shape[0], dtype=bool)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == dc == 1:
                continue  # the node itself
            neighbour = padded[rows + dr, columns + dc] - levels
            positive |= neighbour > 0.0
            negative |= neighbour < 0.0
    return positive & negative


def _runs(
    points: NDArray[np.float64], starts: NDArray[np.intp]
) -> list[NDArray[np.float64]]:
    """Cut contour vertices into polylines: at line starts and NaN vertices.

    Done on the whole array at once, as a busy curve has many thousands of
    lines.

    Args:
        points: Vertices of all lines, shape ``(N, 2)``, NaN where removed.
        starts: The index at which each line starts.

    Returns:
        The runs of defined vertices within a line, with at least two
        points each.
    """
    n = points.shape[0]
    good = ~np.isnan(points[:, 0])
    first = np.zeros(n, dtype=bool)
    first[starts] = True
    previous_good = np.concatenate(([False], good[:-1])) & ~first
    next_good = np.concatenate((good[1:], [False]))
    next_good[starts[1:] - 1] = False
    begin = np.flatnonzero(good & ~previous_good)
    end = np.flatnonzero(good & ~next_good) + 1
    return [points[a:b] for a, b in zip(begin, end, strict=True) if b - a >= 2]
