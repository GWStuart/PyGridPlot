"""Where a draggable point may go: free, along one axis, or along a curve.

A constraint turns a target on screen (the cursor, less the offset at which
the point was grabbed) into the point's new position:

* :class:`AxisConstraint` moves the point to the target in both coordinates
  (a free point), or in one only (``lock="x"`` keeps x and moves y,
  ``lock="y"`` keeps y and moves x), and keeps it within optional bounds.
  It works in *transformed* coordinates, so a point dragged on a log axis
  follows the cursor exactly and can never reach zero or below.
* :class:`CurveConstraint` keeps the point on a parametric curve
  ``(x(t), y(t))``: the curve is sampled on screen, and a dragged point
  goes to the point of the curve at the cursor's x if the curve is a graph
  (``(t, f(t))``, anything running steadily left to right), or else follows
  the curve from where it is towards the cursor (see
  :mod:`pygridplot.core.handles.tracking`). The point is then placed
  exactly at ``(x(t), y(t))``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np

from pygridplot.core.coords import ScreenRect
from pygridplot.core.handles.tracking import (
    along_axis,
    graph_axis,
    nearest_on_curve,
    segment_span,
)
from pygridplot.core.plot.builders import (
    CurveComponent,
    WorldCurve,
    check_range,
    world_curve,
)
from pygridplot.core.plot.curves import (
    DEFAULT_CURVE_SAMPLING,
    CurvePiece,
    CurveSampling,
    sample_curve_pieces,
)
from pygridplot.core.view import View

Lock: TypeAlias = Literal["x", "y"]
"""Which coordinate of a draggable point stays fixed while it is dragged."""

Range: TypeAlias = tuple[float, float]
"""An inclusive ``(low, high)`` range."""

SWITCH_PX: Final = 10.0
"""How much closer to the cursor, in pixels, another part of a curve must be
before a point being dragged along the curve jumps to it."""

CLOSED_PX: Final = 0.5
"""A curve whose two ends are this close on screen is closed: dragging past
one end continues from the other."""

_REFINE_SAMPLES: Final = 33
"""Parameter values tried per round when refining a point on a curve."""

_REFINE_ROUNDS: Final = 2
"""Rounds of refinement, each narrowing the search 16 times."""

_SAMPLE_MARGIN: Final = 0.5
"""The curve is sampled over the plane grown by this fraction of its size
on every side, so a point near the edge sees the curve just beyond it."""


@dataclass(frozen=True, slots=True)
class Placement:
    """A position of a draggable point.

    Attributes:
        x: World x.
        y: World y.
        t: The curve parameter, for points on a curve (otherwise ``None``).
    """

    x: float
    y: float
    t: float | None = None


def check_bounds(name: str, span: Range | None) -> Range | None:
    """Validate an optional coordinate range.

    Args:
        name: Parameter name for error messages.
        span: ``(low, high)`` (either may be infinite), or ``None``.

    Returns:
        The range as floats, or ``None``.

    Raises:
        ValueError: If an end is NaN or ``low > high``.
    """
    if span is None:
        return None
    low, high = float(span[0]), float(span[1])
    if math.isnan(low) or math.isnan(high) or low > high:
        msg = f"{name} must be a (low, high) pair with low <= high, got {span!r}"
        raise ValueError(msg)
    return (low, high)


@dataclass(frozen=True, slots=True)
class AxisConstraint:
    """A free point, or one that moves along only one axis.

    Attributes:
        lock: ``None`` for a free point, ``"x"`` to keep x fixed (the point
            moves vertically), ``"y"`` to keep y fixed.
        x_range: Bounds for x, or ``None``.
        y_range: Bounds for y, or ``None``.
    """

    lock: Lock | None = None
    x_range: Range | None = None
    y_range: Range | None = None

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.lock not in (None, "x", "y"):
            msg = f"lock must be None, 'x' or 'y', got {self.lock!r}"
            raise ValueError(msg)

    def place(self, view: View, x: float, y: float, current: Placement) -> Placement:
        """Return the allowed position nearest ``(x, y)``.

        Each coordinate is clamped to its bounds. A coordinate that is not a
        finite number, or that a log axis cannot show (zero or below), keeps
        its current value.

        Args:
            view: The view (for its kinds of axes).
            x: Wanted world x.
            y: Wanted world y.
            current: The position now.

        Returns:
            The new position.
        """
        return Placement(
            _allowed(x, self.x_range, view.log_x, current.x),
            _allowed(y, self.y_range, view.log_y, current.y),
        )

    def drag(
        self, view: View, target: tuple[float, float], current: Placement
    ) -> Placement:
        """Return the position for a drag towards a screen point.

        Args:
            view: The view the point is dragged in.
            target: The wanted screen position.
            current: The position now.

        Returns:
            The new position; the locked coordinate is unchanged.
        """
        u, w = view.screen_to_transformed(*target)
        with np.errstate(all="ignore"):
            x = float(view.transform_x.inverse(u))
            y = float(view.transform_y.inverse(w))
        if self.lock == "x":
            x = current.x
        elif self.lock == "y":
            y = current.y
        return self.place(view, x, y, current)


def _allowed(value: float, bounds: Range | None, log: bool, current: float) -> float:
    """Clamp one coordinate, keeping the current value if the result is unusable.

    Args:
        value: The wanted value.
        bounds: The allowed range, or ``None``.
        log: Whether the axis is logarithmic.
        current: The value now.

    Returns:
        The value to use.
    """
    if not math.isfinite(value):
        return current
    if bounds is not None:
        value = min(max(value, bounds[0]), bounds[1])
    if log and not value > 0.0:
        return current
    return value


class CurveConstraint:
    """A point that stays on the parametric curve ``(x(t), y(t))``.

    The curve's samples on screen are cached until the view changes, the
    frame ends (:meth:`invalidate`, in case the functions depend on time)
    or the constraint is replaced.

    Args:
        x: The x coordinate as a function of ``t`` (or a constant).
        y: The y coordinate as a function of ``t`` (or a constant).
        t_range: The parameter values the point may take.
        sampling: How the curve is sampled on screen.

    Raises:
        TypeError: If a component is neither callable nor a number.
        ValueError: If ``t_range`` is not an increasing pair of finite numbers.
    """

    def __init__(
        self,
        x: CurveComponent,
        y: CurveComponent,
        t_range: tuple[float, float],
        sampling: CurveSampling = DEFAULT_CURVE_SAMPLING,
    ) -> None:
        self.components = (x, y)
        self.t_range = check_range("t_range", t_range)
        self._world: WorldCurve = world_curve(x, y)
        self._sampling = sampling
        self._cache_key: object = None
        self._cache: tuple[list[CurvePiece], bool, Literal["x", "y"] | None] = (
            [],
            False,
            None,
        )

    def at(self, t: float) -> Placement:
        """Return the point of the curve at parameter ``t`` (clamped to the range).

        Args:
            t: The parameter.

        Returns:
            The position; its coordinates are NaN where the curve is undefined.
        """
        low, high = self.t_range
        t = min(max(float(t), low), high)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, ys = self._world(np.array([t]))
        x, y = float(xs[0]), float(ys[0])
        if not (math.isfinite(x) and math.isfinite(y)):
            x = y = math.nan
        return Placement(x, y, t)

    def drag(
        self, view: View, target: tuple[float, float], current: Placement
    ) -> Placement | None:
        """Return the position for a drag towards a screen point.

        Args:
            view: The view the point is dragged in.
            target: The wanted screen position.
            current: The position now (its ``t`` is followed along the curve).

        Returns:
            The new position, or ``None`` if no part of the curve is defined.
        """
        return self._track(view, target, current.t)

    def nearest(self, view: View, x: float, y: float) -> Placement | None:
        """Return the point of the curve nearest a world point, as seen on screen.

        Args:
            view: The view to measure distances in.
            x: World x.
            y: World y.

        Returns:
            The position, or ``None`` if no part of the curve is defined (or
            the point cannot be shown).
        """
        with np.errstate(all="ignore"):
            px, py = view.to_screen(x, y)
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return self._track(view, (px, py), None)

    def invalidate(self) -> None:
        """Forget the cached samples, so the curve is sampled again when needed."""
        self._cache_key = None

    def _track(
        self, view: View, target: tuple[float, float], current_t: float | None
    ) -> Placement | None:
        """Follow the curve towards a target.

        A point being dragged on a graph goes to the point level with the
        target (see :func:`along_axis`); otherwise it follows the curve from
        where it is (:func:`nearest_on_curve`), and a point being placed goes
        to the nearest point.

        Args:
            view: The view.
            target: The screen point.
            current_t: The point's parameter while dragged, or ``None`` for
                the nearest point.

        Returns:
            The position, or ``None`` if the curve has no defined points.
        """
        pieces, closed, axis = self._pieces(view)
        if current_t is None:
            axis = None
        if axis is not None:
            t = along_axis(pieces, target, axis)
        else:
            t = nearest_on_curve(
                pieces, target, current_t, closed=closed, switch_px=SWITCH_PX
            )
        if t is None:
            return None
        t = self._refine(view, target, t, segment_span(pieces, t), axis)
        placement = self.at(t)
        return placement if math.isfinite(placement.x) else None

    def _refine(
        self,
        view: View,
        target: tuple[float, float],
        t: float,
        span: float,
        axis: Literal["x", "y"] | None = None,
    ) -> float:
        """Move ``t`` to the point of the real curve nearest the target.

        The search samples cover the chord the point was found on, so the
        result no longer depends on how finely the curve was sampled.

        Args:
            view: The view.
            target: The screen point.
            t: The parameter found on the sampled curve.
            span: The parameter width of its segment.
            axis: Measure only along this screen axis (for a graph: the
                point level with the target), or ``None`` for the distance.

        Returns:
            The refined parameter (``t`` itself if nothing nearer is found).
        """
        low, high = self.t_range
        if span <= 0.0:
            return t
        best = t
        width = span
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(_REFINE_ROUNDS):
                candidates = np.clip(
                    np.linspace(best - width, best + width, _REFINE_SAMPLES), low, high
                )
                candidates = np.append(candidates, best)
                xs, ys = self._world(candidates)
                px, py = view.to_screen(xs, ys)
                distance = np.hypot(px - target[0], py - target[1])
                if axis is not None:  # still defined there, but level with it
                    level = np.abs(px - target[0] if axis == "x" else py - target[1])
                    distance = np.where(np.isfinite(distance), level, np.nan)
                if not np.isfinite(distance).any():
                    return t
                best = float(candidates[int(np.nanargmin(distance))])
                width *= 2.0 / (_REFINE_SAMPLES - 1)
        return best

    def _pieces(
        self, view: View
    ) -> tuple[list[CurvePiece], bool, Literal["x", "y"] | None]:
        """Return the curve sampled on screen, whether it is closed, and its graph axis.

        Args:
            view: The view.

        Returns:
            The pieces, the closed flag and the screen axis the curve is a
            graph over (see :func:`graph_axis`; ``None`` if neither, or if it
            is closed), cached for the view.
        """
        key = (
            view.center,
            view.scale_x,
            view.scale_y,
            view.rect,
            view.log_x,
            view.log_y,
        )
        if key == self._cache_key:
            return self._cache

        def screen(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xs, ys = self._world(t)
            return view.to_screen(xs, ys)

        pieces = sample_curve_pieces(
            screen, self.t_range, _sampling_box(view.rect), self._sampling
        )
        closed = _is_closed(pieces, self.t_range)
        axis = None if closed else graph_axis(pieces)
        self._cache_key, self._cache = key, (pieces, closed, axis)
        return self._cache


def _sampling_box(rect: ScreenRect) -> tuple[float, float, float, float]:
    """Return the screen area to sample a curve over: the plane and a margin.

    Args:
        rect: The plane's rectangle.

    Returns:
        ``(left, top, right, bottom)``.
    """
    mx, my = rect.width * _SAMPLE_MARGIN, rect.height * _SAMPLE_MARGIN
    return (rect.left - mx, rect.top - my, rect.right + mx, rect.bottom + my)


def _is_closed(pieces: list[CurvePiece], t_range: tuple[float, float]) -> bool:
    """Return whether a sampled curve ends where it starts.

    Args:
        pieces: The curve on screen.
        t_range: Its parameter range.

    Returns:
        ``True`` if the curve is defined at both ends of the range and they are
        within :data:`CLOSED_PX` of each other.
    """
    if not pieces:
        return False
    (first_points, first_t), (last_points, last_t) = pieces[0], pieces[-1]
    if first_t[0] != t_range[0] or last_t[-1] != t_range[1]:
        return False
    gap = first_points[0] - last_points[-1]
    return bool(math.hypot(float(gap[0]), float(gap[1])) <= CLOSED_PX)
