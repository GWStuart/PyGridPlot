"""The view: a camera looking at part of the infinite world plane.

A :class:`View` owns the state that decides which part of the world is visible
(its centre and its scale in pixels per world unit) and converts coordinates
between the world and the screen. **Every world/screen conversion in the
library goes through this class**, so the sign conventions only live in one
place.

Coordinate conventions
----------------------
* World coordinates are ordinary Cartesian coordinates: ``y`` points up.
* Screen coordinates are pygame pixels: ``y`` points down and integer values
  are pixel centres (see :mod:`pygridplot.core.pixels`).
* The view centre ``(c_x, c_y)`` is drawn at the *origin pixel*
  ``(o_x, o_y) = (rect.x + rect.width // 2, rect.y + rect.height // 2)``.
  The origin is always a whole pixel, so the default view (centre at
  ``(0, 0)``, integer scale) puts grid lines exactly on pixel centres.

The transforms are::

    p_x = o_x + s_x * (x - c_x)        x = c_x + (p_x - o_x) / s_x
    p_y = o_y - s_y * (y - c_y)        y = c_y - (p_y - o_y) / s_y

All transforms accept scalars or NumPy arrays so that whole polylines can be
converted at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import Axis, Bounds, ScreenRect

ZoomAxis = Literal["both", "x", "y"]
"""Which axes a zoom operation scales."""


@dataclass(frozen=True, slots=True)
class ViewLimits:
    """Limits that keep the view inside the range float64 can render accurately.

    ``float64`` has about 16 significant digits. If a pixel becomes smaller
    than about ``1e-12`` times the size of the coordinates on screen, grid
    lines start to jitter and merge. These limits stop the user from zooming
    (or programmatically moving) the view into that regime.

    Attributes:
        min_scale: Smallest allowed scale in pixels per world unit (most
            zoomed out).
        max_scale: Largest allowed scale in pixels per world unit (most
            zoomed in), even around the origin.
        min_relative_pixel: Smallest allowed size of one pixel in world units,
            relative to the magnitude of the view centre. For example ``1e-12``
            means that at a centre of ``1000`` a pixel can be no smaller than
            ``1e-9`` world units.
        max_abs_center: Largest allowed magnitude of either centre coordinate.
    """

    min_scale: float = 1e-12
    max_scale: float = 1e14
    min_relative_pixel: float = 1e-12
    max_abs_center: float = 1e15

    def __post_init__(self) -> None:
        """Validate that the limits are positive and consistent."""
        if not (0 < self.min_scale < self.max_scale < math.inf):
            msg = "ViewLimits requires 0 < min_scale < max_scale < inf"
            raise ValueError(msg)
        if not (0 < self.min_relative_pixel < 1):
            msg = "ViewLimits.min_relative_pixel must be in (0, 1)"
            raise ValueError(msg)
        if not (0 < self.max_abs_center < math.inf):
            msg = "ViewLimits.max_abs_center must be positive and finite"
            raise ValueError(msg)

    def max_scale_near(self, magnitude: float) -> float:
        """Return the largest scale that stays precise near a coordinate.

        Args:
            magnitude: The largest absolute world coordinate that will be on
                screen (typically ``abs(center)``).

        Returns:
            The maximum allowed scale in pixels per world unit. It is never
            below :attr:`min_scale`.
        """
        if magnitude <= 0:
            return self.max_scale
        # Divide twice rather than multiply first: a tiny (even denormal)
        # magnitude times the relative pixel size could underflow to zero.
        precise = 1.0 / magnitude / self.min_relative_pixel
        return max(self.min_scale, min(self.max_scale, precise))

    def clamp_scale(self, scale: float, magnitude: float) -> float:
        """Clamp a scale into the allowed range near a coordinate.

        Args:
            scale: Requested scale in pixels per world unit.
            magnitude: Largest absolute world coordinate that will be on screen.

        Returns:
            ``scale`` limited to ``[min_scale, max_scale_near(magnitude)]``.
        """
        return min(max(scale, self.min_scale), self.max_scale_near(magnitude))

    def clamp_center(self, value: float) -> float:
        """Clamp one centre coordinate into ``[-max_abs_center, max_abs_center]``.

        Args:
            value: Requested centre coordinate in world units.

        Returns:
            The clamped coordinate.
        """
        return min(max(value, -self.max_abs_center), self.max_abs_center)


@dataclass(frozen=True, slots=True)
class ViewState:
    """A snapshot of the parts of a :class:`View` that pan and zoom change.

    Attributes:
        center_x: World x coordinate shown at the origin pixel.
        center_y: World y coordinate shown at the origin pixel.
        scale_x: Horizontal scale in pixels per world unit.
        scale_y: Vertical scale in pixels per world unit.
    """

    center_x: float
    center_y: float
    scale_x: float
    scale_y: float


def _check_finite(name: str, value: float) -> float:
    """Return ``value`` as a float, raising if it is NaN or infinite.

    Args:
        name: Parameter name used in the error message.
        value: Value to check.

    Returns:
        ``float(value)``.

    Raises:
        ValueError: If the value is not finite.
    """
    result = float(value)
    if not math.isfinite(result):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    return result


def _check_scale(name: str, value: float) -> float:
    """Return ``value`` as a float, raising unless it is finite and positive.

    Args:
        name: Parameter name used in the error message.
        value: Value to check.

    Returns:
        ``float(value)``.

    Raises:
        ValueError: If the value is not a positive finite number.
    """
    result = _check_finite(name, value)
    if result <= 0:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return result


class View:
    """A mutable camera that maps the world plane onto a screen rectangle.

    Args:
        rect: The screen rectangle the plane occupies. Anything accepted by
            :meth:`ScreenRect.from_any` works, including ``pygame.Rect``.
        center: World point shown at the middle of ``rect``.
        scale: Pixels per world unit. Either one number (used for both axes)
            or an ``(x_scale, y_scale)`` pair when ``equal_aspect`` is off.
        equal_aspect: If ``True`` (the default) both axes always share one
            scale, so circles look round.
        limits: Zoom and pan limits; see :class:`ViewLimits`.

    Raises:
        ValueError: If the centre or scale is invalid, or if two different
            scales are given while ``equal_aspect`` is on.
    """

    def __init__(
        self,
        rect: object,
        center: tuple[float, float] = (0.0, 0.0),
        scale: float | tuple[float, float] = 50.0,
        *,
        equal_aspect: bool = True,
        limits: ViewLimits | None = None,
    ) -> None:
        self._rect = ScreenRect.from_any(rect)
        self._equal_aspect = equal_aspect
        self._limits = limits if limits is not None else ViewLimits()
        self._cx = 0.0
        self._cy = 0.0
        self._sx = 1.0
        self._sy = 1.0
        self.center = center
        if isinstance(scale, tuple):
            self.set_scale(*scale)
        else:
            self.set_scale(scale)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def rect(self) -> ScreenRect:
        """The screen rectangle the plane occupies.

        Assigning a new rectangle keeps the centre and scale unchanged, so
        resizing a window reveals more (or less) of the plane instead of
        stretching it.
        """
        return self._rect

    @rect.setter
    def rect(self, rect: object) -> None:
        self._rect = ScreenRect.from_any(rect)

    @property
    def equal_aspect(self) -> bool:
        """Whether both axes are forced to share one scale."""
        return self._equal_aspect

    @property
    def limits(self) -> ViewLimits:
        """The zoom and pan limits applied to this view."""
        return self._limits

    @property
    def origin(self) -> tuple[float, float]:
        """The pixel ``(o_x, o_y)`` at which the view centre is drawn."""
        rect = self._rect
        return (float(rect.x + rect.width // 2), float(rect.y + rect.height // 2))

    @property
    def center(self) -> tuple[float, float]:
        """The world point ``(c_x, c_y)`` shown at the origin pixel."""
        return (self._cx, self._cy)

    @center.setter
    def center(self, value: tuple[float, float]) -> None:
        x, y = value
        limits = self._limits
        self._cx = limits.clamp_center(_check_finite("center x", x))
        self._cy = limits.clamp_center(_check_finite("center y", y))
        # Moving the centre far from the origin can make the current zoom
        # level imprecise, so re-apply the scale limits.
        self._apply_scale(self._sx, self._sy)

    @property
    def center_x(self) -> float:
        """World x coordinate shown at the origin pixel."""
        return self._cx

    @property
    def center_y(self) -> float:
        """World y coordinate shown at the origin pixel."""
        return self._cy

    @property
    def scale_x(self) -> float:
        """Horizontal scale in pixels per world unit."""
        return self._sx

    @property
    def scale_y(self) -> float:
        """Vertical scale in pixels per world unit."""
        return self._sy

    @property
    def scale(self) -> float:
        """The shared scale in pixels per world unit.

        Raises:
            ValueError: If the axes have different scales (only possible when
                ``equal_aspect`` is off). Use :attr:`scale_x` and
                :attr:`scale_y` instead.
        """
        if self._sx != self._sy:
            msg = "the axes have different scales; use scale_x and scale_y"
            raise ValueError(msg)
        return self._sx

    def axis_scale(self, axis: Axis) -> float:
        """Return the scale of one axis in pixels per world unit.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            :attr:`scale_x` or :attr:`scale_y`.
        """
        return self._sx if axis == "x" else self._sy

    def axis_center(self, axis: Axis) -> float:
        """Return the centre coordinate of one axis in world units.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            :attr:`center_x` or :attr:`center_y`.
        """
        return self._cx if axis == "x" else self._cy

    @property
    def bounds(self) -> Bounds:
        """The region of the world covered by :attr:`rect`.

        Because integer screen coordinates are pixel centres, the rectangle
        covers continuous coordinates from ``left - 0.5`` to ``right - 0.5``.
        """
        rect = self._rect
        x0, y0 = self.to_world(rect.left - 0.5, rect.bottom - 0.5)
        x1, y1 = self.to_world(rect.right - 0.5, rect.top - 0.5)
        return Bounds(x0, x1, y0, y1)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    @overload
    def to_screen(self, x: float, y: float) -> tuple[float, float]: ...
    @overload
    def to_screen(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def to_screen(self, x: ArrayLike, y: ArrayLike) -> tuple[object, object]:
        """Convert world coordinates to continuous screen coordinates.

        Args:
            x: World x coordinate(s): a number or an array.
            y: World y coordinate(s): a number or an array.

        Returns:
            The screen coordinates ``(p_x, p_y)`` in pixels, as floats for
            scalar input and as float arrays otherwise.
        """
        return (self.to_screen_x(x), self.to_screen_y(y))

    @overload
    def to_world(self, px: float, py: float) -> tuple[float, float]: ...
    @overload
    def to_world(
        self, px: ArrayLike, py: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def to_world(self, px: ArrayLike, py: ArrayLike) -> tuple[object, object]:
        """Convert screen coordinates to world coordinates.

        Args:
            px: Screen x coordinate(s) in pixels: a number or an array.
            py: Screen y coordinate(s) in pixels: a number or an array.

        Returns:
            The world coordinates ``(x, y)``, as floats for scalar input and as
            float arrays otherwise.
        """
        return (self.to_world_x(px), self.to_world_y(py))

    @overload
    def to_screen_x(self, x: float) -> float: ...
    @overload
    def to_screen_x(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def to_screen_x(self, x: ArrayLike) -> object:
        """Convert world x coordinate(s) to screen x coordinate(s).

        Args:
            x: World x coordinate(s).

        Returns:
            ``o_x + s_x * (x - c_x)``.
        """
        ox = self.origin[0]
        if isinstance(x, (int, float)):
            return ox + self._sx * (float(x) - self._cx)
        return ox + self._sx * (np.asarray(x, dtype=np.float64) - self._cx)

    @overload
    def to_screen_y(self, y: float) -> float: ...
    @overload
    def to_screen_y(self, y: ArrayLike) -> NDArray[np.float64]: ...
    def to_screen_y(self, y: ArrayLike) -> object:
        """Convert world y coordinate(s) to screen y coordinate(s).

        Args:
            y: World y coordinate(s).

        Returns:
            ``o_y - s_y * (y - c_y)``. Note the minus sign: screen y points down.
        """
        oy = self.origin[1]
        if isinstance(y, (int, float)):
            return oy - self._sy * (float(y) - self._cy)
        return oy - self._sy * (np.asarray(y, dtype=np.float64) - self._cy)

    @overload
    def to_world_x(self, px: float) -> float: ...
    @overload
    def to_world_x(self, px: ArrayLike) -> NDArray[np.float64]: ...
    def to_world_x(self, px: ArrayLike) -> object:
        """Convert screen x coordinate(s) to world x coordinate(s).

        Args:
            px: Screen x coordinate(s) in pixels.

        Returns:
            ``c_x + (p_x - o_x) / s_x``.
        """
        ox = self.origin[0]
        if isinstance(px, (int, float)):
            return self._cx + (float(px) - ox) / self._sx
        return self._cx + (np.asarray(px, dtype=np.float64) - ox) / self._sx

    @overload
    def to_world_y(self, py: float) -> float: ...
    @overload
    def to_world_y(self, py: ArrayLike) -> NDArray[np.float64]: ...
    def to_world_y(self, py: ArrayLike) -> object:
        """Convert screen y coordinate(s) to world y coordinate(s).

        Args:
            py: Screen y coordinate(s) in pixels.

        Returns:
            ``c_y - (p_y - o_y) / s_y``.
        """
        oy = self.origin[1]
        if isinstance(py, (int, float)):
            return self._cy - (float(py) - oy) / self._sy
        return self._cy - (np.asarray(py, dtype=np.float64) - oy) / self._sy

    def multiples_to_screen(
        self, axis: Axis, step: float, indices: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Precisely convert the world values ``indices * step`` to screen pixels.

        This is the transform the grid uses for its lines. Instead of computing
        ``indices * step - center``, which subtracts two large, nearly equal
        numbers when zoomed in far from the origin, it works relative to the
        multiple of ``step`` nearest the centre::

            i_c    = round(center / step)
            offset = (indices - i_c) * step - (center - i_c * step)

        Both terms stay small, so neighbouring lines keep exactly even spacing
        on screen.

        Args:
            axis: ``"x"`` for vertical grid lines, ``"y"`` for horizontal ones.
            step: Spacing between consecutive multiples, in world units.
            indices: Integer multipliers of ``step``.

        Returns:
            The continuous screen coordinate (x or y respectively) of each
            multiple.
        """
        center = self.axis_center(axis)
        scale = self.axis_scale(axis)
        ratio = center / step
        if abs(ratio) < 2.0**52:
            nearest = round(ratio)
            remainder = center - nearest * step
            offsets = (indices - nearest).astype(np.float64) * step - remainder
        else:  # Far outside the supported range: fall back to the plain formula.
            offsets = indices.astype(np.float64) * step - center
        ox, oy = self.origin
        if axis == "x":
            return ox + scale * offsets
        return oy - scale * offsets

    # ------------------------------------------------------------------
    # Pan and zoom
    # ------------------------------------------------------------------
    def pan_pixels(self, dx: float, dy: float) -> None:
        """Move the plane's content by ``(dx, dy)`` screen pixels.

        This matches dragging: content follows the mouse, so dragging right
        (positive ``dx``) reveals more of the plane to the left.

        Args:
            dx: Horizontal movement in pixels (positive is right).
            dy: Vertical movement in pixels (positive is down).
        """
        self._set_center_raw(self._cx - dx / self._sx, self._cy + dy / self._sy)

    def pan_world(self, dx: float, dy: float) -> None:
        """Move the view centre by ``(dx, dy)`` world units.

        Args:
            dx: Change in the centre's x coordinate.
            dy: Change in the centre's y coordinate.
        """
        self._set_center_raw(self._cx + dx, self._cy + dy)

    def pin(self, world: tuple[float, float], screen: tuple[float, float]) -> None:
        """Move the view (without zooming) so that a world point sits at a pixel.

        Dragging uses this with the world point grabbed when the mouse button
        went down, so the grabbed point stays exactly under the cursor. Unlike
        adding up per-event deltas, this never drifts.

        Args:
            world: The world point ``(x, y)`` to move.
            screen: The screen position ``(p_x, p_y)`` it should appear at.
        """
        wx, wy = world
        px, py = screen
        ox, oy = self.origin
        self._set_center_raw(wx - (px - ox) / self._sx, wy + (py - oy) / self._sy)

    def zoom_at(
        self, px: float, py: float, factor: float, *, axis: ZoomAxis = "both"
    ) -> float:
        """Zoom by ``factor`` while keeping the world point under ``(px, py)`` fixed.

        Factors above 1 zoom in (more pixels per world unit). The result is
        clamped to :attr:`limits`, and the point under the cursor stays put
        even when clamping applies.

        Args:
            px: Screen x coordinate of the zoom anchor (usually the mouse).
            py: Screen y coordinate of the zoom anchor.
            factor: Multiplier applied to the scale.
            axis: ``"both"``, or ``"x"``/``"y"`` to stretch a single axis
                (only allowed when :attr:`equal_aspect` is off).

        Returns:
            The factor actually applied to the scale. This is ``factor`` unless
            a limit was reached. For single-axis zooms it is the factor of that
            axis.

        Raises:
            ValueError: If ``factor`` is not a positive finite number, or a
                single-axis zoom is requested with equal aspect on.
        """
        factor = _check_scale("zoom factor", factor)
        if axis != "both" and self._equal_aspect:
            msg = "single-axis zoom requires equal_aspect=False"
            raise ValueError(msg)
        anchor = self.to_world(float(px), float(py))
        old_sx, old_sy = self._sx, self._sy
        new_sx = old_sx * factor if axis in ("both", "x") else old_sx
        new_sy = old_sy * factor if axis in ("both", "y") else old_sy
        self._apply_scale(new_sx, new_sy, extra_magnitude=anchor)
        self.pin(anchor, (float(px), float(py)))
        if axis == "y":
            return self._sy / old_sy
        return self._sx / old_sx

    def zoom(self, factor: float) -> float:
        """Zoom by ``factor`` about the middle of the plane.

        Args:
            factor: Multiplier applied to the scale (above 1 zooms in).

        Returns:
            The factor actually applied after clamping to :attr:`limits`.
        """
        ox, oy = self.origin
        return self.zoom_at(ox, oy, factor)

    def set_scale(self, scale_x: float, scale_y: float | None = None) -> None:
        """Set the scale directly, keeping the centre fixed.

        Args:
            scale_x: Horizontal scale in pixels per world unit (also the vertical
                scale when ``scale_y`` is omitted).
            scale_y: Vertical scale, only allowed to differ from ``scale_x``
                when :attr:`equal_aspect` is off.

        Raises:
            ValueError: If a scale is not positive and finite, or the scales
                differ while :attr:`equal_aspect` is on.
        """
        sx = _check_scale("scale", scale_x)
        sy = sx if scale_y is None else _check_scale("scale", scale_y)
        if self._equal_aspect and sx != sy:
            msg = "equal_aspect views need scale_x == scale_y"
            raise ValueError(msg)
        self._apply_scale(sx, sy)

    def fit(
        self,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """Centre and scale the view so the given world ranges are visible.

        With equal aspect on, the tighter of the two ranges decides the
        scale, so both ranges fit and the other axis shows extra space. An
        omitted range keeps that axis' current centre (and scale, when
        equal aspect is off).

        Args:
            x_range: ``(x_min, x_max)`` that should span the plane's width.
            y_range: ``(y_min, y_max)`` that should span the plane's height.

        Raises:
            ValueError: If a range is empty, reversed or not finite.
        """
        half_w, half_h = self._half_extents()
        scales: dict[str, float] = {}
        cx, cy = self._cx, self._cy
        if x_range is not None:
            x0, x1 = _check_range("x_range", x_range)
            cx = (x0 + x1) / 2
            if half_w > 0:
                scales["x"] = half_w / ((x1 - x0) / 2)
        if y_range is not None:
            y0, y1 = _check_range("y_range", y_range)
            cy = (y0 + y1) / 2
            if half_h > 0:
                scales["y"] = half_h / ((y1 - y0) / 2)
        self._set_center_raw(cx, cy)
        if not scales:
            return
        if self._equal_aspect:
            shared = min(scales.values())
            self._apply_scale(shared, shared)
        else:
            self._apply_scale(scales.get("x", self._sx), scales.get("y", self._sy))

    def state(self) -> ViewState:
        """Return a snapshot of the centre and scale (see :meth:`restore`).

        Returns:
            The current :class:`ViewState`.
        """
        return ViewState(self._cx, self._cy, self._sx, self._sy)

    def restore(self, state: ViewState) -> None:
        """Restore the centre and scale from a snapshot taken with :meth:`state`.

        Args:
            state: The snapshot to restore. If equal aspect is on and the
                snapshot's scales differ, the x scale is used for both axes.
        """
        self._set_center_raw(state.center_x, state.center_y)
        sy = state.scale_x if self._equal_aspect else state.scale_y
        self._apply_scale(state.scale_x, sy)

    def copy(self) -> View:
        """Return an independent copy of this view.

        Returns:
            A new :class:`View` with the same rect, centre, scale and limits.
        """
        clone = View.__new__(View)
        clone._rect = self._rect
        clone._equal_aspect = self._equal_aspect
        clone._limits = self._limits
        clone._cx, clone._cy = self._cx, self._cy
        clone._sx, clone._sy = self._sx, self._sy
        return clone

    def __repr__(self) -> str:
        return (
            f"View(center=({self._cx:.6g}, {self._cy:.6g}), "
            f"scale=({self._sx:.6g}, {self._sy:.6g}), rect={self._rect})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _half_extents(self) -> tuple[float, float]:
        """Return the shorter distance from the origin pixel to the rect's edges.

        The origin pixel is a whole pixel, so for even sizes it sits half a
        pixel off the true middle. Fitting with the *shorter* side guarantees
        a fitted range is fully visible while grid lines stay on whole
        pixels.

        Returns:
            ``(horizontal, vertical)`` distances in pixels.
        """
        rect = self._rect
        ox, oy = self.origin
        left = ox - (rect.left - 0.5)
        right = (rect.right - 0.5) - ox
        top = oy - (rect.top - 0.5)
        bottom = (rect.bottom - 0.5) - oy
        return (max(0.0, min(left, right)), max(0.0, min(top, bottom)))

    def _set_center_raw(self, cx: float, cy: float) -> None:
        """Set the centre, clamping it to the limits but leaving the scale alone.

        Pan operations use this instead of the :attr:`center` setter so that
        a drag can never change the zoom level.

        Args:
            cx: New centre x coordinate.
            cy: New centre y coordinate.
        """
        limits = self._limits
        if math.isfinite(cx):
            self._cx = limits.clamp_center(cx)
        if math.isfinite(cy):
            self._cy = limits.clamp_center(cy)

    def _apply_scale(
        self,
        sx: float,
        sy: float,
        *,
        extra_magnitude: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Set the scales after clamping them to the precision limits.

        Args:
            sx: Requested horizontal scale.
            sy: Requested vertical scale.
            extra_magnitude: Additional world point (for example a zoom
                anchor) whose coordinates must also stay precise.
        """
        limits = self._limits
        mag_x = max(abs(self._cx), abs(extra_magnitude[0]))
        mag_y = max(abs(self._cy), abs(extra_magnitude[1]))
        if self._equal_aspect:
            shared = limits.clamp_scale(sx, max(mag_x, mag_y))
            self._sx = self._sy = shared
        else:
            self._sx = limits.clamp_scale(sx, mag_x)
            self._sy = limits.clamp_scale(sy, mag_y)


def _check_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    """Validate a ``(low, high)`` world range.

    Args:
        name: Parameter name used in error messages.
        value: The range to check.

    Returns:
        The range as a tuple of floats.

    Raises:
        ValueError: If the range is not finite or ``high <= low``.
    """
    low = _check_finite(f"{name}[0]", value[0])
    high = _check_finite(f"{name}[1]", value[1])
    if high <= low:
        msg = f"{name} must satisfy low < high, got {value!r}"
        raise ValueError(msg)
    return (low, high)
