"""The view: a camera looking at part of the infinite world plane.

A :class:`View` owns the state that decides which part of the world is visible
(its centre and its scale) and converts coordinates between the world and the
screen. **Every world/screen conversion in the library goes through this
class**, so the sign conventions only live in one place.

Coordinate conventions
----------------------
* World coordinates are ordinary Cartesian coordinates: ``y`` points up.
* Screen coordinates are pygame pixels: ``y`` points down and integer values
  are pixel centres (see :mod:`pygridplot.core.pixels`).
* The view centre is drawn at the *origin pixel*
  ``(o_x, o_y) = (rect.x + rect.width // 2, rect.y + rect.height // 2)``.
  The origin is always a whole pixel, so the default view (centre at
  ``(0, 0)``, integer scale) puts grid lines exactly on pixel centres.

Linear and logarithmic axes
---------------------------
Each axis has an :class:`~pygridplot.core.transforms.AxisTransform`. The view
works in **transformed coordinates** ``u = T(x)``: the world value itself on
a linear axis, and ``log10(x)`` on a logarithmic one. The centre
``(c_x, c_y)`` and the scale ``(s_x, s_y)`` are stored in transformed
coordinates, so the scale of a log axis is in *pixels per decade*. The
transforms are::

    p_x = o_x + s_x * (T_x(x) - c_x)        x = T_x⁻¹(c_x + (p_x - o_x) / s_x)
    p_y = o_y - s_y * (T_y(y) - c_y)        y = T_y⁻¹(c_y - (p_y - o_y) / s_y)

The screen is linear in transformed coordinates, so panning, zooming and
fitting work the same way on both kinds of axis. All transforms accept
scalars or NumPy arrays so that whole polylines can be converted at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import Axis, Bounds, ScreenRect
from pygridplot.core.transforms import MAX_LOG_DECADE, AxisTransform, transform_for

ZoomAxis = Literal["both", "x", "y"]
"""Which axes a zoom operation scales."""

DEFAULT_SCALE: Final = 50.0
"""Default scale of a linear axis, in pixels per world unit."""

DEFAULT_LOG_SCALE: Final = 150.0
"""Default scale of a logarithmic axis, in pixels per decade."""

MIN_LOG_SCALE: Final = 1.0
"""Smallest scale of a logarithmic axis, in pixels per decade.

At one pixel per decade a few hundred pixels already show every power of ten
a float can hold, so zooming out further is pointless.
"""

_LOG_PRECISION_FLOOR: Final = 1.0 / math.log(10.0)
"""Smallest magnitude used when limiting the zoom of a log axis.

One pixel on a log axis is a *relative* step of ``ln(10) / scale`` in world
units, so the precision limit never gets looser than it is at ``|log10 x|``
of about ``1 / ln(10)``, however close the view is to ``x = 1``.
"""

_SWITCH_LOG_DECADES: Final = 3.0
"""Decades kept below the top of the view when an axis is switched to log
while the visible range includes zero or negative values."""


@dataclass(frozen=True, slots=True)
class ViewLimits:
    """Limits that keep the view inside the range float64 can render accurately.

    ``float64`` has about 16 significant digits. If a pixel becomes smaller
    than about ``1e-12`` times the size of the coordinates on screen, grid
    lines start to jitter and merge. These limits stop the user from zooming
    (or programmatically moving) the view into that regime.

    On a logarithmic axis the limits apply to the transformed coordinate
    (``log10`` of the value): scales are in pixels per decade and the centre
    is limited to ``10**±300``.

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
        scale_x: Horizontal scale (pixels per world unit, or per decade on a
            log axis).
        scale_y: Vertical scale (pixels per world unit, or per decade on a
            log axis).
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


def default_scale(transform: AxisTransform) -> float:
    """Return the default scale of an axis.

    Args:
        transform: The axis' transform.

    Returns:
        :data:`DEFAULT_LOG_SCALE` (pixels per decade) for a log axis,
        :data:`DEFAULT_SCALE` (pixels per unit) otherwise.
    """
    return DEFAULT_LOG_SCALE if transform.is_log else DEFAULT_SCALE


class View:
    """A mutable camera that maps the world plane onto a screen rectangle.

    Args:
        rect: The screen rectangle the plane occupies. Anything accepted by
            :meth:`ScreenRect.from_any` works, including ``pygame.Rect``.
        center: World point shown at the middle of ``rect``. ``None`` uses 0
            on a linear axis and 1 on a log axis.
        scale: Pixels per world unit (per decade on a log axis). Either one
            number (used for both axes) or an ``(x_scale, y_scale)`` pair when
            ``equal_aspect`` is off. ``None`` uses :data:`DEFAULT_SCALE` and
            :data:`DEFAULT_LOG_SCALE`.
        equal_aspect: If ``True`` both axes always share one scale, so circles
            look round. ``None`` (the default) turns it on unless exactly one
            axis is logarithmic.
        limits: Zoom and pan limits; see :class:`ViewLimits`.
        log_x: Make the x axis logarithmic (base 10).
        log_y: Make the y axis logarithmic (base 10).

    Raises:
        ValueError: If the centre or scale is invalid (for example a centre
            that is not positive on a log axis), or if two different scales
            are given while ``equal_aspect`` is on.
    """

    def __init__(
        self,
        rect: object,
        center: tuple[float, float] | None = None,
        scale: float | tuple[float, float] | None = None,
        *,
        equal_aspect: bool | None = None,
        limits: ViewLimits | None = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> None:
        self._rect = ScreenRect.from_any(rect)
        self._tx = transform_for(log_x)
        self._ty = transform_for(log_y)
        self._equal_aspect = log_x == log_y if equal_aspect is None else equal_aspect
        self._limits = limits if limits is not None else ViewLimits()
        self._cx = 0.0
        self._cy = 0.0
        self._sx = 1.0
        self._sy = 1.0
        if center is None:
            center = (self._tx.default_center, self._ty.default_center)
        self.center = center
        if scale is None:
            sx, sy = default_scale(self._tx), default_scale(self._ty)
            self.set_scale(sx, sx if self._equal_aspect else sy)
        elif isinstance(scale, tuple):
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
        """Whether both axes are forced to share one scale.

        Turning it on gives both axes the smaller of their two scales, so
        everything visible on the tighter axis stays visible.
        """
        return self._equal_aspect

    @equal_aspect.setter
    def equal_aspect(self, value: bool) -> None:
        self._equal_aspect = bool(value)
        if self._equal_aspect and self._sx != self._sy:
            shared = min(self._sx, self._sy)
            self._apply_scale(shared, shared)

    @property
    def limits(self) -> ViewLimits:
        """The zoom and pan limits applied to this view."""
        return self._limits

    @property
    def transform_x(self) -> AxisTransform:
        """The x axis' transform (linear or logarithmic)."""
        return self._tx

    @property
    def transform_y(self) -> AxisTransform:
        """The y axis' transform (linear or logarithmic)."""
        return self._ty

    def axis_transform(self, axis: Axis) -> AxisTransform:
        """Return the transform of one axis.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            :attr:`transform_x` or :attr:`transform_y`.
        """
        return self._tx if axis == "x" else self._ty

    @property
    def log_x(self) -> bool:
        """Whether the x axis is logarithmic."""
        return self._tx.is_log

    @property
    def log_y(self) -> bool:
        """Whether the y axis is logarithmic."""
        return self._ty.is_log

    @property
    def origin(self) -> tuple[float, float]:
        """The pixel ``(o_x, o_y)`` at which the view centre is drawn."""
        rect = self._rect
        return (float(rect.x + rect.width // 2), float(rect.y + rect.height // 2))

    @property
    def center(self) -> tuple[float, float]:
        """The world point ``(x, y)`` shown at the origin pixel.

        Assigning a point that a log axis cannot show (zero or negative)
        raises :class:`ValueError`.
        """
        return (self._tx.inverse(self._cx), self._ty.inverse(self._cy))

    @center.setter
    def center(self, value: tuple[float, float]) -> None:
        x, y = value
        cx = self._tx.check("center x", x)
        cy = self._ty.check("center y", y)
        self._set_center_raw(cx, cy)
        # Moving the centre far from the origin can make the current zoom
        # level imprecise, so re-apply the scale limits.
        self._apply_scale(self._sx, self._sy)

    @property
    def center_x(self) -> float:
        """World x coordinate shown at the origin pixel."""
        return self._tx.inverse(self._cx)

    @property
    def center_y(self) -> float:
        """World y coordinate shown at the origin pixel."""
        return self._ty.inverse(self._cy)

    @property
    def scale_x(self) -> float:
        """Horizontal scale: pixels per world unit, or per decade on a log axis."""
        return self._sx

    @property
    def scale_y(self) -> float:
        """Vertical scale: pixels per world unit, or per decade on a log axis."""
        return self._sy

    @property
    def scale(self) -> float:
        """The shared scale of both axes.

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
        """Return the scale of one axis in pixels per transformed unit.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            :attr:`scale_x` or :attr:`scale_y` (pixels per world unit, or per
            decade on a log axis).
        """
        return self._sx if axis == "x" else self._sy

    def axis_center(self, axis: Axis) -> float:
        """Return the centre of one axis in transformed coordinates.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            The centre coordinate: in world units on a linear axis, as
            ``log10`` of the value on a log axis.
        """
        return self._cx if axis == "x" else self._cy

    @property
    def bounds(self) -> Bounds:
        """The region of the world covered by :attr:`rect`.

        Because integer screen coordinates are pixel centres, the rectangle
        covers continuous coordinates from ``left - 0.5`` to ``right - 0.5``.
        On a log axis the bounds are always positive (and may underflow to 0
        or overflow to infinity when zoomed out extremely far).
        """
        rect = self._rect
        x0, y0 = self.to_world(rect.left - 0.5, rect.bottom - 0.5)
        x1, y1 = self.to_world(rect.right - 0.5, rect.top - 0.5)
        return Bounds(x0, x1, y0, y1)

    @property
    def transformed_bounds(self) -> Bounds:
        """The region covered by :attr:`rect`, in transformed coordinates.

        Identical to :attr:`bounds` on linear axes. On a log axis it holds the
        ``log10`` of the visible range, which is always finite.
        """
        rect = self._rect
        u0, w0 = self.screen_to_transformed(rect.left - 0.5, rect.bottom - 0.5)
        u1, w1 = self.screen_to_transformed(rect.right - 0.5, rect.top - 0.5)
        return Bounds(u0, u1, w0, w1)

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
            scalar input and as float arrays otherwise. Values a log axis
            cannot show give NaN (negative values) or an infinite coordinate
            off the edge of the plane (zero).
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
            ``o_x + s_x * (T_x(x) - c_x)``.
        """
        return self.origin[0] + self._sx * (self._tx.forward(x) - self._cx)

    @overload
    def to_screen_y(self, y: float) -> float: ...
    @overload
    def to_screen_y(self, y: ArrayLike) -> NDArray[np.float64]: ...
    def to_screen_y(self, y: ArrayLike) -> object:
        """Convert world y coordinate(s) to screen y coordinate(s).

        Args:
            y: World y coordinate(s).

        Returns:
            ``o_y - s_y * (T_y(y) - c_y)``. Note the minus sign: screen y
            points down.
        """
        return self.origin[1] - self._sy * (self._ty.forward(y) - self._cy)

    @overload
    def to_world_x(self, px: float) -> float: ...
    @overload
    def to_world_x(self, px: ArrayLike) -> NDArray[np.float64]: ...
    def to_world_x(self, px: ArrayLike) -> object:
        """Convert screen x coordinate(s) to world x coordinate(s).

        Args:
            px: Screen x coordinate(s) in pixels.

        Returns:
            ``T_x⁻¹(c_x + (p_x - o_x) / s_x)``.
        """
        return self._tx.inverse(self.screen_to_axis("x", px))

    @overload
    def to_world_y(self, py: float) -> float: ...
    @overload
    def to_world_y(self, py: ArrayLike) -> NDArray[np.float64]: ...
    def to_world_y(self, py: ArrayLike) -> object:
        """Convert screen y coordinate(s) to world y coordinate(s).

        Args:
            py: Screen y coordinate(s) in pixels.

        Returns:
            ``T_y⁻¹(c_y - (p_y - o_y) / s_y)``.
        """
        return self._ty.inverse(self.screen_to_axis("y", py))

    @overload
    def axis_to_screen(self, axis: Axis, values: float) -> float: ...
    @overload
    def axis_to_screen(self, axis: Axis, values: ArrayLike) -> NDArray[np.float64]: ...
    def axis_to_screen(self, axis: Axis, values: ArrayLike) -> object:
        """Convert transformed coordinates along one axis to screen coordinates.

        Args:
            axis: ``"x"`` or ``"y"``.
            values: Transformed coordinate(s): world units on a linear axis,
                ``log10`` of the value on a log axis.

        Returns:
            The screen x (for ``"x"``) or y (for ``"y"``) coordinate(s).
        """
        center = self.axis_center(axis)
        ox, oy = self.origin
        if isinstance(values, (int, float)):
            offset = float(values) - center
            return ox + self._sx * offset if axis == "x" else oy - self._sy * offset
        offsets = np.asarray(values, dtype=np.float64) - center
        return ox + self._sx * offsets if axis == "x" else oy - self._sy * offsets

    @overload
    def screen_to_axis(self, axis: Axis, positions: float) -> float: ...
    @overload
    def screen_to_axis(
        self, axis: Axis, positions: ArrayLike
    ) -> NDArray[np.float64]: ...
    def screen_to_axis(self, axis: Axis, positions: ArrayLike) -> object:
        """Convert screen coordinates along one axis to transformed coordinates.

        Args:
            axis: ``"x"`` (screen x positions) or ``"y"`` (screen y positions).
            positions: Screen coordinate(s) in pixels.

        Returns:
            The transformed coordinate(s) (see :meth:`axis_to_screen`).
        """
        ox, oy = self.origin
        if isinstance(positions, (int, float)):
            pixel = float(positions)
            if axis == "x":
                return self._cx + (pixel - ox) / self._sx
            return self._cy - (pixel - oy) / self._sy
        pixels = np.asarray(positions, dtype=np.float64)
        if axis == "x":
            return self._cx + (pixels - ox) / self._sx
        return self._cy - (pixels - oy) / self._sy

    @overload
    def transformed_to_screen(self, u: float, w: float) -> tuple[float, float]: ...
    @overload
    def transformed_to_screen(
        self, u: ArrayLike, w: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def transformed_to_screen(
        self, u: ArrayLike, w: ArrayLike
    ) -> tuple[object, object]:
        """Convert transformed coordinates to screen coordinates.

        Args:
            u: Transformed x coordinate(s).
            w: Transformed y coordinate(s).

        Returns:
            The screen coordinates ``(p_x, p_y)``.
        """
        return (self.axis_to_screen("x", u), self.axis_to_screen("y", w))

    @overload
    def screen_to_transformed(self, px: float, py: float) -> tuple[float, float]: ...
    @overload
    def screen_to_transformed(
        self, px: ArrayLike, py: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def screen_to_transformed(
        self, px: ArrayLike, py: ArrayLike
    ) -> tuple[object, object]:
        """Convert screen coordinates to transformed coordinates.

        Args:
            px: Screen x coordinate(s) in pixels.
            py: Screen y coordinate(s) in pixels.

        Returns:
            The transformed coordinates ``(u, w)``.
        """
        return (self.screen_to_axis("x", px), self.screen_to_axis("y", py))

    def multiples_to_screen(
        self, axis: Axis, step: float, indices: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Precisely convert the transformed values ``indices * step`` to pixels.

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
            step: Spacing between consecutive multiples, in transformed units
                (world units, or decades on a log axis).
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

        Raises:
            ValueError: If the move would take a log axis' centre to zero or
                below.
        """
        cx = self._tx.check("panned center x", self.center_x + dx) if dx else self._cx
        cy = self._ty.check("panned center y", self.center_y + dy) if dy else self._cy
        self._set_center_raw(cx, cy)

    def pin(self, world: tuple[float, float], screen: tuple[float, float]) -> None:
        """Move the view (without zooming) so that a world point sits at a pixel.

        Dragging uses this with the point grabbed when the mouse button went
        down, so the grabbed point stays exactly under the cursor. Unlike
        adding up per-event deltas, this never drifts.

        Args:
            world: The world point ``(x, y)`` to move. A coordinate a log axis
                cannot show leaves that axis unchanged.
            screen: The screen position ``(p_x, p_y)`` it should appear at.
        """
        point = (self._tx.forward(float(world[0])), self._ty.forward(float(world[1])))
        self.pin_transformed(point, screen)

    def pin_transformed(
        self, point: tuple[float, float], screen: tuple[float, float]
    ) -> None:
        """Like :meth:`pin`, for a point given in transformed coordinates.

        Args:
            point: The transformed point ``(u, w)`` to move.
            screen: The screen position ``(p_x, p_y)`` it should appear at.
        """
        u, w = point
        px, py = screen
        ox, oy = self.origin
        self._set_center_raw(u - (px - ox) / self._sx, w + (py - oy) / self._sy)

    def zoom_at(
        self, px: float, py: float, factor: float, *, axis: ZoomAxis = "both"
    ) -> float:
        """Zoom by ``factor`` while keeping the world point under ``(px, py)`` fixed.

        Factors above 1 zoom in (more pixels per unit). The result is clamped
        to :attr:`limits`, and the point under the cursor stays put even when
        clamping applies.

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
        anchor = self.screen_to_transformed(float(px), float(py))
        old_sx, old_sy = self._sx, self._sy
        new_sx = old_sx * factor if axis in ("both", "x") else old_sx
        new_sy = old_sy * factor if axis in ("both", "y") else old_sy
        self._apply_scale(new_sx, new_sy, extra_magnitude=anchor)
        self.pin_transformed(anchor, (float(px), float(py)))
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
            scale_x: Horizontal scale in pixels per world unit (per decade on a
                log axis). Also the vertical scale when ``scale_y`` is omitted.
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
        equal aspect is off). On a log axis the range is fitted in decades,
        so ``(1, 1000)`` puts 10 and 100 at the thirds.

        Args:
            x_range: ``(x_min, x_max)`` that should span the plane's width.
            y_range: ``(y_min, y_max)`` that should span the plane's height.

        Raises:
            ValueError: If a range is empty, reversed or not finite, or not
                positive on a log axis.
        """
        half_w, half_h = self._half_extents()
        scales: dict[str, float] = {}
        cx, cy = self._cx, self._cy
        if x_range is not None:
            u0, u1 = _check_range("x_range", x_range, self._tx)
            cx = (u0 + u1) / 2
            if half_w > 0:
                scales["x"] = half_w / ((u1 - u0) / 2)
        if y_range is not None:
            w0, w1 = _check_range("y_range", y_range, self._ty)
            cy = (w0 + w1) / 2
            if half_h > 0:
                scales["y"] = half_h / ((w1 - w0) / 2)
        self._set_center_raw(cx, cy)
        if not scales:
            return
        if self._equal_aspect:
            shared = min(scales.values())
            self._apply_scale(shared, shared)
        else:
            self._apply_scale(scales.get("x", self._sx), scales.get("y", self._sy))

    def set_log(self, axis: Axis, log: bool) -> None:
        """Switch one axis between linear and logarithmic.

        The visible range of the axis is kept where the new kind of axis can
        show it. Switching to log while zero or negative values are visible
        keeps the top :data:`_SWITCH_LOG_DECADES` decades below the largest
        visible value, and falls back to the default view of that axis if no
        positive value is visible at all.

        With :attr:`equal_aspect` on, refitting the axis changes the shared
        scale, so turn equal aspect off first when mixing a linear and a log
        axis.

        Args:
            axis: ``"x"`` or ``"y"``.
            log: Whether the axis should be logarithmic.
        """
        new = transform_for(log)
        if new == self.axis_transform(axis):
            return
        low, high = self.bounds.range(axis)
        if new.is_log and high > 0 and math.isfinite(high):
            low = low if low > 0 else high / 10**_SWITCH_LOG_DECADES
            span: tuple[float, float] | None = (low, high)
        elif not new.is_log and math.isfinite(low) and math.isfinite(high):
            span = (low, high)
        else:
            span = None
        if axis == "x":
            self._tx = new
        else:
            self._ty = new
        if span is not None and span[0] < span[1]:
            if axis == "x":
                self.fit(x_range=span)
            else:
                self.fit(y_range=span)
            return
        center = new.forward(new.default_center)
        if axis == "x":
            self._cx = center
            self._apply_scale(default_scale(new), self._sy)
        else:
            self._cy = center
            self._apply_scale(self._sx, default_scale(new))

    def state(self) -> ViewState:
        """Return a snapshot of the centre and scale (see :meth:`restore`).

        Returns:
            The current :class:`ViewState`.
        """
        return ViewState(self.center_x, self.center_y, self._sx, self._sy)

    def restore(self, state: ViewState) -> None:
        """Restore the centre and scale from a snapshot taken with :meth:`state`.

        Args:
            state: The snapshot to restore. If equal aspect is on and the
                snapshot's scales differ, the x scale is used for both axes.
                A centre coordinate a log axis cannot show is ignored.
        """
        self._set_center_raw(
            self._tx.forward(state.center_x), self._ty.forward(state.center_y)
        )
        sy = state.scale_x if self._equal_aspect else state.scale_y
        self._apply_scale(state.scale_x, sy)

    def copy(self) -> View:
        """Return an independent copy of this view.

        Returns:
            A new :class:`View` with the same rect, axes, centre, scale and
            limits.
        """
        clone = View.__new__(View)
        clone._rect = self._rect
        clone._tx, clone._ty = self._tx, self._ty
        clone._equal_aspect = self._equal_aspect
        clone._limits = self._limits
        clone._cx, clone._cy = self._cx, self._cy
        clone._sx, clone._sy = self._sx, self._sy
        return clone

    def __repr__(self) -> str:
        kinds = ""
        if self.log_x or self.log_y:
            kinds = f", log_x={self.log_x}, log_y={self.log_y}"
        cx, cy = self.center
        return (
            f"View(center=({cx:.6g}, {cy:.6g}), "
            f"scale=({self._sx:.6g}, {self._sy:.6g}){kinds}, rect={self._rect})"
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
        """Set the transformed centre, clamping it but leaving the scale alone.

        Pan operations use this instead of the :attr:`center` setter so that
        a drag can never change the zoom level. Non-finite values are ignored.

        Args:
            cx: New transformed centre x coordinate.
            cy: New transformed centre y coordinate.
        """
        if math.isfinite(cx):
            self._cx = self._clamp_center(self._tx, cx)
        if math.isfinite(cy):
            self._cy = self._clamp_center(self._ty, cy)

    def _clamp_center(self, transform: AxisTransform, value: float) -> float:
        """Clamp one transformed centre coordinate to the limits.

        Args:
            transform: The axis' transform.
            value: The requested transformed coordinate.

        Returns:
            The clamped coordinate: within ``±max_abs_center`` on a linear
            axis and within ``±MAX_LOG_DECADE`` decades on a log axis.
        """
        if transform.is_log:
            return min(max(value, -MAX_LOG_DECADE), MAX_LOG_DECADE)
        return self._limits.clamp_center(value)

    def _scale_range(
        self, transform: AxisTransform, center: float, extra: float
    ) -> tuple[float, float]:
        """Return the allowed scale range of one axis.

        Args:
            transform: The axis' transform.
            center: The axis' transformed centre.
            extra: Another transformed coordinate that must stay precise (for
                example a zoom anchor).

        Returns:
            ``(lowest, highest)`` allowed scale.
        """
        limits = self._limits
        magnitude = max(abs(center), abs(extra))
        lowest = limits.min_scale
        if transform.is_log:
            magnitude = max(magnitude, _LOG_PRECISION_FLOOR)
            lowest = max(lowest, MIN_LOG_SCALE)
        return (lowest, max(lowest, limits.max_scale_near(magnitude)))

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
            extra_magnitude: Additional transformed point (for example a zoom
                anchor) whose coordinates must also stay precise.
        """
        lo_x, hi_x = self._scale_range(self._tx, self._cx, extra_magnitude[0])
        lo_y, hi_y = self._scale_range(self._ty, self._cy, extra_magnitude[1])
        if self._equal_aspect:
            low, high = max(lo_x, lo_y), max(max(lo_x, lo_y), min(hi_x, hi_y))
            self._sx = self._sy = min(max(sx, low), high)
        else:
            self._sx = min(max(sx, lo_x), hi_x)
            self._sy = min(max(sy, lo_y), hi_y)


def _check_range(
    name: str, value: tuple[float, float], transform: AxisTransform
) -> tuple[float, float]:
    """Validate a ``(low, high)`` world range and transform it.

    Args:
        name: Parameter name used in error messages.
        value: The range to check.
        transform: The axis' transform.

    Returns:
        The range in transformed coordinates.

    Raises:
        ValueError: If the range is not finite, ``high <= low``, or (on a log
            axis) ``low`` is not positive.
    """
    low = _check_finite(f"{name}[0]", value[0])
    high = _check_finite(f"{name}[1]", value[1])
    if high <= low:
        msg = f"{name} must satisfy low < high, got {value!r}"
        raise ValueError(msg)
    return (transform.check(f"{name}[0]", low), transform.check(f"{name}[1]", high))
