"""Objects drawn as curves: functions, parametric, polar and implicit curves.

:class:`Function`, :class:`ParametricCurve` and :class:`PolarCurve` are
*curve objects*: a draggable :class:`~pygridplot.Point` can follow them
(``pgp.Point(curve=f, draggable=True)``), and they can be called like the
functions they draw (``f(2.0)``, ``pgp.derivative(f)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np

from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.derivatives import Derivative, derivative
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import LineStyle
from pygridplot.objects.base import LineObject
from pygridplot.objects.values import (
    FULL_TURN,
    Component,
    component,
    finite_range,
    function,
    is_number,
    optional_setting,
    show,
)

if TYPE_CHECKING:
    import pygame

    from pygridplot.core.view import View
    from pygridplot.plane import Plane

CurveSpec: TypeAlias = "tuple[Any, Any, tuple[float, float]]"
"""A curve for a draggable point: ``(x(t), y(t), t_range)``."""


class CurveObject(LineObject):
    """A curve a draggable point can follow: a function, a parametric or polar curve."""

    curve_from_zero: ClassVar[bool] = False
    """Whether a point without a parameter starts at ``t = 0`` (on a function's
    graph, at ``x = 0``) rather than at the start of the range."""

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        """Return the curve to follow, for a view and the point's parameter."""
        raise NotImplementedError


def curve_start(curve: CurveSpec, from_zero: bool) -> float:
    """Return where a point without a parameter starts on ``curve``."""
    low, high = curve[2]
    return 0.0 if from_zero and low <= 0.0 <= high else low


def _identity(t: Any) -> Any:
    return t


class Function(CurveObject):
    """The graph of a function, ``y = f(x)``, drawn like :meth:`Plane.plot`.

    The object is itself a function: ``f(2.0)`` evaluates it, so it can be
    used in other functions (``pgp.Function(lambda x: f(x) ** 2)``) and
    differentiated (``pgp.derivative(f)``, or :meth:`derivative`).

    Args:
        func: ``f``: a function of ``x``, written with NumPy or ``math``.
        domain: Draw only for ``x`` in ``(start, end)``.
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`; the other arguments
            override its fields.
        sampling: How the function is sampled (see
            :class:`~pygridplot.SamplingConfig`); default: the plane's.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane).
    """

    curve_from_zero: ClassVar[bool] = True

    def __init__(
        self,
        func: Callable[[Any], Any],
        *,
        domain: tuple[float, float] | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: SamplingConfig | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color, "width": width, "opacity": opacity},
            palette=palette,
            visible=visible,
        )
        self.func = func
        self.domain = domain
        self.sampling = sampling

    @property
    def func(self) -> Callable[[Any], Any]:
        """``f``: the function of ``x``."""
        return self._func

    @func.setter
    def func(self, func: Callable[[Any], Any]) -> None:
        self._func = function(func, "func")

    @property
    def domain(self) -> tuple[float, float] | None:
        """The ``x`` interval drawn (``None`` for everywhere)."""
        return self._domain

    @domain.setter
    def domain(self, domain: tuple[float, float] | None) -> None:
        self._domain = None if domain is None else finite_range(domain, "domain")

    @property
    def sampling(self) -> SamplingConfig | None:
        """Sampling settings (``None`` for the plane's)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: SamplingConfig | None) -> None:
        self._sampling: SamplingConfig | None = optional_setting(
            sampling, SamplingConfig, "sampling"
        )

    def __call__(self, x: Any) -> Any:
        """Evaluate the function at ``x`` (a number or an array)."""
        return self._func(x)

    def derivative(self, n: int = 1) -> Derivative:
        """Return the ``n``-th derivative (exact for functions written with NumPy).

        It follows the function: changing :attr:`func` changes it too.

        Args:
            n: The order.

        Returns:
            The derivative, a function of ``x``.
        """
        return derivative(self, n)

    def _render(self, plane: Plane, surface: pygame.Surface) -> None:
        plane.plot(
            surface,
            self._func,
            style=self.style,
            color=self._color(plane),
            domain=self._domain,
            sampling=self._sampling,
        )

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        """The graph as a curve ``(t, f(t))``.

        Without a domain, the curve spans the visible x range and as much
        again on both sides (and always the point's own parameter), so the
        point can be dragged anywhere on screen.
        """
        if self._domain is not None:
            return (_identity, self._func, self._domain)
        bounds = view.bounds
        if view.log_x:
            low, high = bounds.x_min / 10.0, bounds.x_max * 10.0
        else:
            width = bounds.x_max - bounds.x_min
            low, high = bounds.x_min - width, bounds.x_max + width
        if t is not None and math.isfinite(t):
            low, high = min(low, t - 1.0), max(high, t + 1.0)
        return (_identity, self._func, (low, high))

    def _describe(self) -> str:
        return f"y = {show(self._func)}"


class ParametricCurve(CurveObject):
    """A parametric curve ``(x(t), y(t))``, drawn like :meth:`Plane.parametric`.

    Calling it gives the point at a parameter: ``c(0.5)`` is ``(x, y)``.

    Args:
        x: ``x(t)``: a function of ``t``, or a number.
        y: ``y(t)``: likewise.
        t_range: ``(t_min, t_max)``; one full turn by default.
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane).
    """

    def __init__(
        self,
        x: Component,
        y: Component,
        t_range: tuple[float, float] = FULL_TURN,
        *,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color, "width": width, "opacity": opacity},
            palette=palette,
            visible=visible,
        )
        self.x = x
        self.y = y
        self.t_range = t_range
        self.sampling = sampling

    @property
    def x(self) -> Component:
        """``x(t)``: a function of ``t``, or a number."""
        return self._x

    @x.setter
    def x(self, x: Component) -> None:
        self._x = component(x, "x(t)")

    @property
    def y(self) -> Component:
        """``y(t)``: a function of ``t``, or a number."""
        return self._y

    @y.setter
    def y(self, y: Component) -> None:
        self._y = component(y, "y(t)")

    @property
    def t_range(self) -> tuple[float, float]:
        """``(t_min, t_max)``."""
        return self._t_range

    @t_range.setter
    def t_range(self, t_range: tuple[float, float]) -> None:
        self._t_range = finite_range(t_range, "t_range")

    @property
    def sampling(self) -> CurveSampling | None:
        """Sampling settings (``None`` for the defaults)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: CurveSampling | None) -> None:
        self._sampling: CurveSampling | None = optional_setting(
            sampling, CurveSampling, "sampling"
        )

    def __call__(self, t: Any) -> tuple[Any, Any]:
        """Return the point ``(x(t), y(t))`` (numbers, or arrays for an array ``t``)."""
        return (_value(self._x, t), _value(self._y, t))

    def _render(self, plane: Plane, surface: pygame.Surface) -> None:
        plane.parametric(
            surface,
            self._x,
            self._y,
            self._t_range,
            style=self.style,
            color=self._color(plane),
            sampling=self._sampling,
        )

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        return (self._x, self._y, self._t_range)

    def _describe(self) -> str:
        return f"({show(self._x)}, {show(self._y)})"


class PolarCurve(CurveObject):
    """A polar curve ``r = r(θ)``, drawn like :meth:`Plane.polar`.

    Calling it gives the radius at an angle: ``R(math.pi)``.

    Args:
        r: ``r(θ)``: a function of the angle, or a number (a circle).
        theta_range: ``(θ_min, θ_max)``; one full turn by default.
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane).
    """

    def __init__(
        self,
        r: Component,
        theta_range: tuple[float, float] = FULL_TURN,
        *,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color, "width": width, "opacity": opacity},
            palette=palette,
            visible=visible,
        )
        self.r = r
        self.theta_range = theta_range
        self.sampling = sampling

    @property
    def r(self) -> Component:
        """``r(θ)``: a function of the angle, or a number."""
        return self._r

    @r.setter
    def r(self, r: Component) -> None:
        self._r = component(r, "r(θ)")

    @property
    def theta_range(self) -> tuple[float, float]:
        """``(θ_min, θ_max)``."""
        return self._theta_range

    @theta_range.setter
    def theta_range(self, theta_range: tuple[float, float]) -> None:
        self._theta_range = finite_range(theta_range, "theta_range")

    @property
    def sampling(self) -> CurveSampling | None:
        """Sampling settings (``None`` for the defaults)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: CurveSampling | None) -> None:
        self._sampling: CurveSampling | None = optional_setting(
            sampling, CurveSampling, "sampling"
        )

    def __call__(self, theta: Any) -> Any:
        """Return the radius at the angle ``theta`` (a number or an array)."""
        return _value(self._r, theta)

    def _render(self, plane: Plane, surface: pygame.Surface) -> None:
        plane.polar(
            surface,
            self._r,
            self._theta_range,
            style=self.style,
            color=self._color(plane),
            sampling=self._sampling,
        )

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        radius = self._r

        def x(theta: Any) -> Any:
            return np.multiply(_value(radius, theta), np.cos(theta))

        def y(theta: Any) -> Any:
            return np.multiply(_value(radius, theta), np.sin(theta))

        return (x, y, self._theta_range)

    def _describe(self) -> str:
        return f"r = {show(self._r)}"


class ImplicitCurve(LineObject):
    """An implicit curve ``F(x, y) = level``, drawn like :meth:`Plane.implicit`.

    Calling it evaluates ``F``: ``F(1.0, 2.0)``.

    Args:
        func: ``F``: a function of ``x`` and ``y``.
        level: The value of ``F`` on the curve, or several values (contour
            lines).
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is found (see
            :class:`~pygridplot.ImplicitSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane).
    """

    def __init__(
        self,
        func: Callable[[Any, Any], Any],
        level: float | Sequence[float] = 0.0,
        *,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: ImplicitSampling | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color, "width": width, "opacity": opacity},
            palette=palette,
            visible=visible,
        )
        self.func = func
        self.level = level
        self.sampling = sampling

    @property
    def func(self) -> Callable[[Any, Any], Any]:
        """``F``: the function of ``x`` and ``y``."""
        return self._func

    @func.setter
    def func(self, func: Callable[[Any, Any], Any]) -> None:
        self._func = function(func, "func")

    @property
    def level(self) -> float | tuple[float, ...]:
        """The value of ``F`` on the curve, or a tuple of values."""
        return self._level

    @level.setter
    def level(self, level: float | Sequence[float]) -> None:
        self._level = _levels(level)

    @property
    def sampling(self) -> ImplicitSampling | None:
        """Sampling settings (``None`` for the defaults)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: ImplicitSampling | None) -> None:
        self._sampling: ImplicitSampling | None = optional_setting(
            sampling, ImplicitSampling, "sampling"
        )

    def __call__(self, x: Any, y: Any) -> Any:
        """Evaluate ``F(x, y)`` (numbers or arrays)."""
        return self._func(x, y)

    def _render(self, plane: Plane, surface: pygame.Surface) -> None:
        plane.implicit(
            surface,
            self._func,
            self._level,
            style=self.style,
            color=self._color(plane),
            sampling=self._sampling,
        )

    def _describe(self) -> str:
        level = self._level
        shown = (
            ", ".join(f"{v:g}" for v in level)
            if isinstance(level, tuple)
            else f"{level:g}"
        )
        return f"{show(self._func)} = {shown}"


def _value(value: Component, at: Any) -> Any:
    """Evaluate a component that may be a number (spread over ``at``'s shape)."""
    if callable(value):
        return value(at)
    return np.full(np.shape(at), float(value)) if np.ndim(at) else float(value)


def _levels(level: object) -> float | tuple[float, ...]:
    """Validate the level(s) of an implicit curve."""
    if is_number(level):
        number = float(level)  # type: ignore[arg-type]
        if not math.isfinite(number):
            msg = f"level must be finite, got {level!r}"
            raise ValueError(msg)
        return number
    try:
        values = tuple(float(v) for v in level)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        msg = f"level must be a number or a sequence of numbers, got {level!r}"
        raise ValueError(msg) from None
    if not values or not all(math.isfinite(v) for v in values):
        msg = f"levels must be finite numbers, at least one, got {level!r}"
        raise ValueError(msg)
    return values
