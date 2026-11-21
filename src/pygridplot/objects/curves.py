"""Objects drawn as curves: functions, parametric, polar and implicit curves.

:class:`Function`, :class:`ParametricCurve` and :class:`PolarCurve` are
*curve objects*: a draggable :class:`~pygridplot.Point` can follow them
(``pgp.Point(curve=f, draggable=True)``), and they can be called like the
functions they draw (``f(2.0)``, ``pgp.derivative(f)``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np

from pygridplot.core.expr import Binding, FunctionBinding
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.derivatives import Derivative, derivative
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import LineStyle
from pygridplot.objects.base import LineObject
from pygridplot.objects.values import (
    FULL_TURN,
    Expression,
    Formula,
    Maker,
    component,
    finite_range,
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
        """Return the curve to follow, for a view and the point's parameter.

        Raises:
            ValueError: If the curve has an error.
        """
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
    differentiated (``pgp.derivative(f)``, or :meth:`derivative`). In a
    graph, a named function can also be used by name: ``"f(x - 1)"``,
    ``"f'(x)"``.

    Args:
        func: ``f``: a Python function of ``x`` (NumPy or ``math``), or text
            in ``x`` (``"a sin(x)"``, see the expression language).
        name: The function's name (optional).
        domain: Draw only for ``x`` in ``(start, end)``.
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`; the other arguments
            override its fields.
        sampling: How the function is sampled (see
            :class:`~pygridplot.SamplingConfig`); default: the plane's.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "plot"
    curve_from_zero: ClassVar[bool] = True

    def __init__(
        self,
        func: Expression,
        *,
        name: str | None = None,
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
            name=name,
            visible=visible,
        )
        self._func = Formula.make(func, "a function", functions=True)
        self._domain = None if domain is None else finite_range(domain, "domain")
        self._sampling: SamplingConfig | None = optional_setting(
            sampling, SamplingConfig, "sampling"
        )

    @property
    def func(self) -> Expression:
        """``f``: the Python function, or the text in ``x``."""
        return self._func.source  # type: ignore[return-value]

    @func.setter
    def func(self, func: Expression) -> None:
        self._func = Formula.make(func, "a function", functions=True)
        self._changed()

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
        self._sampling = optional_setting(sampling, SamplingConfig, "sampling")

    def __call__(self, x: Any) -> Any:
        """Evaluate the function at ``x`` (a number or an array).

        Raises:
            ValueError: If the function has an error (see :attr:`error`).
        """
        return self._working()(x)

    def derivative(self, n: int = 1) -> Derivative:
        """Return the ``n``-th derivative (exact for NumPy functions and text).

        It follows the function: changing :attr:`func` changes it too.

        Args:
            n: The order.

        Returns:
            The derivative, a function of ``x``.
        """
        return derivative(self, n)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._func, ("x",))]

    def _runtime(self, make: Maker) -> Any:
        return make(self._func, ("x",))

    def _binding(self, runtime: Any) -> Binding | None:
        return FunctionBinding(runtime, 1)

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.plot(
            surface,
            runtime,
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
        runtime = self._working()
        if self._domain is not None:
            return (_identity, runtime, self._domain)
        bounds = view.bounds
        if view.log_x:
            low, high = bounds.x_min / 10.0, bounds.x_max * 10.0
        else:
            width = bounds.x_max - bounds.x_min
            low, high = bounds.x_min - width, bounds.x_max + width
        if t is not None and math.isfinite(t):
            low, high = min(low, t - 1.0), max(high, t + 1.0)
        return (_identity, runtime, (low, high))

    def _describe(self) -> str:
        return f"y = {show(self._func)}"


class ParametricCurve(CurveObject):
    """A parametric curve ``(x(t), y(t))``, drawn like :meth:`Plane.parametric`.

    Calling it gives the point at a parameter: ``c(0.5)`` is ``(x, y)``.

    Args:
        x: ``x(t)``: a Python function of ``t``, text in ``t``, or a number.
        y: ``y(t)``: likewise.
        t_range: ``(t_min, t_max)``; one full turn by default.
        name: The curve's name (optional).
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "parametric"

    def __init__(
        self,
        x: Expression | float,
        y: Expression | float,
        t_range: tuple[float, float] = FULL_TURN,
        *,
        name: str | None = None,
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
            name=name,
            visible=visible,
        )
        self._x = component(x, "x(t)")
        self._y = component(y, "y(t)")
        self._t_range = finite_range(t_range, "t_range")
        self._sampling: CurveSampling | None = optional_setting(
            sampling, CurveSampling, "sampling"
        )

    @property
    def x(self) -> Expression | float:
        """``x(t)``: a function, text or a number."""
        return self._x.source

    @x.setter
    def x(self, x: Expression | float) -> None:
        self._x = component(x, "x(t)")
        self._changed()

    @property
    def y(self) -> Expression | float:
        """``y(t)``: a function, text or a number."""
        return self._y.source

    @y.setter
    def y(self, y: Expression | float) -> None:
        self._y = component(y, "y(t)")
        self._changed()

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
        self._sampling = optional_setting(sampling, CurveSampling, "sampling")

    def __call__(self, t: Any) -> tuple[Any, Any]:
        """Return the point ``(x(t), y(t))`` (numbers, or arrays for an array ``t``).

        Raises:
            ValueError: If the curve has an error (see :attr:`error`).
        """
        fx, fy = self._working()
        return (_value(fx, t), _value(fy, t))

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._x, ("t",)), (self._y, ("t",))]

    def _runtime(self, make: Maker) -> Any:
        return (make(self._x, ("t",)), make(self._y, ("t",)))

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        fx, fy = runtime
        plane.parametric(
            surface,
            fx,
            fy,
            self._t_range,
            style=self.style,
            color=self._color(plane),
            sampling=self._sampling,
        )

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        fx, fy = self._working()
        return (fx, fy, self._t_range)

    def _describe(self) -> str:
        return f"({show(self._x)}, {show(self._y)})"


class PolarCurve(CurveObject):
    """A polar curve ``r = r(θ)``, drawn like :meth:`Plane.polar`.

    Calling it gives the radius at an angle: ``R(math.pi)``. In text, the
    angle is written ``θ`` or ``theta``.

    Args:
        r: ``r(θ)``: a Python function of ``θ``, text in ``θ``, or a number.
        theta_range: ``(θ_min, θ_max)``; one full turn by default.
        name: The curve's name (optional).
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "polar"
    VARIABLES: ClassVar[tuple[str, ...]] = ("θ", "theta")

    def __init__(
        self,
        r: Expression | float,
        theta_range: tuple[float, float] = FULL_TURN,
        *,
        name: str | None = None,
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
            name=name,
            visible=visible,
        )
        self._r = component(r, "r(θ)")
        self._theta_range = finite_range(theta_range, "theta_range")
        self._sampling: CurveSampling | None = optional_setting(
            sampling, CurveSampling, "sampling"
        )

    @property
    def r(self) -> Expression | float:
        """``r(θ)``: a function, text or a number."""
        return self._r.source

    @r.setter
    def r(self, r: Expression | float) -> None:
        self._r = component(r, "r(θ)")
        self._changed()

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
        self._sampling = optional_setting(sampling, CurveSampling, "sampling")

    def __call__(self, theta: Any) -> Any:
        """Return the radius at the angle ``theta`` (a number or an array).

        Raises:
            ValueError: If the curve has an error (see :attr:`error`).
        """
        return _value(self._working(), theta)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._r, self.VARIABLES)]

    def _runtime(self, make: Maker) -> Any:
        compiled = make(self._r, self.VARIABLES)
        if not callable(compiled) or not self._r.is_text:
            return compiled  # a number, or a Python function of θ
        return lambda theta: compiled(theta, theta)

    def _binding(self, runtime: Any) -> Binding | None:
        if callable(runtime):
            return FunctionBinding(runtime, 1)
        value = float(runtime)
        return FunctionBinding(lambda theta: np.full(np.shape(theta), value), 1)

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.polar(
            surface,
            runtime,
            self._theta_range,
            style=self.style,
            color=self._color(plane),
            sampling=self._sampling,
        )

    def _curve_spec(self, view: View, t: float | None) -> CurveSpec:
        radius = self._working()

        def x(theta: Any) -> Any:
            return np.multiply(_value(radius, theta), np.cos(theta))

        def y(theta: Any) -> Any:
            return np.multiply(_value(radius, theta), np.sin(theta))

        return (x, y, self._theta_range)

    def _describe(self) -> str:
        return f"r = {show(self._r)}"


class ImplicitCurve(LineObject):
    """An implicit curve ``F(x, y) = level``, drawn like :meth:`Plane.implicit`.

    Calling it evaluates ``F``: ``F(1.0, 2.0)``. A named implicit curve in a
    graph also defines ``F(x, y)`` for other expressions.

    Args:
        func: ``F``: a Python function of ``x`` and ``y``, or text in both.
        level: The value of ``F`` on the curve, or several values (contour
            lines).
        name: ``F``'s name (optional).
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is found (see
            :class:`~pygridplot.ImplicitSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "implicit"

    def __init__(
        self,
        func: Expression,
        level: float | Sequence[float] = 0.0,
        *,
        name: str | None = None,
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
            name=name,
            visible=visible,
        )
        self._func = Formula.make(func, "an implicit curve's function", functions=True)
        self._level = _levels(level)
        self._sampling: ImplicitSampling | None = optional_setting(
            sampling, ImplicitSampling, "sampling"
        )

    @property
    def func(self) -> Expression:
        """``F``: the Python function, or the text in ``x`` and ``y``."""
        return self._func.source  # type: ignore[return-value]

    @func.setter
    def func(self, func: Expression) -> None:
        self._func = Formula.make(func, "an implicit curve's function", functions=True)
        self._changed()

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
        self._sampling = optional_setting(sampling, ImplicitSampling, "sampling")

    def __call__(self, x: Any, y: Any) -> Any:
        """Evaluate ``F(x, y)`` (numbers or arrays).

        Raises:
            ValueError: If the curve has an error (see :attr:`error`).
        """
        return self._working()(x, y)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._func, ("x", "y"))]

    def _runtime(self, make: Maker) -> Any:
        return make(self._func, ("x", "y"))

    def _binding(self, runtime: Any) -> Binding | None:
        return FunctionBinding(runtime, 2)

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.implicit(
            surface,
            runtime,
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


def _value(compiled: Any, at: Any) -> Any:
    """Evaluate a component that may be a number (spread over ``at``'s shape)."""
    if callable(compiled):
        return compiled(at)
    return np.full(np.shape(at), float(compiled)) if np.ndim(at) else float(compiled)


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
