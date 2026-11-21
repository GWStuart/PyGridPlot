"""Items drawn as curves: plots, parametric, polar and implicit curves."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from pygridplot.core.expr import (
    Binding,
    FunctionBinding,
)
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import (
    LineStyle,
)
from pygridplot.graph.items.base import (
    FULL_TURN,
    Expression,
    Formula,
    LineItem,
    Maker,
    component,
    finite_range,
    is_number,
    optional_setting,
    show,
)


class PlotItem(LineItem):
    """The graph of a function, ``y = f(x)``, drawn like :meth:`Plane.plot`.

    A named plot also defines the function, so other items can call it:
    after ``graph.plot("x^2", name="f")``, ``"f(x - 1)"`` and ``"f'(x)"``
    work anywhere.

    Args:
        expression: ``f`` as text in ``x`` (``"a sin(x)"``) or a function.
        name: The function's name (optional).
        domain: Draw only for ``x`` in ``(start, end)``.
        color: Line colour (default: the item's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`; the other arguments
            override its fields.
        sampling: How the function is sampled (see
            :class:`~pygridplot.SamplingConfig`); default: the plane's.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
    """

    kind: ClassVar[str] = "plot"

    def __init__(
        self,
        expression: Expression,
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
        self._expression = Formula.make(
            expression, "a plot's expression", functions=True
        )
        self._domain = None if domain is None else finite_range(domain, "domain")
        self._sampling: SamplingConfig | None = optional_setting(
            sampling, SamplingConfig, "sampling"
        )

    @property
    def expression(self) -> Expression:
        """``f``: the text in ``x``, or the Python function."""
        return self._expression.source  # type: ignore[return-value]

    @expression.setter
    def expression(self, expression: Expression) -> None:
        self._expression = Formula.make(
            expression, "a plot's expression", functions=True
        )
        self._changed()

    @property
    def domain(self) -> tuple[float, float] | None:
        """The ``x`` interval drawn (``None`` for everywhere)."""
        return self._domain

    @domain.setter
    def domain(self, domain: tuple[float, float] | None) -> None:
        self._domain = None if domain is None else finite_range(domain, "domain")
        self._changed()

    @property
    def sampling(self) -> SamplingConfig | None:
        """Sampling settings (``None`` for the plane's)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: SamplingConfig | None) -> None:
        self._sampling = optional_setting(sampling, SamplingConfig, "sampling")

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._expression, ("x",))]

    def _runtime(self, make: Maker) -> Any:
        return make(self._expression, ("x",))

    def _binding(self, runtime: Any) -> Binding | None:
        return FunctionBinding(runtime, 1)

    def _describe(self) -> str:
        return f"y = {show(self._expression)}"


class ParametricItem(LineItem):
    """A parametric curve ``(x(t), y(t))``, drawn like :meth:`Plane.parametric`.

    A named curve can be followed by a draggable point
    (``graph.draggable(curve="C")``).

    Args:
        x: ``x(t)``: text in ``t``, a function or a number.
        y: ``y(t)``: likewise.
        t_range: ``(t_min, t_max)``; one full turn by default.
        name: The curve's name (optional).
        color: Line colour (default: the item's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
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
        """``x(t)``: text, a function or a number."""
        return self._x.source

    @x.setter
    def x(self, x: Expression | float) -> None:
        self._x = component(x, "x(t)")
        self._changed()

    @property
    def y(self) -> Expression | float:
        """``y(t)``: text, a function or a number."""
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
        self._changed()

    @property
    def sampling(self) -> CurveSampling | None:
        """Sampling settings (``None`` for the defaults)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: CurveSampling | None) -> None:
        self._sampling = optional_setting(sampling, CurveSampling, "sampling")

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._x, ("t",)), (self._y, ("t",))]

    def _runtime(self, make: Maker) -> Any:
        return (make(self._x, ("t",)), make(self._y, ("t",)))

    def _describe(self) -> str:
        return f"({show(self._x)}, {show(self._y)})"


class PolarItem(LineItem):
    """A polar curve ``r = r(θ)``, drawn like :meth:`Plane.polar`.

    The angle is written ``θ`` or ``theta``. A named polar curve also
    defines the function ``r(θ)`` for other items, and can be followed by a
    draggable point.

    Args:
        r: ``r(θ)``: text in ``θ``, a function or a number.
        theta_range: ``(θ_min, θ_max)``; one full turn by default.
        name: The curve's name (optional).
        color: Line colour (default: the item's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is sampled (see
            :class:`~pygridplot.CurveSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
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
        """``r(θ)``: text, a function or a number."""
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
        self._changed()

    @property
    def sampling(self) -> CurveSampling | None:
        """Sampling settings (``None`` for the defaults)."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: CurveSampling | None) -> None:
        self._sampling = optional_setting(sampling, CurveSampling, "sampling")

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

    def _describe(self) -> str:
        return f"r = {show(self._r)}"


class ImplicitItem(LineItem):
    """An implicit curve ``F(x, y) = level``, drawn like :meth:`Plane.implicit`.

    A named implicit curve also defines ``F(x, y)`` for other items.

    Args:
        expression: ``F``: text in ``x`` and ``y`` or a function of both.
        level: The value of ``F`` on the curve, or several values (contour
            lines).
        name: ``F``'s name (optional).
        color: Line colour (default: the item's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        sampling: How the curve is found (see
            :class:`~pygridplot.ImplicitSampling`).
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
    """

    kind: ClassVar[str] = "implicit"

    def __init__(
        self,
        expression: Expression,
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
        self._expression = Formula.make(
            expression, "an implicit curve's expression", functions=True
        )
        self._level = _levels(level)
        self._sampling: ImplicitSampling | None = optional_setting(
            sampling, ImplicitSampling, "sampling"
        )

    @property
    def expression(self) -> Expression:
        """``F``: the text in ``x`` and ``y``, or the Python function."""
        return self._expression.source  # type: ignore[return-value]

    @expression.setter
    def expression(self, expression: Expression) -> None:
        self._expression = Formula.make(
            expression, "an implicit curve's expression", functions=True
        )
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

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._expression, ("x", "y"))]

    def _runtime(self, make: Maker) -> Any:
        return make(self._expression, ("x", "y"))

    def _binding(self, runtime: Any) -> Binding | None:
        return FunctionBinding(runtime, 2)

    def _describe(self) -> str:
        level = self._level
        shown = (
            ", ".join(f"{v:g}" for v in level)
            if isinstance(level, tuple)
            else f"{level:g}"
        )
        return f"{show(self._expression)} = {shown}"


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
