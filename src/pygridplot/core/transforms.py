"""Axis transforms: linear and logarithmic axes.

Every axis maps world values onto a straight *transformed* coordinate, and
the view lays that coordinate out evenly on screen:

* a **linear** axis uses the values themselves;
* a **logarithmic** axis uses ``log10(value)``, so every decade gets the same
  space on screen.

The :class:`~pygridplot.core.view.View` keeps its centre and scale in
transformed coordinates (so the scale of a log axis is in pixels per
decade). Panning, zooming and fitting therefore work the same way on both
kinds of axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

TransformKind: TypeAlias = Literal["linear", "log"]
"""The kinds of axis transform."""

MAX_LOG_DECADE: Final = 300.0
"""Largest ``|log10(value)|`` a log axis centre may reach.

``10 ** 308`` is about the largest float, so this keeps every value near the
middle of the plane representable.
"""


@dataclass(frozen=True, slots=True)
class AxisTransform:
    """How world values along one axis map onto its transformed coordinate.

    Attributes:
        kind: ``"linear"`` or ``"log"`` (base 10).
    """

    kind: TransformKind = "linear"

    def __post_init__(self) -> None:
        """Validate the kind."""
        if self.kind not in ("linear", "log"):
            msg = f"axis transform must be 'linear' or 'log', got {self.kind!r}"
            raise ValueError(msg)

    @property
    def is_log(self) -> bool:
        """Whether this is a logarithmic axis."""
        return self.kind == "log"

    @property
    def default_center(self) -> float:
        """The natural centre of a new view, in world units (0, or 1 on a log axis)."""
        return 1.0 if self.is_log else 0.0

    @overload
    def forward(self, values: float) -> float: ...
    @overload
    def forward(self, values: ArrayLike) -> NDArray[np.float64]: ...
    def forward(self, values: ArrayLike) -> object:
        """Convert world values to transformed coordinates.

        Args:
            values: A number or an array of world values.

        Returns:
            The values themselves on a linear axis. On a log axis, their
            base-10 logarithm, which is NaN for negative values and ``-inf``
            for zero (neither can be shown on a log axis).
        """
        if not self.is_log:
            if isinstance(values, (int, float)):
                return float(values)
            return np.asarray(values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            if isinstance(values, (int, float)):
                value = float(values)
                if value > 0:
                    return math.log10(value)
                return -math.inf if value == 0 else math.nan
            return np.log10(np.asarray(values, dtype=np.float64))

    @overload
    def inverse(self, values: float) -> float: ...
    @overload
    def inverse(self, values: ArrayLike) -> NDArray[np.float64]: ...
    def inverse(self, values: ArrayLike) -> object:
        """Convert transformed coordinates back to world values.

        Args:
            values: A number or an array of transformed coordinates.

        Returns:
            World values (``10 ** u`` on a log axis, ``inf`` where too large).
        """
        if not self.is_log:
            if isinstance(values, (int, float)):
                return float(values)
            return np.asarray(values, dtype=np.float64)
        with np.errstate(over="ignore"):
            if isinstance(values, (int, float)):
                exponent = float(values)
                return math.inf if exponent > 308.25 else 10.0**exponent
            return np.power(10.0, np.asarray(values, dtype=np.float64))

    def check(self, name: str, value: float) -> float:
        """Return the transformed coordinate of a world value, or raise if invalid.

        Args:
            name: Description of the value, for the error message.
            value: A world value.

        Returns:
            Its transformed coordinate.

        Raises:
            ValueError: If the value is not finite, or not positive on a log
                axis.
        """
        transformed = self.forward(float(value))
        if not math.isfinite(transformed):
            need = "a positive number on a log axis" if self.is_log else "finite"
            msg = f"{name} must be {need}, got {value!r}"
            raise ValueError(msg)
        return transformed


LINEAR: Final = AxisTransform("linear")
"""A linear axis."""

LOG: Final = AxisTransform("log")
"""A base-10 logarithmic axis."""


def transform_for(log: bool) -> AxisTransform:
    """Return the transform for an axis that is (or is not) logarithmic.

    Args:
        log: Whether the axis is logarithmic.

    Returns:
        :data:`LOG` or :data:`LINEAR`.
    """
    return LOG if log else LINEAR
