"""What a graph is meant to be seen at: the ranges and the kinds of axes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygridplot.plane import Plane


@dataclass(frozen=True, slots=True)
class GraphView:
    """A view to show a graph at, saved with it.

    Ranges rather than a centre and scale are kept, so a graph opened in a
    window of another size shows the same part of the plane.

    Attributes:
        x_range: ``(x_min, x_max)``.
        y_range: ``(y_min, y_max)``.
        log_x: Whether the x axis is logarithmic.
        log_y: Whether the y axis is logarithmic.
        equal_aspect: Whether both axes keep the same scale (``None`` to
            leave the plane's setting alone). With equal aspect, the plane
            shows at least both ranges.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    log_x: bool = False
    log_y: bool = False
    equal_aspect: bool | None = None

    def __post_init__(self) -> None:
        """Validate the ranges.

        Raises:
            ValueError: If a range is not an increasing pair of finite
                numbers, or not positive on a log axis.
        """
        for name, span, log in (
            ("x_range", self.x_range, self.log_x),
            ("y_range", self.y_range, self.log_y),
        ):
            low, high = float(span[0]), float(span[1])
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                msg = f"{name} must be increasing and finite, got {span!r}"
                raise ValueError(msg)
            if log and low <= 0:
                msg = f"{name} must be positive on a log axis, got {span!r}"
                raise ValueError(msg)
            object.__setattr__(self, name, (low, high))

    @classmethod
    def of(cls, plane: Plane) -> GraphView:
        """Return what a plane shows now.

        Applying the result to a plane of the same size gives exactly this
        view again (see :meth:`View.fitted_ranges
        <pygridplot.View.fitted_ranges>`).

        Args:
            plane: The plane.

        Returns:
            Its visible ranges and kinds of axes.
        """
        view = plane.view
        x_range, y_range = view.fitted_ranges()
        return cls(x_range, y_range, view.log_x, view.log_y, view.equal_aspect)

    def apply(self, plane: Plane) -> None:
        """Make a plane show this view.

        Args:
            plane: The plane.
        """
        plane.log_x = self.log_x
        plane.log_y = self.log_y
        if self.equal_aspect is not None:
            plane.view.equal_aspect = self.equal_aspect
        plane.set_view(self.x_range, self.y_range)
