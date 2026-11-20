"""The interface every rendering backend implements.

The rest of the library talks to backends only through :class:`Renderer`.
To add a backend (OpenGL, SVG export, ...), implement these methods for
that backend's surface type. Nothing in :mod:`pygridplot.core` has to
change.

The interface grows one milestone at a time. Polylines, points and
polygons will be added when plotting functions and shapes is implemented.
"""

from __future__ import annotations

from typing import Any, Protocol

from pygridplot.core.grid import GridScene
from pygridplot.core.labels import LabelText
from pygridplot.core.primitives import FontSpec, TextSize


class Renderer(Protocol):
    """A backend that can measure labels and draw grid scenes."""

    def measure_text(self, text: LabelText, font: FontSpec) -> TextSize:
        """Return the size ``text`` will have when drawn with ``font``.

        Args:
            text: The label to measure.
            font: The font it will be drawn with.

        Returns:
            The label's digit-box size.
        """
        ...

    def draw_grid(self, surface: Any, scene: GridScene) -> None:
        """Draw a complete grid scene onto ``surface``.

        Args:
            surface: The backend's drawing target.
            scene: The grid to draw.
        """
        ...
