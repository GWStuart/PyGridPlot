"""The warning issued when an object cannot be drawn.

An object the plane refuses to draw (an invalid setting, say) is skipped,
and the rest of the picture still draws: a frame never breaks because of
one object. :class:`PlotWarning` says so once, when the error appears or
changes, and the object's ``error`` attribute always holds it.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Final

_PACKAGE: Final = str(Path(__file__).resolve().parents[1])
"""The folder of the ``pygridplot`` package, to find the caller outside it."""


class PlotWarning(UserWarning):
    """An object could not be drawn, and was skipped (see its ``error``)."""


def warn(message: str) -> None:
    """Issue a :class:`PlotWarning`, attributed to the program's own call.

    The warning points at the first frame outside PyGridPlot (the program's
    ``plane.draw(...)``), however deep inside the library it was found, so
    the message shows the line to look at.

    Args:
        message: The warning's text.
    """
    level = 2
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    warnings.warn(message, PlotWarning, stacklevel=level)
