"""Framing a view: initial ranges, the home view, and switching axis kinds.

A plane is usually told what to show as world ranges
(``Grid(x_range=..., y_range=...)``), but ranges can only be turned into a
centre and scale once the plane knows its size in pixels. An automatic
plane (one that fills whatever surface it is drawn on) only learns its real
size when it is first drawn: the window may not exist yet, or may still
change size. :class:`ViewFraming` takes care of that:

* A fit requested before the plane is **settled** (has its real size) is
  done provisionally on the current size, and done again once the real size
  is known, unless the view was moved in between.
* The **home view** (what ``reset_view`` and the Home key return to) is the
  initial fit, computed at the real size.
* **Switching an axis** between linear and logarithmic refits that axis,
  updates the equal-aspect setting and makes the result the new home view.

It works on a :class:`~pygridplot.core.view.View` only, so it is pure and
tested directly; :class:`pygridplot.Grid` delegates to it.
"""

from __future__ import annotations

from typing import TypeAlias

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.view import View, ViewState

Range: TypeAlias = tuple[float, float]
"""A world range ``(low, high)``."""

Ranges: TypeAlias = tuple[Range | None, Range | None]
"""An ``(x_range, y_range)`` pair to fit, either of which may be ``None``."""


def aspect_choice(
    equal_aspect: bool | None, x_range: Range | None, y_range: Range | None
) -> bool | None:
    """Return the explicit equal-aspect choice implied by a plane's arguments.

    Args:
        equal_aspect: What the user asked for (``None``: decide automatically).
        x_range: The initial x range, if given.
        y_range: The initial y range, if given.

    Returns:
        ``equal_aspect`` if given. Otherwise ``False`` when both ranges are
        given (each range then fills its own axis exactly), else ``None``
        (decided by the kinds of axes, see :func:`equal_aspect_for`).
    """
    if equal_aspect is None and x_range is not None and y_range is not None:
        return False
    return equal_aspect


def equal_aspect_for(choice: bool | None, log_x: bool, log_y: bool) -> bool:
    """Return whether the axes should share one scale.

    Args:
        choice: The explicit choice (see :func:`aspect_choice`), or ``None``.
        log_x: Whether the x axis is (or is about to be) logarithmic.
        log_y: Whether the y axis is (or is about to be) logarithmic.

    Returns:
        ``choice`` if given, else whether both axes are of the same kind (a
        pixel per unit and a pixel per decade cannot sensibly be equal).
    """
    return choice if choice is not None else log_x == log_y


class ViewFraming:
    """Keeps a view's fits, home view and axis kinds consistent.

    Args:
        view: The view to frame (changed in place).
        choice: The explicit equal-aspect choice (see :func:`aspect_choice`).
        settled: Whether the view already has its real size. ``False`` for
            an automatic plane that has not been drawn yet.
    """

    def __init__(self, view: View, choice: bool | None, *, settled: bool) -> None:
        self._view = view
        self._choice = choice
        self._settled = settled and not view.rect.is_empty
        self._home = view.state()
        self._home_ranges: Ranges | None = None
        self._refit: tuple[Ranges, ViewState] | None = None

    @property
    def home(self) -> ViewState:
        """The view that :meth:`reset` returns to."""
        return self._home

    @home.setter
    def home(self, state: ViewState) -> None:
        self._home = state
        self._home_ranges = None  # an explicit home replaces a pending one

    @property
    def settled(self) -> bool:
        """Whether the view has had its real size since it was created."""
        return self._settled

    def reset(self) -> None:
        """Return the view to the home view."""
        self._view.restore(self._home)

    def fit(self, ranges: Ranges, *, home: bool) -> None:
        """Fit the view to world ranges, now and once its real size is known.

        Args:
            ranges: ``(x_range, y_range)``, either of which may be ``None``.
            home: Whether the fitted view also becomes the home view.

        Raises:
            ValueError: If a range is empty, reversed, not finite, or not
                positive on a log axis. The view is unchanged then.
        """
        validate_ranges(self._view, ranges)
        view = self._view
        if not view.rect.is_empty:
            view.fit(*ranges)
            if home:
                self._home = view.state()
        if not self._settled:
            self._refit = (ranges, view.state())
            if home:
                self._home_ranges = ranges

    def settle(self) -> None:
        """Note that the view has its real size, redoing provisional fits.

        Call it whenever the view's rectangle may have changed. Only the
        first call with a non-empty rectangle does anything. A provisional
        fit is redone only if the view was not moved since (so a pan or zoom
        before the first frame is kept).
        """
        view = self._view
        if self._settled or view.rect.is_empty:
            return
        self._settled = True
        if self._home_ranges is not None:
            probe = view.copy()
            probe.fit(*self._home_ranges)
            self._home = probe.state()
            self._home_ranges = None
        if self._refit is not None:
            ranges, fitted = self._refit
            self._refit = None
            if view.state() == fitted:
                view.fit(*ranges)

    def set_log(self, axis: Axis, log: bool) -> bool:
        """Switch one axis between linear and logarithmic.

        The axis keeps its visible range where possible (see
        :meth:`~pygridplot.core.view.View.set_log`), the equal-aspect setting
        follows :func:`equal_aspect_for`, and the result becomes the home
        view.

        Args:
            axis: ``"x"`` or ``"y"``.
            log: Whether the axis should be logarithmic.

        Returns:
            Whether anything changed.
        """
        view = self._view
        if log == view.axis_transform(axis).is_log:
            return False
        log_x = log if axis == "x" else view.log_x
        log_y = log if axis == "y" else view.log_y
        equal = equal_aspect_for(self._choice, log_x, log_y)
        view.equal_aspect = False  # refit the switched axis on its own
        view.set_log(axis, log)
        view.equal_aspect = equal
        self.home = view.state()
        return True


def validate_ranges(view: View, ranges: Ranges) -> None:
    """Raise if the ranges cannot be fitted on the view's axes.

    The check works on a copy, so ranges can be validated before the view
    has a size.

    Args:
        view: The view the ranges are for.
        ranges: ``(x_range, y_range)``, either of which may be ``None``.

    Raises:
        ValueError: If a range is empty, reversed, not finite, or not
            positive on a log axis.
    """
    probe = view.copy()
    probe.rect = ScreenRect(0, 0, 100, 100)
    probe.fit(*ranges)
