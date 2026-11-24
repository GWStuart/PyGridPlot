"""The :class:`Checkbox`: a switch the user turns on and off with a click.

Checkboxes are for parameters that are switches rather than numbers: show
the derivative, shade the area, use the other rule. Like a
:class:`~pygridplot.core.handles.slider.Slider`, a checkbox lives in a panel
in a corner of the plane (see :mod:`pygridplot.core.handles.panel`),
stacked with the sliders there, and stays put while the plane is panned
and zoomed.

A checkbox is created once, with :meth:`pygridplot.Grid.checkbox`; your
code reads :attr:`Checkbox.value` (a ``bool``) every frame, and
:attr:`Checkbox.changed` says whether it changed since the previous frame.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pygridplot.core.coords import ColorLike
from pygridplot.core.handles.slider import Corner, check_corner


class Checkbox:
    """A switch, on or off, shown as a box to tick in a panel.

    Create checkboxes with :meth:`pygridplot.Grid.checkbox`, which registers
    them with the plane so that :meth:`~pygridplot.Grid.handle_event` toggles
    them and :meth:`~pygridplot.Grid.draw_controls` draws them.

    Args:
        label: The text beside the box.
        value: Whether it starts ticked.
        color: The colour of the ticked box; ``None`` gives it a palette
            colour, in the order the plane's sliders and checkboxes were
            created.
        corner: Which corner of the plane its panel is in.
        enabled: Whether it can be clicked (a disabled checkbox is drawn
            faded).
        on_remove: Called by :meth:`remove` (the plane forgets the checkbox).

    Raises:
        TypeError: If the label is not a string, or the value not a bool.
        ValueError: If the corner is unknown.
    """

    def __init__(
        self,
        label: str,
        *,
        value: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
        on_remove: Callable[[Checkbox], None] | None = None,
    ) -> None:
        self.label = _check_label(label)
        """The text beside the box."""
        self.color = color
        """The ticked box's colour, or ``None`` for the checkbox's palette colour."""
        self.corner = corner
        self.enabled = enabled
        """Whether it can be clicked (it is drawn faded when not)."""
        self._value = _check_value(value)
        self._on_remove = on_remove
        self._version = 0
        self._seen_version = 0
        self._changed = False
        self._hovered = False

    @property
    def value(self) -> bool:
        """Whether the box is ticked. Assign to tick or untick it."""
        return self._value

    @value.setter
    def value(self, value: bool) -> None:
        checked = _check_value(value)
        if checked != self._value:
            self._value = checked
            self._version += 1

    @property
    def corner(self) -> Corner:
        """Which corner of the plane the checkbox's panel is in."""
        return self._corner

    @corner.setter
    def corner(self, corner: Corner) -> None:
        self._corner = check_corner(corner)

    @property
    def changed(self) -> bool:
        """Whether the value changed between the previous frame and this one.

        Frames start at :meth:`pygridplot.Grid.draw` (or
        :meth:`~pygridplot.Grid.begin_frame`), so this stays the same while
        you draw, like a slider's :attr:`~pygridplot.Slider.moved`.
        """
        return self._changed

    @property
    def hovered(self) -> bool:
        """Whether the cursor is over the checkbox (and it can be clicked)."""
        return self._hovered

    def set_interaction(self, *, hovered: bool) -> None:
        """Record whether the checkbox is hovered (by the plane).

        Args:
            hovered: Whether the cursor is over it.
        """
        self._hovered = hovered

    def begin_frame(self) -> None:
        """Start a new frame: update :attr:`changed`.

        :meth:`pygridplot.Grid.begin_frame` calls this for every checkbox.
        """
        self._changed = self._version != self._seen_version
        self._seen_version = self._version

    def remove(self) -> None:
        """Remove the checkbox from its plane: it is no longer clicked or drawn."""
        self._hovered = False
        if self._on_remove is not None:
            callback, self._on_remove = self._on_remove, None
            callback(self)

    def __repr__(self) -> str:
        """Return a short description with the value."""
        return f"Checkbox({self.label!r}, {self._value!r})"


def _check_label(label: object) -> str:
    """Return a checkbox's label.

    Raises:
        TypeError: If it is not a string.
    """
    if not isinstance(label, str):
        msg = f"the label must be a string, got {label!r}"
        raise TypeError(msg)
    return label


def _check_value(value: object) -> bool:
    """Return a checkbox's value as a bool.

    Raises:
        TypeError: If it is not a bool (or a NumPy bool).
    """
    if not isinstance(value, (bool, np.bool_)):
        msg = f"a checkbox's value must be True or False, got {value!r}"
        raise TypeError(msg)
    return bool(value)
