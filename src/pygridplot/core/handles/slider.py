"""The :class:`Slider`: a value the user sets by dragging a handle along a track.

Sliders are for parameters that are not positions in the plane: an
amplitude, the number of terms of a series, a rate. Unlike a
:class:`~pygridplot.core.handles.point.DraggablePoint`, a slider lives on
the screen, in a panel in a corner of the plane (see
:mod:`pygridplot.core.handles.panel`), so it stays put while the plane is
panned and zoomed, and it shows its label and value.

A slider is created once, with :meth:`pygridplot.Grid.slider`; your code
reads :attr:`Slider.value` every frame. The value is kept within the
slider's range, and is exact:

* **with a step**, the value is ``low + k · step`` for a whole number ``k``,
  computed in decimal so that a step of 0.1 gives 0.3, never
  0.30000000000000004. A step given as an ``int`` (``step=1``) makes an
  integer slider, whose value is an ``int``, ready for :func:`range` or
  :func:`pygridplot.taylor`;
* **without a step**, a dragged value is rounded to the size of one pixel
  of the track (the smallest change a drag can make), as the coordinates of
  draggable points are: ``1.23`` rather than ``1.2345678``;
* **on a log slider** (``log=True``), the track is geometric: equal
  distances along it multiply the value by equal factors, for parameters
  spanning several powers of ten.
"""

from __future__ import annotations

import decimal
import math
import numbers
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Final, Generic, Literal, TypeAlias, TypeVar, cast

from pygridplot.core.coords import ColorLike
from pygridplot.core.grid.formatting import NumberFormat, format_value
from pygridplot.core.handles.labels import rounded_value
from pygridplot.core.primitives import LabelText, TextRun
from pygridplot.core.transforms import AxisTransform

Corner: TypeAlias = Literal["bottom-left", "bottom-right", "top-left", "top-right"]
"""Where in the plane a slider's panel sits."""

CORNERS: tuple[Corner, ...] = ("bottom-left", "bottom-right", "top-left", "top-right")
"""The allowed values of :attr:`Slider.corner`."""

ValueT = TypeVar("ValueT", int, float)
"""The type of a slider's value: ``int`` for an integer slider, else ``float``."""

WHEEL_FRACTION: Final = 0.01
"""How far along the track one notch of the wheel moves a slider without a step."""

_CONTEXT: Final = decimal.Context(prec=60)
"""Enough digits for exact steps (see :meth:`Slider._snap`)."""


class Slider(Generic[ValueT]):
    """A value set by dragging a handle along a track, shown in a panel.

    Create sliders with :meth:`pygridplot.Grid.slider`, which registers them
    with the plane so that :meth:`~pygridplot.Grid.handle_event` drags them
    and :meth:`~pygridplot.Grid.draw_controls` draws them.

    Args:
        label: The name shown before the value (``"a"`` shows ``a = 1.5``).
        value_range: ``(low, high)``, the values at the ends of the track.
        value: The starting value (default: ``low``), kept within the range
            and on the step.
        step: The spacing of the allowed values, from ``low``; an ``int``
            makes an integer slider. ``None`` (the default) for any value.
        log: Make the track geometric (``low`` must then be positive).
        color: The colour of the handle and of the track up to it; ``None``
            gives each slider its own palette colour, in the order the
            sliders were created.
        corner: Which corner of the plane the slider's panel is in.
        enabled: Whether the slider can be dragged (a disabled slider is
            drawn faded).
        on_remove: Called by :meth:`remove` (the plane forgets the slider).

    Raises:
        TypeError: If the label is not a string, or the range, value or step
            are not numbers.
        ValueError: If the range is not two finite numbers in increasing
            order (positive for a log slider), the step is not positive or
            wider than the range, an integer step has ends that are not
            whole numbers, the value is NaN, or the corner is unknown.
    """

    def __init__(
        self,
        label: str,
        value_range: tuple[float, float],
        *,
        value: float | None = None,
        step: float | None = None,
        log: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
        on_remove: Callable[[Slider[Any]], None] | None = None,
    ) -> None:
        if not isinstance(label, str):
            msg = f"the label must be a string, got {label!r}"
            raise TypeError(msg)
        low, high = _check_range(value_range, log=log)
        self._integer = _is_int(step)
        self._step = _check_step(step, low, high, integer=self._integer)
        self._range = (low, high)
        self._log = bool(log)
        self.label = label
        """The name shown before the value."""
        self.color = color
        """The handle's colour, or ``None`` for the slider's palette colour."""
        self.corner = corner
        self.enabled = enabled
        """Whether the slider can be dragged (it is drawn faded when not)."""
        self._on_remove = on_remove
        self._value = self._snap(low if value is None else _number("value", value))
        self._version = 0
        self._seen_version = 0
        self._moved = False
        self._hovered = False
        self._dragging = False
        self.track_px = 0.0
        """The length of the track as last drawn, in pixels (0 before then)."""

    # ------------------------------------------------------------------
    # The value
    # ------------------------------------------------------------------
    @property
    def value(self) -> ValueT:
        """The value: within the range, on the step (an ``int`` for an integer slider).

        Setting it moves the handle; a value outside the range is clamped,
        and one between steps goes to the nearest step.
        """
        return int(self._value) if self._integer else self._value  # type: ignore[return-value]

    @value.setter
    def value(self, value: float) -> None:
        self._set(self._snap(_number("value", value)))

    @property
    def value_range(self) -> tuple[float, float]:
        """``(low, high)``, the values at the ends of the track."""
        return self._range

    @property
    def step(self) -> float | None:
        """The spacing of the allowed values, or ``None`` for any value."""
        return self._step

    @property
    def log(self) -> bool:
        """Whether the track is geometric."""
        return self._log

    @property
    def integer(self) -> bool:
        """Whether the value is always a whole number (an ``int``)."""
        return self._integer

    @property
    def corner(self) -> Corner:
        """Which corner of the plane the slider's panel is in."""
        return self._corner

    @corner.setter
    def corner(self, corner: Corner) -> None:
        if corner not in CORNERS:
            msg = f"corner must be one of {CORNERS}, got {corner!r}"
            raise ValueError(msg)
        self._corner = corner

    @property
    def fraction(self) -> float:
        """How far along the track the handle is, from 0 (``low``) to 1 (``high``)."""
        low, high = self._range
        if self._log:
            return math.log(self._value / low) / math.log(high / low)
        return (self._value - low) / (high - low)

    def value_at(self, fraction: float) -> float:
        """Return the value at a point of the track, as a drag would set it.

        Args:
            fraction: How far along the track, from 0 to 1 (clamped).

        Returns:
            The value: on the step, or rounded to the size of a pixel of the
            track (see :attr:`track_px`).
        """
        t = min(max(float(fraction), 0.0), 1.0)
        low, high = self._range
        raw = low * (high / low) ** t if self._log else low + t * (high - low)
        if self._step is None and self.track_px > 0:
            raw = float(rounded_value(raw, self._transform, self._scale))
        return self._snap(raw)

    def drag_to(self, fraction: float) -> bool:
        """Move the handle to a point of the track, as a drag does.

        Args:
            fraction: How far along the track, from 0 to 1 (clamped).

        Returns:
            Whether the value changed.
        """
        return self._set(self.value_at(fraction))

    def nudge(self, notches: int) -> bool:
        """Move the value by whole steps, as the mouse wheel does.

        Args:
            notches: How many steps up (negative for down). Without a step,
                each is :data:`WHEEL_FRACTION` of the track.

        Returns:
            Whether the value changed.
        """
        if self._step is not None:
            return self._set(self._snap(self._value + notches * self._step))
        return self._set(self.value_at(self.fraction + notches * WHEEL_FRACTION))

    def text(self, fmt: NumberFormat) -> LabelText:
        """Return what the slider shows: ``label = value``.

        The value is written like a tick label, with as many digits as the
        step has (at most two beyond its first: ``π/12`` shows ``0.262``), or
        as one pixel of the track resolves.

        Args:
            fmt: How numbers are written (the theme's number format).

        Returns:
            The text.
        """
        if self._integer:
            shown = Decimal(int(self._value))
        elif self._step is not None:
            shown = Decimal(repr(self._value)).quantize(
                self._step_quantum(), context=_CONTEXT
            )
        elif self.track_px > 0:
            shown = rounded_value(self._value, self._transform, self._scale)
        else:
            shown = Decimal(repr(self._value))
        runs = [TextRun(f"{self.label} = ")] if self.label else []
        for run in format_value(shown, fmt).runs:
            role = "text" if run.role == "sign" and runs else run.role
            runs.append(TextRun(run.text, role))
        return LabelText(tuple(runs))

    # ------------------------------------------------------------------
    # State (used by the plane)
    # ------------------------------------------------------------------
    @property
    def moved(self) -> bool:
        """Whether the value changed between the previous frame and this one.

        Frames start at :meth:`pygridplot.Grid.draw` (or
        :meth:`~pygridplot.Grid.begin_frame`), so the value stays the same
        while you draw; use it to recompute something only when needed.
        """
        return self._moved

    @property
    def dragging(self) -> bool:
        """Whether the slider is being dragged."""
        return self._dragging

    @property
    def hovered(self) -> bool:
        """Whether the cursor is over the slider's track (and it can be dragged)."""
        return self._hovered

    def set_interaction(self, *, hovered: bool, dragging: bool) -> None:
        """Record whether the slider is hovered and dragged (by the plane).

        Args:
            hovered: Whether the cursor is over the track.
            dragging: Whether the handle is being dragged.
        """
        self._hovered = hovered
        self._dragging = dragging

    def begin_frame(self) -> None:
        """Start a new frame: update :attr:`moved`.

        :meth:`pygridplot.Grid.begin_frame` calls this for every slider.
        """
        self._moved = self._version != self._seen_version
        self._seen_version = self._version

    def remove(self) -> None:
        """Remove the slider from its plane: it is no longer dragged or drawn."""
        self._dragging = self._hovered = False
        if self._on_remove is not None:
            callback, self._on_remove = self._on_remove, None
            callback(self)

    def __repr__(self) -> str:
        """Return a short description with the value."""
        return f"Slider({self.label!r}, {self.value!r}, value_range={self._range!r})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @property
    def _transform(self) -> AxisTransform:
        """The track's transform: linear, or log10 for a log slider."""
        return AxisTransform("log" if self._log else "linear")

    @property
    def _scale(self) -> float:
        """Pixels of track per transformed unit (per decade on a log slider)."""
        low, high = self._range
        width = math.log10(high / low) if self._log else high - low
        return self.track_px / width

    def _step_quantum(self) -> Decimal:
        """The precision a stepped value is shown to.

        As many decimals as the step has (``0.25`` shows ``1.25``), but at
        most two digits beyond the step's first, for a step such as ``π/12``
        whose decimals never end (it shows ``2.88``, not ``2.8797932657906435``).
        """
        step = float(self._step or 1.0)
        exponent = int(Decimal(repr(step)).normalize().as_tuple().exponent)
        places = min(max(-exponent, 0), max(0, 2 - math.floor(math.log10(step))))
        return Decimal(1).scaleb(-places)

    def _set(self, value: float) -> bool:
        """Set the value, counting a change for :attr:`moved`."""
        if value == self._value:
            return False
        self._value = value
        self._version += 1
        return True

    def _snap(self, value: float) -> float:
        """Keep a value within the range and on the step.

        Steps are counted from ``low`` in decimal, so the result is the float
        nearest the exact decimal ``low + k · step``.
        """
        low, high = self._range
        value = min(max(value, low), high)
        if self._step is None:
            return value
        base, step = Decimal(repr(low)), Decimal(repr(self._step))
        k = math.floor((value - low) / self._step + 0.5)
        snapped = float(base + k * step)
        if snapped > high:  # high is not on the step: the last step below it
            snapped = float(base + (k - 1) * step)
        return snapped


def _number(name: str, value: object) -> float:
    """Return a real number as a float.

    Raises:
        TypeError: If it is not a real number.
        ValueError: If it is NaN.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"the {name} must be a number, got {value!r}"
        raise TypeError(msg)
    number = float(value)
    if math.isnan(number):
        msg = f"the {name} must not be NaN"
        raise ValueError(msg)
    return number


def _check_range(value_range: object, *, log: bool) -> tuple[float, float]:
    """Validate a slider's range.

    Raises:
        TypeError: If its ends are not numbers.
        ValueError: If it is not two finite numbers in increasing order
            (positive for a log slider).
    """
    try:
        low, high = cast(tuple[object, object], value_range)
    except (TypeError, ValueError):
        msg = f"the range must be a pair (low, high), got {value_range!r}"
        raise ValueError(msg) from None
    lo, hi = _number("range", low), _number("range", high)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        msg = f"the range must be two finite numbers, low < high, got {value_range!r}"
        raise ValueError(msg)
    if log and lo <= 0:
        msg = f"a log slider needs a positive range, got {value_range!r}"
        raise ValueError(msg)
    return lo, hi


def _is_int(step: object) -> bool:
    """Whether a step makes an integer slider (it is an ``int``, not a bool)."""
    return isinstance(step, (int, numbers.Integral)) and not isinstance(step, bool)


def _check_step(
    step: object, low: float, high: float, *, integer: bool
) -> float | None:
    """Validate a slider's step.

    Raises:
        TypeError: If it is not a number.
        ValueError: If it is not positive and at most the range's width, or
            an integer step has ends that are not whole numbers.
    """
    if step is None:
        return None
    size = _number("step", step)
    # A step as wide as the range is allowed even when rounding makes it a
    # little wider than high - low in floats (1.001 - 1.0 < 0.001).
    slack = 4 * math.ulp(max(abs(low), abs(high)))
    if not (0 < size <= high - low + slack):
        msg = f"the step must be positive and at most high - low, got {step!r}"
        raise ValueError(msg)
    if integer and not (low.is_integer() and high.is_integer()):
        msg = "an integer step (an int) needs whole-number ends; give step as a float"
        raise ValueError(msg)
    return size
