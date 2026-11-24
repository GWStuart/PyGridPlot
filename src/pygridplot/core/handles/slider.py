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

A slider made with ``play=`` has a **play button**, as in Desmos: pressed,
the handle sweeps along the track by itself, one sweep taking
:attr:`Slider.duration` seconds, until pressed again (:attr:`Slider.playing`
from code). At the end of the track it starts again from the low end
(``"loop"``), turns round (``"bounce"``) or stops (``"once"``). The sweep is
even along the track (geometric on a log slider); a stepped slider goes
from step to step, evenly in time, and one without a step moves smoothly
(its value is not rounded to a pixel while it plays). The plane advances
playing sliders at the start of each frame (:meth:`Slider.advance`), by the
time since the previous one.
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
"""Where in the plane a slider's (or checkbox's) panel sits."""

CORNERS: tuple[Corner, ...] = ("bottom-left", "bottom-right", "top-left", "top-right")
"""The allowed values of :attr:`Slider.corner`."""

ValueT = TypeVar("ValueT", int, float)
"""The type of a slider's value: ``int`` for an integer slider, else ``float``."""

PlayMode: TypeAlias = Literal["loop", "bounce", "once"]
"""What a playing slider does when it reaches the end of its track.

* ``"loop"``: start again from the low end.
* ``"bounce"``: turn round, and sweep back and forth between the ends.
* ``"once"``: stop at the high end (pressing play there starts again from
  the low end).
"""

PLAY_MODES: tuple[PlayMode, ...] = ("loop", "bounce", "once")
"""The allowed values of :attr:`Slider.play`."""

DEFAULT_DURATION: Final = 5.0
"""Seconds a playing slider takes to sweep its track, unless given."""

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
        enabled: Whether the slider can be dragged, and its play button
            pressed (a disabled slider is drawn faded).
        play: What a playing slider does at the end of its track (see
            :data:`PlayMode`), which gives it a play button; ``None`` (the
            default) for none.
        duration: How many seconds a playing slider takes to sweep the
            whole track.
        on_remove: Called by :meth:`remove` (the plane forgets the slider).

    Raises:
        TypeError: If the label is not a string, or the range, value, step
            or duration are not numbers.
        ValueError: If the range is not two finite numbers in increasing
            order (positive for a log slider), the step is not positive or
            wider than the range, an integer step has ends that are not
            whole numbers, the value is NaN, the corner or play mode is
            unknown, or the duration is not positive and finite.
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
        play: PlayMode | None = None,
        duration: float = DEFAULT_DURATION,
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
        """Whether the slider can be dragged and played (it is drawn faded when not)."""
        self._on_remove = on_remove
        self._value = self._snap(low if value is None else _number("value", value))
        self._version = 0
        self._seen_version = 0
        self._moved = False
        self._hovered = False
        self._dragging = False
        self._play_hovered = False
        self._playing = False
        self.play = play
        self.duration = duration
        # Where a playing slider is along its track, unrounded (its value is
        # on the step), and which way it is going (-1 coming back on a bounce).
        self._position = self.fraction
        self._direction = 1.0
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
        self._position = self.fraction

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
        self._corner = check_corner(corner)

    @property
    def play(self) -> PlayMode | None:
        """What the slider does when playing reaches the end of the track.

        ``"loop"``, ``"bounce"`` or ``"once"`` (see :data:`PlayMode`), or
        ``None`` for a slider without a play button. Setting ``None`` stops
        it playing.
        """
        return self._play

    @play.setter
    def play(self, mode: PlayMode | None) -> None:
        if mode is not None and mode not in PLAY_MODES:
            msg = f"play must be one of {PLAY_MODES} or None, got {mode!r}"
            raise ValueError(msg)
        self._play = mode
        if mode is None:
            self._playing = False

    @property
    def duration(self) -> float:
        """How many seconds playing takes to sweep the whole track."""
        return self._duration

    @duration.setter
    def duration(self, seconds: float) -> None:
        value = _number("duration", seconds)
        if not (math.isfinite(value) and value > 0):
            msg = f"the duration must be positive and finite, got {seconds!r}"
            raise ValueError(msg)
        self._duration = value

    @property
    def playing(self) -> bool:
        """Whether the handle is sweeping along the track by itself.

        The play button switches it; set it to start or stop playing from
        code (a slider needs a :attr:`play` mode to play). Playing ``"once"``
        stops by itself at the end of the track, and starting again there
        goes back to the low end. Dragging the handle holds it where it is
        dragged; it plays on from there when let go.

        Raises:
            ValueError: If set to ``True`` on a slider without a play mode.
        """
        return self._playing

    @playing.setter
    def playing(self, playing: bool) -> None:
        if playing and self._play is None:
            msg = "this slider cannot play: give it play='loop', 'bounce' or 'once'"
            raise ValueError(msg)
        start = bool(playing) and not self._playing
        self._playing = bool(playing)
        if start and self._play == "once" and self._position >= 1.0:
            self._direction = 1.0
            self._set(self._played_value(0.0))
            self._position = 0.0

    def advance(self, seconds: float) -> bool:
        """Move a playing slider on by some time, as the plane does every frame.

        Nothing happens if the slider is not playing, or is being dragged.

        Args:
            seconds: The time since the previous frame.

        Returns:
            Whether the value changed.
        """
        if not self._playing or self._dragging or not seconds > 0:
            return False
        travel = seconds / self._duration
        if self._play == "loop":
            position = (self._position + travel) % 1.0
        elif self._play == "bounce":
            # Unfold the back-and-forth into one direction, two tracks long.
            unfolded = self._position if self._direction > 0 else 2.0 - self._position
            unfolded = (unfolded + travel) % 2.0
            self._direction = 1.0 if unfolded < 1.0 else -1.0
            position = unfolded if unfolded <= 1.0 else 2.0 - unfolded
        else:
            position = min(self._position + travel, 1.0)
            if position >= 1.0:
                self._playing = False
        self._position = position
        return self._set(self._played_value(position))

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
        changed = self._set(self.value_at(fraction))
        self._position = self.fraction
        return changed

    def nudge(self, notches: int) -> bool:
        """Move the value by whole steps, as the mouse wheel does.

        Args:
            notches: How many steps up (negative for down). Without a step,
                each is :data:`WHEEL_FRACTION` of the track.

        Returns:
            Whether the value changed.
        """
        if self._step is not None:
            changed = self._set(self._snap(self._value + notches * self._step))
        else:
            changed = self._set(self.value_at(self.fraction + notches * WHEEL_FRACTION))
        self._position = self.fraction
        return changed

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

    @property
    def play_hovered(self) -> bool:
        """Whether the cursor is over the slider's play button."""
        return self._play_hovered

    def set_interaction(
        self, *, hovered: bool, dragging: bool, play_hovered: bool = False
    ) -> None:
        """Record whether the slider is hovered and dragged (by the plane).

        Args:
            hovered: Whether the cursor is over the track.
            dragging: Whether the handle is being dragged.
            play_hovered: Whether the cursor is over the play button.
        """
        self._hovered = hovered
        self._dragging = dragging
        self._play_hovered = play_hovered

    def begin_frame(self) -> None:
        """Start a new frame: update :attr:`moved`.

        :meth:`pygridplot.Grid.begin_frame` calls this for every slider.
        """
        self._moved = self._version != self._seen_version
        self._seen_version = self._version

    def remove(self) -> None:
        """Remove the slider from its plane: it is no longer dragged or drawn."""
        self._dragging = self._hovered = self._play_hovered = False
        self._playing = False
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

    def _played_value(self, position: float) -> float:
        """The value at a point of the track while playing.

        Without a step it is exact: unlike a drag's, it is not rounded to a
        pixel of the track, so the sweep is smooth. With a step, each of the
        allowed values takes an equal share of the sweep (rounding to the
        nearest would give the two ends half as long as the others, so an
        integer slider looping from 1 to 5 would seem to skip them); on a
        log track, where the values are not evenly spread along it, the
        value is the nearest step.
        """
        low, high = self._range
        if self._log:
            return self._snap(low * (high / low) ** position)
        if self._step is None:
            return self._snap(low + position * (high - low))
        top = round((self._snap(high) - low) / self._step)
        k = min(math.floor(position * (top + 1)), top)
        base, step = Decimal(repr(low)), Decimal(repr(self._step))
        return float(base + k * step)

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


def check_corner(corner: object) -> Corner:
    """Validate the corner of a control's panel.

    Args:
        corner: The corner.

    Returns:
        The corner.

    Raises:
        ValueError: If it is not one of :data:`CORNERS`.
    """
    if corner not in CORNERS:
        msg = f"corner must be one of {CORNERS}, got {corner!r}"
        raise ValueError(msg)
    return corner


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
