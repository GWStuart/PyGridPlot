"""What every object shares: a name, visibility, errors, drawing and styles.

* :class:`PlotObject`: the base of every object: a name, visibility, the
  error that stops it being drawn, and drawing in two phases
  (:meth:`~PlotObject.update`, then :meth:`~PlotObject.render`).
* :class:`StyledObject` and its kinds (:class:`LineObject`,
  :class:`MarkerObject`): objects drawn with a style, with a palette slot
  for their default colour.

**Owners.** An object may belong to an *owner*, a
:class:`~pygridplot.Graph`, which compiles its text with the names of the
other objects. The owner's interface is the :class:`Owner` protocol below,
so this package never imports the graph package that builds on it.

**The objects' interface to their owner.** Besides their public
attributes, objects have a few methods starting with ``_`` that the graph
package uses and programs do not: ``_formulas``, ``_runtime``,
``_binding`` and ``_describe`` (what an object compiles and stands for, see
:class:`PlotObject`), ``_changed`` (it tells the owner to compile again),
and ``_owner``, which the owner sets. They are private to the two packages,
not to this module: a new kind of object implements them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from pygridplot.core.expr import Binding, ExpressionError
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.plot.style import (
    LineStyle,
    PointStyle,
    with_overrides,
)
from pygridplot.objects.errors import warn
from pygridplot.objects.values import Formula, Maker, check_name, standalone

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane


class Owner(Protocol):
    """What holds objects by name and compiles their text: a ``Graph``."""

    @property
    def items(self) -> tuple[PlotObject, ...]:
        """The objects, in order."""
        ...

    def _invalidate(self) -> None:
        """Forget compiled expressions (names or expressions changed)."""
        ...

    def _rename(self, obj: PlotObject, name: str | None) -> None:
        """Check that an object may take a name (raise if not)."""
        ...

    def _resolve(self, obj: PlotObject) -> tuple[Any, str | None]:
        """Return an object's compiled runtime and its error, compiling if needed."""
        ...

    def _curve_named(self, obj: PlotObject) -> Any:
        """Return the object whose curve a draggable point follows by name."""
        ...


class PlotObject:
    """Something to draw (or, in a graph, a name defined for other objects).

    Objects are made once and changed through their attributes; every
    attribute is read when the object is drawn. Draw them by adding them to
    a plane (``plane.add(obj)``, then ``plane.draw(screen)`` every frame), by
    putting them in a :class:`~pygridplot.Graph`, or one at a time with
    :meth:`draw`.

    Subclasses describe themselves through a few hooks: ``_formulas()``
    (their formulas, with the variables of each), ``_runtime(make)``
    (compile them with ``make`` into what drawing needs), ``_binding(runtime)``
    (what their name stands for in a graph's expressions, if anything),
    ``_render(plane, surface, runtime)`` (draw) and ``_describe()`` (for
    ``repr``). Setters that change names or expressions call ``_changed()``.

    Attributes:
        kind: The kind of object, as written in saved graphs (``"plot"``, ...).
    """

    kind: ClassVar[str] = "object"
    drawn: ClassVar[bool] = True
    """Whether the object draws anything (parameters and definitions do not)."""

    def __init__(self, *, name: str | None = None, visible: bool = True) -> None:
        self._owner: Owner | None = None
        self._name = None if name is None else check_name(name)
        self._visible = bool(visible)
        self._own: tuple[Any, str | None] | None = None
        self._draw_error: str | None = None
        self._reported: str | None = None

    # --- Attributes -------------------------------------------------------
    @property
    def name(self) -> str | None:
        """The object's name (``None`` if unnamed).

        In a graph, other objects' expressions refer to it by this name.
        Renaming does not rewrite expressions that used the old name.
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        new = None if name is None else check_name(name)
        if self._owner is not None:
            self._owner._rename(self, new)
        self._name = new
        self._changed()

    @property
    def visible(self) -> bool:
        """Whether the object is drawn (a hidden object in a graph keeps its name)."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._visible = bool(visible)

    @property
    def graph(self) -> Owner | None:
        """The :class:`~pygridplot.Graph` the object belongs to, if any."""
        return self._owner

    @property
    def error(self) -> str | None:
        """Why the object cannot be drawn, or ``None`` if it can.

        For example ``"unknown name 'b'"`` or ``"circular definition: f → g
        → f"``. Errors in expressions are always up to date; the rare errors
        found only when drawing (an invalid setting for a plane) are those
        of the last time the object was drawn.
        """
        _, error = self._live()
        return error if error is not None else self._draw_error

    def references(self) -> set[str]:
        """Return the names of other objects this object's expressions use.

        Returns:
            The names used in its text, besides variables and built-ins
            (only meaningful in a graph).
        """
        names: set[str] = set()
        for formula, variables in self._formulas():
            names |= formula.names(variables)
        return names

    # --- Drawing ------------------------------------------------------------
    def update(self, plane: Plane) -> None:
        """Prepare to draw on a plane: the first half of :meth:`draw`.

        A plane (or graph) updates every object before rendering any, so
        that an object reading a draggable point sees where it was dragged
        this frame. Programs normally call :meth:`draw`, or let the plane do
        it.

        Args:
            plane: The plane about to be drawn on.
        """
        self._draw_error = None

    def render(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw onto a plane: the second half of :meth:`draw`.

        A hidden object or one with an error draws nothing; an error is
        reported with a :class:`~pygridplot.PlotWarning` the first time it
        appears.

        Args:
            plane: The plane.
            surface: The surface (usually the display).
        """
        runtime, error = self._live()
        if error is None and self._draw_error is None and self._visible:
            try:
                self._render(plane, surface, runtime)
            except (TypeError, ValueError) as failure:
                self._draw_error = str(failure)
        if self._visible or not self.drawn:
            self._report(error if error is not None else self._draw_error)

    def handles(self, plane: Plane) -> tuple[DraggablePoint, ...]:
        """Return the draggable handles the object shows on a plane (to draw last).

        Args:
            plane: The plane.

        Returns:
            The handles (none, except for draggable points).
        """
        return ()

    def detach(self, plane: Plane) -> None:
        """Take anything the object keeps on a plane off it (draggable handles).

        Args:
            plane: The plane the object will no longer be drawn on.
        """

    def draw(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw the object onto a plane now, with its draggable handle on top.

        Use it to place one object among immediate drawing calls; objects
        added to a plane are drawn by :meth:`Plane.draw_objects
        <pygridplot.Plane.draw_objects>`.

        Args:
            plane: The plane.
            surface: The surface (usually the display).
        """
        self.update(plane)
        self.render(plane, surface)
        shown = self.handles(plane)
        if shown:
            plane.draw_draggables(surface, *shown)

    def __repr__(self) -> str:
        """Return the class, the name and a description."""
        name = f" {self._name!r}" if self._name else ""
        detail = self._describe()
        return f"<{type(self).__name__}{name}{': ' if detail else ''}{detail}>"

    # --- For subclasses and owners ----------------------------------------
    def _changed(self) -> None:
        """Note that names or expressions changed: compile again when drawn."""
        self._own = None
        if self._owner is not None:
            self._owner._invalidate()

    def _live(self) -> tuple[Any, str | None]:
        """Return the compiled runtime and the error, compiling if needed."""
        if self._owner is not None:
            return self._owner._resolve(self)
        if self._own is None:
            try:
                self._own = (self._runtime(standalone), None)
            except ExpressionError as error:
                self._own = (None, error.message)
        return self._own

    def _working(self) -> Any:
        """Return the compiled runtime, or raise if the object has an error.

        Raises:
            ValueError: If the object has an error in its expressions.
        """
        runtime, error = self._live()
        if error is not None:
            msg = f"{self._label()} has an error: {error}"
            raise ValueError(msg)
        return runtime

    def _report(self, message: str | None) -> None:
        """Warn about an error the first time it appears (or changes)."""
        if message == self._reported:
            return
        self._reported = message
        if message is not None:
            warn(f"{self._label()}: {message}")

    def _label(self) -> str:
        """Name the object in a warning."""
        if self._name:
            return repr(self._name)
        if self._owner is not None:
            items = self._owner.items
            index = next(i for i, item in enumerate(items) if item is self)
            return f"item {index} ({self.kind})"
        return repr(self)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        """Return the object's formulas and the variables of each."""
        return []

    def _runtime(self, make: Maker) -> Any:
        """Compile the object's formulas into what drawing needs."""
        return None

    def _binding(self, runtime: Any) -> Binding | None:
        """Return what the object's name stands for in expressions, if anything."""
        return None

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        """Draw with the compiled runtime (the object is visible, without errors)."""

    def _describe(self) -> str:
        """Return a short description of the object, for ``repr`` and errors."""
        return ""


def style_property(field: str, annotation: str, doc: str) -> property:
    """Make a property for one field of a styled object's style.

    Reading it reads the style's field; setting it replaces the style with
    one where the field is changed (styles are immutable).

    Args:
        field: The style's field.
        annotation: Its type, for the documentation.
        doc: The property's documentation.

    Returns:
        The property.
    """

    def get(self: StyledObject) -> Any:
        return getattr(self._style, field)

    def put(self: StyledObject, value: Any) -> None:
        self._set_style(**{field: value})

    get.__annotations__["return"] = annotation
    return property(get, put, doc=doc)


class StyledObject(PlotObject):
    """An object drawn with a style: a line, a point or a polygon style.

    An object without a colour takes the colour of its *palette slot* from
    the plane's theme. A plane or graph gives each object it is given the
    next slot, so colours stay with their objects when others are hidden or
    removed, and follow the theme; an object drawn without one takes the
    next colour of the frame, like the plane's drawing methods.
    """

    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    def __init__(
        self,
        *,
        style: Any = None,
        overrides: dict[str, Any] | None = None,
        palette: int | None = None,
        name: str | None = None,
        visible: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(name=name, visible=visible)
        base = self._check_style(style) if style is not None else default
        if base is None:
            base = self.default_style
        self._style = with_overrides(base, **(overrides or {}))
        self._palette = self._check_palette(palette)

    @property
    def style(self) -> Any:
        """The object's style: a :class:`~pygridplot.LineStyle`, ..."""
        return self._style

    @style.setter
    def style(self, style: Any) -> None:
        self._style = self._check_style(style)

    @property
    def palette(self) -> int | None:
        """The palette slot giving the object its colour when ``color`` is ``None``."""
        return self._palette

    @palette.setter
    def palette(self, slot: int | None) -> None:
        self._palette = self._check_palette(slot)

    color = style_property(
        "color",
        "ColorLike | None",
        "The colour (``None`` for the colour of the object's palette slot).",
    )
    opacity = style_property("opacity", "float", "The opacity, in ``[0, 1]``.")

    def _color(self, plane: Plane) -> Any:
        """Return the colour to draw with (``None``: the plane's next colour)."""
        if self.color is not None:
            return self.color
        if self._palette is None:
            return None
        palette = plane.theme.palette
        return palette[self._palette % len(palette)]

    def _set_style(self, **changes: Any) -> None:
        self._style = replace(self._style, **changes)

    def _check_style(self, style: Any) -> Any:
        if not isinstance(style, self.style_type):
            msg = f"style must be a {self.style_type.__name__}, got {style!r}"
            raise TypeError(msg)
        return style

    @staticmethod
    def _check_palette(slot: object) -> int | None:
        if slot is None:
            return None
        if (
            isinstance(slot, bool)
            or not isinstance(slot, (int, np.integer))
            or slot < 0
        ):
            msg = f"palette must be a non-negative whole number or None, got {slot!r}"
            raise ValueError(msg)
        return int(slot)


class LineObject(StyledObject):
    """An object drawn as lines, with a :class:`~pygridplot.LineStyle`."""

    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    width = style_property("width", "float", "The line width, in pixels.")


class MarkerObject(StyledObject):
    """An object drawn as markers, with a :class:`~pygridplot.PointStyle`."""

    style_type: ClassVar[type] = PointStyle
    default_style: ClassVar[Any] = PointStyle()

    radius = style_property("radius", "float", "The marker size, in pixels.")
    marker = style_property(
        "marker", "MarkerShape", 'The marker shape (``"circle"``, ``"square"``, ...).'
    )
    edge_color = style_property(
        "edge_color",
        "ColorLike | None",
        "The colour of the marker's outline (``None`` for none).",
    )
    edge_width = style_property(
        "edge_width", "float", "The outline's thickness, in pixels."
    )
