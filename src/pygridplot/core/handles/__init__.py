"""Draggable points: positions the user changes with the mouse.

The only stateful plotted objects. Pure maths like the rest of ``core``:
the pygame side (hit testing, events, the cursor) is in
:mod:`pygridplot.dragging`.

* :mod:`~pygridplot.core.handles.point`: :class:`DraggablePoint` itself.
* :mod:`~pygridplot.core.handles.constraints`: free, locked and on-curve
  movement, with bounds.
* :mod:`~pygridplot.core.handles.tracking`: following a curve with the
  cursor without jumping between its branches.
* :mod:`~pygridplot.core.handles.labels`: the coordinate label's text,
  rounded to a pixel, and placement.
* :mod:`~pygridplot.core.handles.drawing`: the markers and label as
  primitives.

This subpackage builds on both :mod:`pygridplot.core.grid` (number
formatting, the theme) and :mod:`pygridplot.core.plot` (curves, styles,
markers); neither of them, nor the foundation modules, imports it.
"""
