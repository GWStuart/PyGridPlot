"""Persistent plot objects: functions, curves and points you make once and change.

Each object mirrors a drawing method of :class:`~pygridplot.Plane`
(:class:`Function` draws like :meth:`~pygridplot.Plane.plot`, and so on),
takes the same arguments, and keeps them as attributes that can be changed
at any time; every attribute is read when the object is drawn::

    a = pgp.Point((1, 2), draggable=True)
    f = pgp.Function(lambda x: a.y * np.sin(x - a.x), color="red")
    plane.add(a, f)
    ...
    f.width = 4                      # in the main loop, whenever
    plane.draw(screen)               # grid, objects, draggable points

Objects are drawn by a plane they were added to (:meth:`Plane.add
<pygridplot.Plane.add>`), or one at a time with :meth:`PlotObject.draw`.
Python closures make them follow each other: a function reading ``a.x``
draws where the point is.

* :mod:`~pygridplot.objects.base`: :class:`PlotObject` (two-phase drawing,
  errors) and the styled bases.
* :mod:`~pygridplot.objects.values`: argument checks (functions, numbers,
  coordinates).
* :mod:`~pygridplot.objects.curves`: :class:`Function`,
  :class:`ParametricCurve`, :class:`PolarCurve`, :class:`ImplicitCurve`.
* :mod:`~pygridplot.objects.points`: :class:`Point`, and keeping draggable
  points in step with their handles.
* :mod:`~pygridplot.objects.shapes`: :class:`Points`, :class:`Polyline`,
  :class:`Polygon`.
* :mod:`~pygridplot.objects.errors`: :class:`PlotWarning`.

This package builds on the plane's public API and on ``core``; the plane
and ``core`` never import it.
"""

from pygridplot.objects.base import LineObject, MarkerObject, PlotObject, StyledObject
from pygridplot.objects.curves import (
    CurveObject,
    Function,
    ImplicitCurve,
    ParametricCurve,
    PolarCurve,
)
from pygridplot.objects.errors import PlotWarning
from pygridplot.objects.points import CurveTarget, Point
from pygridplot.objects.shapes import Points, Polygon, Polyline
from pygridplot.objects.values import FULL_TURN

__all__ = [
    "FULL_TURN",
    "CurveObject",
    "CurveTarget",
    "Function",
    "ImplicitCurve",
    "LineObject",
    "MarkerObject",
    "ParametricCurve",
    "PlotObject",
    "PlotWarning",
    "Point",
    "Points",
    "PolarCurve",
    "Polygon",
    "Polyline",
    "StyledObject",
]
