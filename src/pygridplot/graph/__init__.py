"""Graphs: retained lists of plots, curves, points and parameters.

A :class:`Graph` holds the items of a picture, by name and in order, drawn
onto a plane with one call each frame and saved to JSON. It is built on
the plane's immediate-mode API, so the two mix freely.

* :mod:`~pygridplot.graph.graph`: :class:`Graph`.
* :mod:`~pygridplot.graph.items`: the kinds of item (:class:`PlotItem`, ...).
* :mod:`~pygridplot.graph.lines`: reading one line of maths into an item.
* :mod:`~pygridplot.graph.namespace`: compiling items with their names.
* :mod:`~pygridplot.graph.view`: :class:`GraphView`, the view saved with a
  graph.
* :mod:`~pygridplot.graph.serialize`: the JSON format.
* :mod:`~pygridplot.graph.errors`: :class:`GraphError` and
  :class:`GraphWarning`.
"""

from pygridplot.core.expr import ExpressionError
from pygridplot.graph.errors import GraphError, GraphWarning
from pygridplot.graph.graph import Graph
from pygridplot.graph.items import (
    DefinitionItem,
    DraggableItem,
    ImplicitItem,
    Item,
    LineItem,
    ParameterItem,
    ParametricItem,
    PlotItem,
    PointItem,
    PointLikeItem,
    PointsItem,
    PolarItem,
    PolygonItem,
    PolylineItem,
    StyledItem,
)
from pygridplot.graph.view import GraphView

__all__ = [
    "DefinitionItem",
    "DraggableItem",
    "ExpressionError",
    "Graph",
    "GraphError",
    "GraphView",
    "GraphWarning",
    "ImplicitItem",
    "Item",
    "LineItem",
    "ParameterItem",
    "ParametricItem",
    "PlotItem",
    "PointItem",
    "PointLikeItem",
    "PointsItem",
    "PolarItem",
    "PolygonItem",
    "PolylineItem",
    "StyledItem",
]
