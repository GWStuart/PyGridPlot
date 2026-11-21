"""Graphs: named lists of objects, with text expressions and saving.

A :class:`Graph` holds the objects of a picture (the functions, curves and
points of :mod:`pygridplot.objects`, plus :class:`Parameter` and
:class:`Definition`), by name and in order. It reads lines of maths
(``graph.add("y = a sin(x)")``), lets objects' text use each other's names
in any order, reports errors per object, and saves to JSON. It is built on
the objects, which are built on the plane's immediate-mode API, so the
three mix freely.

* :mod:`~pygridplot.graph.graph`: :class:`Graph`.
* :mod:`~pygridplot.graph.definitions`: :class:`Parameter` and
  :class:`Definition`, the items only a graph has.
* :mod:`~pygridplot.graph.lines`: reading one line of maths into an object.
* :mod:`~pygridplot.graph.namespace`: compiling objects with their names.
* :mod:`~pygridplot.graph.view`: :class:`GraphView`, the view saved with a
  graph.
* :mod:`~pygridplot.graph.serialize`: the JSON format.
* :mod:`~pygridplot.graph.errors`: :class:`GraphError`.
"""

from pygridplot.core.expr import ExpressionError
from pygridplot.graph.definitions import Definition, Parameter
from pygridplot.graph.errors import GraphError
from pygridplot.graph.graph import Graph
from pygridplot.graph.view import GraphView

__all__ = [
    "Definition",
    "ExpressionError",
    "Graph",
    "GraphError",
    "GraphView",
    "Parameter",
]
