"""The items of a graph: one per row of the expression list.

Each kind of item mirrors a plotting method of :class:`~pygridplot.Plane`
(:class:`PlotItem` draws like :meth:`~pygridplot.Plane.plot`, and so on),
and takes the same styling arguments; :class:`ParameterItem` and
:class:`DefinitionItem` draw nothing but define names that other items
use.

Items are created through a :class:`~pygridplot.graph.Graph`'s methods and
can be changed at any time through their attributes; the graph notices.
Expressions are given as text in the expression language (see
:mod:`pygridplot.core.expr`), such as ``"a sin(x - P.x)"``, or as Python
functions; numbers stand for themselves where a value is expected.
Expressions are checked for syntax as soon as they are set, and names are
resolved when the graph is drawn, so items may refer to items added later.

* :mod:`~pygridplot.graph.items.base`: what every item shares.
* :mod:`~pygridplot.graph.items.curves`: plots, parametric, polar and
  implicit curves.
* :mod:`~pygridplot.graph.items.shapes`: points, polylines and polygons.
* :mod:`~pygridplot.graph.items.draggable`: draggable points.
* :mod:`~pygridplot.graph.items.names`: parameters and definitions.
"""

from typing import Final

from pygridplot.graph.items.base import (
    FULL_TURN,
    VARIABLE_NAMES,
    Coordinate,
    Expression,
    Formula,
    Item,
    LineItem,
    Maker,
    PointLikeItem,
    StyledItem,
    check_name,
)
from pygridplot.graph.items.curves import (
    ImplicitItem,
    ParametricItem,
    PlotItem,
    PolarItem,
)
from pygridplot.graph.items.draggable import CurveTarget, DraggableItem
from pygridplot.graph.items.names import DefinitionItem, ParameterItem, parse_signature
from pygridplot.graph.items.shapes import (
    PointItem,
    PointsItem,
    PolygonItem,
    PolylineItem,
)

ITEM_TYPES: Final[dict[str, type[Item]]] = {
    cls.kind: cls
    for cls in (
        PlotItem,
        ParametricItem,
        PolarItem,
        ImplicitItem,
        PointItem,
        PointsItem,
        PolylineItem,
        PolygonItem,
        DraggableItem,
        ParameterItem,
        DefinitionItem,
    )
}
"""Every kind of item, by its :attr:`Item.kind`."""

__all__ = [
    "FULL_TURN",
    "ITEM_TYPES",
    "VARIABLE_NAMES",
    "Coordinate",
    "CurveTarget",
    "DefinitionItem",
    "DraggableItem",
    "Expression",
    "Formula",
    "ImplicitItem",
    "Item",
    "LineItem",
    "Maker",
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
    "check_name",
    "parse_signature",
]
