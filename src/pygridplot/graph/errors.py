"""Errors and warnings of the graph layer."""

from __future__ import annotations


class GraphError(ValueError):
    """An invalid operation on a graph, or an invalid saved graph.

    Raised for a name used twice, an item that is not in the graph, a graph
    that cannot be saved (an item uses a Python function that was not
    registered by name), or a file that is not a valid saved graph. Errors
    in saved graphs say where the problem is, as in
    ``items[2].style.width: expected a number``.
    """


class GraphWarning(UserWarning):
    """An item cannot be drawn: an unknown name, a circular definition, ...

    Graphs keep drawing the items that work, as a graphing calculator does,
    and record the problem in the item's :attr:`~pygridplot.graph.Item.error`.
    This warning is issued once each time an item's error changes, so that
    mistakes do not go unnoticed. Silence it with :mod:`warnings` if the
    errors are shown some other way (in an expression list, say).
    """
