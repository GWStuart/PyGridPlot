"""The error of the graph layer."""

from __future__ import annotations


class GraphError(ValueError):
    """An invalid operation on a graph, or an invalid saved graph.

    Raised for a name used twice, an object that is not in the graph, a
    graph that cannot be saved (an object uses a Python function that was
    not registered by name), or a file that is not a valid saved graph.
    Errors in saved graphs say where the problem is, as in
    ``items[2].style.width: expected a number``.

    Objects that cannot be *drawn* (an unknown name in an expression, a
    circular definition) are not errors of this kind: they are skipped,
    with their reason in their ``error`` attribute and a
    :class:`~pygridplot.PlotWarning`.
    """
