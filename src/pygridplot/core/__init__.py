"""The pure maths layer of PyGridPlot.

Everything in this package depends only on the standard library, NumPy and
(for implicit curves) contourpy. It must **never** import pygame
(``tests/test_architecture.py`` enforces this), so it stays fast to test,
easy to reason about and reusable with other rendering backends. Code here
describes *what* to draw, as backend-neutral primitives in screen space;
renderers decide *how*.

Foundations, used by everything else:

* :mod:`~pygridplot.core.coords`: screen rectangles, world bounds and type
  aliases.
* :mod:`~pygridplot.core.pixels`: the pixel-snapping conventions.
* :mod:`~pygridplot.core.transforms`: linear and logarithmic axes.
* :mod:`~pygridplot.core.view`: the camera, with world/screen transforms and
  pan/zoom maths.
* :mod:`~pygridplot.core.framing`: initial ranges, the home view and
  switching axis kinds.
* :mod:`~pygridplot.core.easing`: the spring behind eased pan and zoom.
* :mod:`~pygridplot.core.primitives`: the drawing primitives handed to
  renderers, and the text-measuring protocol.

Subpackages:

* :mod:`pygridplot.core.grid`: the adaptive grid, its tick labels and their
  layout (entry point :func:`pygridplot.core.grid.scene.build_grid`), and
  the theme (:mod:`pygridplot.core.grid.theme`).
* :mod:`pygridplot.core.plot`: plotted objects (functions and their
  derivatives, parametric, polar and implicit curves, points, polylines,
  polygons) to primitives (entry point :mod:`pygridplot.core.plot.builders`).
* :mod:`pygridplot.core.handles`: draggable points, built on the other two.
"""
