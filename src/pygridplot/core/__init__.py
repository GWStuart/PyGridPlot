"""The pure maths layer of PyGridPlot.

Everything in this package depends only on the standard library and NumPy.
It must **never** import pygame (a test enforces this), so it stays fast to
test, easy to reason about and reusable with other rendering backends.

Modules, roughly in dependency order:

* :mod:`~pygridplot.core.coords`: screen rectangles, world bounds, type
  aliases.
* :mod:`~pygridplot.core.pixels`: pixel-snapping conventions.
* :mod:`~pygridplot.core.view`: the camera with world/screen transforms and
  pan/zoom maths.
* :mod:`~pygridplot.core.nice`: the exact 1-2-5 spacing ladder.
* :mod:`~pygridplot.core.spacing`: choosing a spacing for the zoom level
  (fade or snap).
* :mod:`~pygridplot.core.labels`: exact tick label formatting.
* :mod:`~pygridplot.core.primitives`: drawing primitives handed to renderers.
* :mod:`~pygridplot.core.theme`: grid appearance and presets.
* :mod:`~pygridplot.core.ticks`: grid line generation and classification.
* :mod:`~pygridplot.core.gridlines`: painting grid lines and axes.
* :mod:`~pygridplot.core.ticklabels`: which labels to show (thinning, fades).
* :mod:`~pygridplot.core.layout`: tick label placement.
* :mod:`~pygridplot.core.grid`: assembling the full grid for a frame.
* :mod:`~pygridplot.core.easing`: the spring behind eased pan and zoom.
"""
