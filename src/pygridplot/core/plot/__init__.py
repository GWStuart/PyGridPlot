"""Turning plotted objects into screen-space primitives.

:mod:`~pygridplot.core.plot.builders` is the entry point: one function per
kind of plot, each returning clipped primitives for the renderer. The rest
are its building blocks:

* :mod:`~pygridplot.core.plot.style`: line, point and polygon styles, and
  the colour palette.
* :mod:`~pygridplot.core.plot.evaluation`: calling user functions robustly.
* :mod:`~pygridplot.core.plot.sampling`: adaptive sampling of ``y = f(x)``.
* :mod:`~pygridplot.core.plot.curves`: adaptive sampling of parametric
  (and polar) curves.
* :mod:`~pygridplot.core.plot.geometry`: point input and clipping.
"""
