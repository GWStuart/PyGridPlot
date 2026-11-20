"""Building the grid: spacing, lines, tick labels and their layout.

:func:`pygridplot.core.grid.scene.build_grid` is the entry point. The
pipeline, one module per step:

* :mod:`~pygridplot.core.grid.nice`: exact 1-2-5 spacings (``NiceStep``).
* :mod:`~pygridplot.core.grid.spacing`: choosing a spacing for a zoom level,
  with cross-fades or hysteresis.
* :mod:`~pygridplot.core.grid.ticks` and
  :mod:`~pygridplot.core.grid.ticklabels`: lines and labels of linear axes.
* :mod:`~pygridplot.core.grid.logticks` and
  :mod:`~pygridplot.core.grid.loglabels`: lines and labels of log axes.
* :mod:`~pygridplot.core.grid.gridlines`: painting lines as line batches.
* :mod:`~pygridplot.core.grid.formatting`: exact number formatting.
* :mod:`~pygridplot.core.grid.layout`: placing labels around the axes.
* :mod:`~pygridplot.core.grid.scene`: putting it all together.
"""
