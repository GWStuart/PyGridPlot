"""Rendering backends: turning core primitives into pixels.

* :mod:`~pygridplot.render.base`: the :class:`Renderer` protocol.
* :mod:`~pygridplot.render.pygame_renderer`: the pygame-ce backend.
* :mod:`~pygridplot.render.text`, :mod:`~pygridplot.render.fonts` and
  :mod:`~pygridplot.render.colors`: helpers used by the pygame backend.
"""

from pygridplot.render.base import Renderer
from pygridplot.render.pygame_renderer import PygameRenderer

__all__ = ["PygameRenderer", "Renderer"]
