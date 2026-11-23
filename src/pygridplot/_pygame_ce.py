"""Fail early, and clearly, if the original pygame is installed, not pygame-ce.

pygame community edition (pygame-ce) and the original pygame install under
the same import name, ``pygame``. With the original installed (alone, or
over pygame-ce) PyGridPlot would fail later with an obscure error, a missing
function or argument, so the package checks once, on import, and says what
to do. pygame-ce marks itself with ``pygame.IS_CE``.

The package's ``__init__`` imports this module before anything that uses
pygame.
"""

import pygame

if not getattr(pygame, "IS_CE", False):
    _version = getattr(getattr(pygame, "version", None), "ver", "?")
    _msg = (
        "PyGridPlot needs pygame-ce (pygame community edition), but the pygame "
        f"installed is the original pygame {_version}. Both install as "
        "'pygame': run 'pip uninstall pygame', then "
        "'pip install --force-reinstall pygame-ce'."
    )
    raise ImportError(_msg)
