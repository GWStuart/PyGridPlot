"""Quick look: plot a few functions without writing a loop.

``pgp.show`` opens a window, plots the functions and returns when the
window is closed (or Escape is pressed). Drag to pan, scroll to zoom.
"""

import numpy as np

import pygridplot as pgp


def main() -> None:
    """Show three functions."""
    pgp.show(np.sin, np.cos, lambda x: x**2 / 4 - 1, title="PyGridPlot: quick look")


if __name__ == "__main__":
    main()
