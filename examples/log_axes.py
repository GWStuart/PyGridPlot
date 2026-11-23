"""Logarithmic axes: power laws are straight lines.

On log-log axes ``y = xᵏ`` is a straight line of slope ``k``, and every
decade is the same size (given only the x range, the plane keeps both
axes' decades equal). Zoom out and the decades thin out; zoom into one
decade and its subdivisions fade in until it looks linear.
"""

from collections.abc import Callable

import pygame

import pygridplot as pgp


def power(k: float) -> Callable[[float], float]:
    """Return the function ``x**k``.

    (A lambda written in the loop below would see only the last ``k``.)
    """
    return lambda x: x**k


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: logarithmic axes")
    clock = pygame.time.Clock()
    grid = pgp.Grid(log_x=True, log_y=True, x_range=(0.01, 100))
    powers = [(-1, "1/x"), (0.5, "√x"), (1, "x"), (2, "x²"), (3, "x³")]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_title="x", y_title="y")
        for k, name in powers:
            grid.plot(screen, power(k), label=f"y = {name}")
        grid.legend(screen, corner="top-left")
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
