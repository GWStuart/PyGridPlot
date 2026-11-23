"""Slope fields: the solutions of y′ = x − y.

``slope_field(f)`` draws a short segment of slope ``f(x, y)`` at the
points of a lattice: the direction a solution of ``y′ = f(x, y)`` takes
there. This equation's solutions are ``y = x − 1 + C e^(−x)``, one for
each ``C``. Drag the point: the solution through it is drawn in bold.
"""

from collections.abc import Callable

import numpy as np
import pygame

import pygridplot as pgp


def solution(c: float) -> Callable[[float], float]:
    """Return the solution ``y = x - 1 + c e^(-x)``."""
    return lambda x: x - 1 + c * np.exp(-x)


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: slope fields")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-3.5, 6.5))
    point = grid.draggable((0.0, 2.0))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_title="x", y_title="y")
        grid.slope_field(screen, lambda x, y: x - y, color="gray", label="y′ = x − y")
        for c in (-4, -2, -1, -0.5, 0.5, 1, 2, 4):
            grid.plot(
                screen, solution(c), color="steelblue", width=1.5, label="solutions"
            )
        # The constant C of the solution through the point.
        c0 = (point.y - point.x + 1) * np.exp(point.x)
        grid.plot(screen, solution(c0), color="red", width=3, label=f"C = {c0:.2f}")
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
