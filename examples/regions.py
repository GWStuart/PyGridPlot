"""Regions: inequalities, and the area between two graphs.

``inequality(condition)`` shades the points where a condition holds.
Conditions are plain Python, combined with ``&`` (and), ``|`` (or) and
``~`` (not); brackets matter, as ``&`` binds more tightly than ``<``. Drag
the point to move the ring. ``fill_between(f, g)`` shades between two
graphs, and stops cleanly where they cross.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: regions")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-9.5, 9.5))
    centre = grid.draggable((-5.0, 1.5))

    def ring(x: float, y: float) -> bool:
        """Whether (x, y) is between the two circles around the centre."""
        r2 = (x - centre.x) ** 2 + (y - centre.y) ** 2
        return (r2 > 2.25) & (r2 < 9)

    def f(x: float) -> float:
        """The upper graph."""
        return 1.5 + np.cos(1.5 * x)

    def g(x: float) -> float:
        """The lower graph."""
        return (x - 5) ** 2 / 3 - 3

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen)
        grid.inequality(screen, ring, label="1.5 < distance < 3")
        grid.inequality(screen, lambda x, y: y < -x / 2 - 3.5, label="y < −x/2 − 3.5")
        grid.fill_between(screen, f, g, domain=(1, 9), label="between f and g")
        grid.plot(screen, f, domain=(1, 9), label="f(x) = 1.5 + cos 1.5x")
        grid.plot(screen, g, domain=(1, 9), label="g(x) = (x − 5)²/3 − 3")
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
