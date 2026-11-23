"""Curves: parametric and polar curves, on the dark theme.

``parametric(x, y, t_range)`` draws the points ``(x(t), y(t))``, and
``polar(r)`` draws ``r(θ)`` around the origin. Both sample more where the
curve bends, so loops and cusps stay smooth at any zoom. Each curve moves
with the time ``t``: plots are redrawn from scratch every frame, so
animation needs nothing special.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: curves")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-12, 12), theme=pgp.DARK)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        t = pygame.time.get_ticks() / 1000  # seconds
        grid.draw(screen)
        # A Lissajous figure whose phase drifts.
        grid.parametric(
            screen,
            lambda s: -7.5 + 3 * np.sin(3 * s + 0.4 * t),
            lambda s: 3 * np.sin(2 * s),
            (0, 2 * np.pi),
            label="Lissajous figure",
        )
        # A rose with five petals, turning.
        grid.polar(screen, lambda a: 3.5 * np.cos(5 * (a - 0.2 * t)), label="rose")
        # A spirograph (hypotrochoid) whose pen moves in and out.
        pen = 2.2 + 0.6 * np.sin(0.5 * t)
        grid.parametric(
            screen,
            lambda s: 7.5 + 1.5 * np.cos(s) + pen * np.cos(0.6 * s),
            lambda s: 1.5 * np.sin(s) - pen * np.sin(0.6 * s),
            (0, 10 * np.pi),
            label="spirograph",
        )
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
