"""A family of curves, animated: ``y = sin(x - k t) * k / 4`` for several ``k``.

This example also shows the one classic Python pitfall when building
functions in a loop, *late binding*::

    fs = [lambda x: x**k for k in range(3)]       # all three use k = 2!
    fs = [lambda x, k=k: x**k for k in range(3)]  # correct: k is bound

Each lambda below binds ``k`` (and the time ``t``) as default arguments.
"""

from __future__ import annotations

import numpy as np
import pygame

import pygridplot as pgp

COUNT = 8


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: a family of curves")
    plane = pgp.Plane(scale=70)
    clock = pygame.time.Clock()
    t = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            plane.handle_event(event)
        t += clock.tick(60) / 1000

        plane.draw_grid(screen)
        family = [
            lambda x, k=k, t=t: np.sin(x - 0.6 * k * t) * k / 4
            for k in range(1, COUNT + 1)
        ]
        for func in family:
            plane.plot(screen, func, width=2.0)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
