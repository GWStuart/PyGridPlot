"""Quick start: a plane you can pan and zoom, with a few plots on it.

Drag to pan, scroll to zoom about the cursor, and press Home to go back to
the start. Every frame draws the grid first, then the plots on top of it.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: quick start")
    clock = pygame.time.Clock()
    grid = pgp.Grid()  # fills the window and follows resizes

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)  # pan, zoom and the Home key

        grid.draw(screen)
        grid.plot(screen, np.sin, label="sin x")
        grid.plot(screen, lambda x: x**2 / 4 - 2, label="x²/4 − 2")
        grid.polar(screen, lambda a: 1 + np.cos(a), label="r = 1 + cos θ")
        grid.points(screen, [(-np.pi, 0), (0, 0), (np.pi, 0)])
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
