"""The smallest useful PyGridPlot program: an interactive grid in a window.

Drag with the left mouse button to pan, scroll to zoom about the cursor,
use the arrow keys and ``+``/``-`` too, and press ``Home`` or ``0`` to reset.
The window is resizable, and the plane follows it.
"""

from __future__ import annotations

import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and run the main loop until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: basic grid")
    grid = pgp.Grid()  # rect=None: fill the window and follow resizes
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
