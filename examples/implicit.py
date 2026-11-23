"""Implicit curves: the contour lines of a landscape.

``implicit(F, c)`` draws the curve where ``F(x, y) = c``, however many
loops and pieces it has. Given several values of ``c`` it draws a contour
map. Choose a level with the slider: its curve is drawn in bold, and the
region above it is shaded with ``inequality``.
"""

import numpy as np
import pygame

import pygridplot as pgp


def hills(x: float, y: float) -> float:
    """The height of a landscape with two hills."""
    return np.exp(-((x - 1.5) ** 2 + (y - 0.5) ** 2) / 3) + 0.8 * np.exp(
        -((x + 2) ** 2 + (y + 1) ** 2) / 1.5
    )


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: implicit curves")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-6.5, 6.5))
    level = grid.slider("level c", (0.05, 0.95), value=0.35)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        c = level.value
        grid.draw(screen)
        grid.inequality(screen, lambda x, y: hills(x, y) > c, label=f"F > {c:.2f}")
        grid.implicit(
            screen,
            hills,
            np.linspace(0.1, 0.9, 9),
            color="gray",
            width=1,
            label="F = 0.1, 0.2, …, 0.9",
        )
        grid.implicit(screen, hills, c, width=3, label=f"F(x, y) = {c:.2f}")
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
