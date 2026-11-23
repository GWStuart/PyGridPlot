"""Functions: plotting y = f(x), awkward functions included.

Poles, jumps, square roots that stop and functions written with ``math``
for single numbers all plot correctly: asymptotes are not joined up,
steps stay steps, and curves end exactly where they stop being defined.
Zoom in on the ends and the jumps to see. ``domain=`` limits a function
to part of the x axis.
"""

import math

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: functions")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-6.5, 6.5), y_range=(-4.5, 4.5))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_title="x", y_title="y")
        grid.plot(screen, np.tan, label="tan x: poles")
        grid.plot(screen, np.floor, label="⌊x⌋: jumps")
        grid.plot(screen, lambda x: np.sqrt(9 - x**2), label="√(9 − x²): ends")
        grid.plot(screen, math.gamma, domain=(0, 7), label="Γ(x) for x > 0, with math")
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
