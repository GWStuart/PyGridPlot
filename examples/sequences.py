"""Sequences: the terms of a sequence, and its partial sums.

``sequence(terms)`` draws the terms ``aₙ`` as stems, where ``terms`` is a
function of ``n`` (or a list of values). The alternating harmonic series
1 − 1/2 + 1/3 − … converges to ln 2: its partial sums, drawn as points,
close in on it from either side. Zoom out to see further along.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: sequences")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-1, 25), y_range=(-0.7, 1.2))
    n = np.arange(1, 1001)
    partial_sums = np.cumsum((-1.0) ** (n + 1) / n)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_title="n")
        grid.plot(screen, lambda x: 0 * x + np.log(2), color="gray", label="ln 2")
        grid.sequence(
            screen, lambda n: (-1.0) ** (n + 1) / n, start=1, label="aₙ = (−1)ⁿ⁺¹/n"
        )
        grid.points(
            screen, np.column_stack([n, partial_sums]), label="partial sums a₁ + … + aₙ"
        )
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
