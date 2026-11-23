"""Bar charts: stacked bars, named categories and error bars.

Each bar stands on a baseline, so stacking is drawing each layer on top
of the totals so far. The bars stand at 1, 2, 3, ... (``start=1``), and
``draw(x_labels=...)`` names those positions; ``points(..., y_error=...)``
with ``radius=0`` draws error bars alone.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    walk = np.array([12, 15, 10, 14, 18, 40, 32])  # minutes a day
    cycle = np.array([22, 18, 25, 20, 26, 12, 8])
    bus = np.array([24, 26, 22, 25, 18, 4, 0])
    total = walk + cycle + bus
    spread = np.array([6, 5, 7, 6, 8, 12, 10])  # over four weeks

    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: bar charts")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(0.2, 7.8), y_range=(-6, 90))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_labels=dict(enumerate(days, start=1)), y_title="minutes")
        grid.bars(screen, walk, 1, label="walking")
        grid.bars(screen, walk + cycle, 1, baseline=walk, label="cycling")
        grid.bars(screen, total, 1, baseline=walk + cycle, label="bus")
        grid.points(
            screen,
            np.column_stack([np.arange(1, 8), total]),
            radius=0,
            y_error=spread,
            color="black",
            label="spread over four weeks",
        )
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
