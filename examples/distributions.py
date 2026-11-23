"""Box and violin plots: comparing groups of samples.

A violin shows how a group's values are spread (a smoothed density,
mirrored), and a box plot its median, quartiles, whiskers and outliers.
Here a narrow box sits inside each violin. Each call takes a list of
groups, drawn at 1, 2, 3, ... (``start=1``), which ``draw(x_labels=...)``
names.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    rng = np.random.default_rng(3)
    groups = [
        rng.normal(170, 7, 500),
        rng.normal(163, 6, 500),
        140 + rng.gamma(4, 5, 500),
        np.concatenate([rng.normal(156, 5, 250), rng.normal(178, 5, 250)]),
    ]
    names = ["group A", "group B", "skewed", "two peaks"]

    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: box and violin plots")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(0.3, 4.7), y_range=(132, 205))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(
            screen, x_labels=dict(enumerate(names, start=1)), y_title="height (cm)"
        )
        grid.violin_plot(screen, groups, 1, label="distribution")
        grid.box_plot(
            screen, groups, 1, width=0.1, opacity=1, label="median and quartiles"
        )
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
