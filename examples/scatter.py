"""Scatter plots: colour by a third value, and a fitted line.

``points`` takes a colour for each point: ``pgp.color_map`` turns values
into colours, and ``color_bar`` shows which colour is which value.
``pgp.data_range`` gives the y axis a range that shows all the data; the x
axis leaves room on the right for the legend and the colour bar.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    rng = np.random.default_rng(7)
    study = rng.uniform(0, 10, 300)  # hours of study
    sleep = rng.uniform(4, 9, 300)  # hours of sleep the night before
    score = 35 + 4.5 * study + 3 * (sleep - 6.5) + rng.normal(0, 5, 300)
    slope, intercept = np.polyfit(study, score, 1)
    colors = pgp.color_map(sleep, "viridis", (4, 9))

    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: scatter plot")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(0, 13), y_range=pgp.data_range(score))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen, x_title="hours of study", y_title="score")
        grid.points(
            screen, np.column_stack([study, score]), color=colors, label="students"
        )
        grid.plot(
            screen,
            lambda x: slope * x + intercept,
            color="black",
            width=2,
            label=f"best fit: {slope:.1f} points an hour",
        )
        grid.legend(screen)
        grid.color_bar(screen, "viridis", (4, 9), title="hours of sleep")  # below it
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
