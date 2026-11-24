"""Controls: a rolling circle driven by a slider that plays, and checkboxes.

The slider ``t`` plays by itself, in a loop: press its play button to stop
and start it, or drag it. As the circle rolls along the x axis, a point on
its rim traces a cycloid. The checkboxes show or hide the circle and the
whole path; your code reads their ``value`` every frame.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: controls")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-1.0, 14.0), center=(6.5, 1.3))
    t = grid.slider("t", (0, 4 * np.pi), play="loop", duration=8)
    t.playing = True  # from the start
    circle = grid.checkbox("the rolling circle", value=True)
    path = grid.checkbox("the whole path")
    color = grid.theme.palette[0]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        a = t.value
        point = (a - np.sin(a), 1 - np.cos(a))  # on the rim, at angle a
        grid.draw(screen)
        if path.value:
            grid.parametric(
                screen,
                lambda s: s - np.sin(s),
                lambda s: 1 - np.cos(s),
                (0, 4 * np.pi),
                color="gray",
                width=1.5,
                dash="dashed",
            )
        if a > 0:
            grid.parametric(
                screen, lambda s: s - np.sin(s), lambda s: 1 - np.cos(s), (0, a)
            )
        if circle.value:
            grid.parametric(
                screen, lambda s: a + np.cos(s), lambda s: 1 + np.sin(s), (0, 2 * np.pi)
            )
            grid.polyline(screen, [(a, 1), point], color="gray", width=1.5)
        grid.points(screen, [point], color=color)
        grid.draw_controls(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
