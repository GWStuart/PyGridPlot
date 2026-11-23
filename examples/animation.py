"""Animation: a rolling wheel draws a cycloid.

Nothing is created, moved or deleted: every frame draws the wheel, the
pen's velocity and the curve traced so far from the time alone. That is
all animation takes, and you can still pan and zoom while it runs.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: animation")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-0.8, 13.5))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        t = pygame.time.get_ticks() / 1000  # seconds
        turn = 0.3 + (1.5 * t) % (4 * np.pi)  # the angle turned: two turns, again
        pen = np.array([turn - np.sin(turn), 1 - np.cos(turn)])  # on the rim
        velocity = np.array([1 - np.cos(turn), np.sin(turn)])

        grid.draw(screen)
        grid.parametric(
            screen,
            lambda s: s - np.sin(s),
            lambda s: 1 - np.cos(s),
            (0, turn),
            width=3,
            label="cycloid",
        )
        grid.parametric(
            screen,
            lambda s: turn + np.cos(s),
            lambda s: 1 + np.sin(s),
            (0, 2 * np.pi),
            color="gray",
            label="wheel",
        )
        for k in range(6):  # spokes, turning with the wheel
            angle = -np.pi / 2 - turn + k * np.pi / 3
            spoke_end = (turn + np.cos(angle), 1 + np.sin(angle))
            grid.polyline(screen, [(turn, 1), spoke_end], color="gray", width=1)
        grid.arrow(screen, pen, pen + velocity, label="the pen's velocity")
        grid.points(screen, [pen], radius=6)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
