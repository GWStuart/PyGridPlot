"""Taylor polynomials: approximating sin x.

``pgp.taylor(f, a, n)`` is the Taylor polynomial of degree ``n`` at ``a``,
built from exact derivatives. Choose the degree with the slider, and drag
the point along the x axis to move the centre: the polynomial hugs the
curve near it and leaves it further away.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: Taylor polynomials")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-9, 9))
    degree = grid.slider("degree", (0, 21), value=7, step=1)  # an int slider
    centre = grid.draggable((0.0, 0.0), lock="y")  # slides along the x axis

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        polynomial = pgp.taylor(np.sin, centre.x, degree.value)
        grid.draw(screen)
        grid.plot(screen, np.sin, width=4, opacity=0.4, label="sin x")
        grid.plot(screen, polynomial, label=f"degree {degree.value} at {centre.x:.2f}")
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
