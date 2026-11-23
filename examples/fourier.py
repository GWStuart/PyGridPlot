"""Fourier series: adding up a square wave.

``pgp.fourier(f, n)`` is the sum of the first ``n`` harmonics of ``f``,
over one period (2π unless you give another). Add terms with the slider:
the sum closes in on the square wave, except for the overshoot beside each
jump (Gibbs' phenomenon), which narrows but never shrinks.
"""

import numpy as np
import pygame

import pygridplot as pgp


def square(x: float) -> float:
    """A square wave: 1 where sin x is positive, -1 where it is negative."""
    return np.sign(np.sin(x))


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: Fourier series")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-7, 7))
    terms = grid.slider("terms", (1, 40), value=7, step=1)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        series = pgp.fourier(square, terms.value)
        grid.draw(screen)
        grid.plot(screen, square, width=4, opacity=0.4, label="square wave")
        grid.plot(screen, series, label=f"the first {terms.value} terms")
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
