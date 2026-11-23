"""Sliders: shape a wave with three parameters.

Sliders are created once, before the loop, and read every frame. Drag a
knob, press its track, or scroll over a slider to move it a step. The
frequency slider is geometric (``log=True``): equal distances along it
multiply the value by the same factor.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: sliders")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-8, 8))
    a = grid.slider("amplitude a", (0, 3), value=2)
    b = grid.slider("frequency b", (0.25, 4), value=1.5, log=True)
    c = grid.slider("phase c", (-np.pi, np.pi), value=0.5)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)  # sliders first, then pan and zoom

        grid.draw(screen)
        grid.plot(screen, np.sin, color="gray", width=1.5, label="sin x")
        grid.plot(
            screen,
            lambda x: a.value * np.sin(b.value * x + c.value),
            label="a sin(bx + c)",
        )
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
