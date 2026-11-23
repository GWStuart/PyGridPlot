"""Fourier series: partial sums closing in on a function, with their spectrum.

Press ``Space`` to switch between four functions on ``(-π, π)``, each
repeated with period ``2π``: a square wave, a sawtooth, a triangle wave and
a half-wave rectified sine. Drag the slider (or turn the mouse wheel over
it) to change the number of harmonics, from 1 to 128: it is geometric, so
there is as much room for 1 to 11 as for 11 to 128.

The faded curve is the function, the bold one the partial sum
:func:`pygridplot.fourier` returns. Near a jump the partial sums overshoot
by about 9% however many harmonics there are (the Gibbs phenomenon), while
at a kink (the triangle wave) they converge quickly. The lower plane shows
the amplitude of each harmonic, ``√(aₖ² + bₖ²)``, the series'
``amplitudes``, drawn with :meth:`pygridplot.Grid.sequence`: they fall
off like ``1/k`` for jumps and like ``1/k²`` for kinks. Both planes pan and
zoom on their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

BLUE = (40, 100, 200)
RED = (200, 50, 60)
FADED = pgp.LineStyle(width=2.0, opacity=0.4)

FUNCTIONS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("square wave: sign x", np.sign),
    ("sawtooth: x", lambda x: x),
    ("triangle wave: |x|", np.abs),
    ("half-wave rectified sine: max(sin x, 0)", lambda x: np.maximum(np.sin(x), 0)),
]


def periodic(func: Callable[[np.ndarray], np.ndarray]) -> Callable[..., np.ndarray]:
    """Repeat a function's values on ``(-π, π)`` with period ``2π``."""
    return lambda x: func(np.remainder(x + math.pi, 2 * math.pi) - math.pi)


def layout(screen: pygame.Surface) -> tuple[pygame.Rect, pygame.Rect]:
    """Return the rectangles of the upper and lower planes."""
    width, height = screen.get_size()
    split = int(height * 0.7)
    return pygame.Rect(0, 0, width, split), pygame.Rect(0, split, width, height - split)


def draw_caption(
    screen: pygame.Surface, font: pygame.font.Font, lines: list[str]
) -> None:
    """Draw caption lines in a panel at the top left."""
    texts = [font.render(line, True, (40, 40, 40)) for line in lines]
    panel = pygame.Surface(
        (
            max(text.get_width() for text in texts) + 20,
            sum(text.get_height() for text in texts) + 12,
        ),
        pygame.SRCALPHA,
    )
    panel.fill((255, 255, 255, 225))
    y = 6
    for text in texts:
        panel.blit(text, (10, y))
        y += text.get_height()
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: Fourier series")
    font = pygame.font.Font(bundled_font_path(), 14)
    upper, lower = layout(screen)
    grid = pgp.Grid(upper, x_range=(-10, 10))
    spectrum = pgp.Grid(lower, x_range=(-1, 33), y_range=(-0.25, 2.25))
    harmonics = grid.slider("harmonics", (1, 128), value=7, step=1, log=True)
    current = 0
    terms = -1
    series = pgp.fourier(FUNCTIONS[0][1], 0)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWSIZECHANGED:
                grid.rect, spectrum.rect = layout(screen)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(FUNCTIONS)
                terms = -1  # compute the series again
            grid.handle_event(event)
            spectrum.handle_event(event)

        name, func = FUNCTIONS[current]
        if harmonics.value != terms:  # only when it changes (about a millisecond)
            terms = harmonics.value
            series = pgp.fourier(func, terms)

        grid.draw(screen)
        grid.plot(screen, periodic(func), style=FADED, color=BLUE)
        grid.plot(screen, series, color=RED)
        grid.draw_controls(screen)

        spectrum.draw(screen)
        spectrum.sequence(screen, series.amplitudes, width=3, color=RED)  # k = 0 ... n

        draw_caption(
            screen,
            font,
            [
                f"{name} on (-π, π), with {terms} harmonics",
                "drag the slider for the number of harmonics; Space: next function",
                "lower plane: the amplitude of each harmonic",
            ],
        )
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
