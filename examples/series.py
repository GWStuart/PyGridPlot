"""Fourier series: partial sums closing in on a function, with their spectrum.

Press ``Space`` to switch between four functions on ``(-π, π)``, each
repeated with period ``2π``: a square wave, a sawtooth, a triangle wave and
a half-wave rectified sine. Drag the slider to change the number of
harmonics, from none (the mean alone) to 127.

The faded curve is the function, the bold one the partial sum
:func:`pygridplot.fourier` returns. Near a jump the partial sums overshoot
by about 9% however many harmonics there are (the Gibbs phenomenon), while
at a kink (the triangle wave) they converge quickly. The lower plane shows
the amplitude of each harmonic, ``√(aₖ² + bₖ²)``, from the series'
``cosines`` and ``sines``: they fall off like ``1/k`` for jumps and like
``1/k²`` for kinks. Both planes pan and zoom on their own.
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
TRACK = (-7.0, 7.0)
"""The slider's track, at y = -2.6 on the upper plane."""

FUNCTIONS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("square wave: sign x", np.sign),
    ("sawtooth: x", lambda x: x),
    ("triangle wave: |x|", np.abs),
    ("half-wave rectified sine: max(sin x, 0)", lambda x: np.maximum(np.sin(x), 0)),
]


def periodic(func: Callable[[np.ndarray], np.ndarray]) -> Callable[..., np.ndarray]:
    """Repeat a function's values on ``(-π, π)`` with period ``2π``."""
    return lambda x: func(np.remainder(x + math.pi, 2 * math.pi) - math.pi)


def harmonics(slider: pgp.DraggablePoint) -> int:
    """The number of harmonics: 0 to 127 along the track, doubling every 2."""
    position = (slider.x - TRACK[0]) / (TRACK[1] - TRACK[0])
    return round(2 ** (7 * position)) - 1


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
    plane = pgp.Plane(upper, x_range=(-10, 10))
    spectrum = pgp.Plane(lower, x_range=(-1, 33), y_range=(-0.25, 2.25))
    slider = plane.draggable((-1.0, -2.6), lock="y", x_range=TRACK, coordinates="never")
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
                plane.rect, spectrum.rect = layout(screen)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(FUNCTIONS)
                terms = -1  # compute the series again
            plane.handle_event(event)
            spectrum.handle_event(event)

        name, func = FUNCTIONS[current]
        if harmonics(slider) != terms:  # only when it changes (about a millisecond)
            terms = harmonics(slider)
            series = pgp.fourier(func, terms)

        plane.draw_grid(screen)
        plane.plot(screen, periodic(func), style=FADED, color=BLUE)
        plane.plot(screen, series, color=RED)
        plane.polyline(screen, [(x, -2.6) for x in TRACK], width=4, opacity=0.3)
        plane.draw_draggables(screen)

        spectrum.draw_grid(screen)
        amplitudes = np.hypot(series.cosines, series.sines)
        amplitudes[0] /= 2  # the mean, a₀/2
        k = np.arange(terms + 1, dtype=float)
        gap = np.full(k.size, np.nan)
        bars = np.column_stack([k, 0 * k, k, amplitudes, gap, gap]).reshape(-1, 2)
        spectrum.polyline(screen, bars, width=3, color=RED)
        spectrum.points(screen, np.column_stack([k, amplitudes]), radius=3, color=RED)

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
