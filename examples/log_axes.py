"""Logarithmic axes, and setting up a view from x and y ranges.

Press ``Space`` to switch between four planes:

1. **log-log**: power laws ``y = c·xᵏ`` are straight lines whose slope is
   the power ``k``, so data following one lines up.
2. **semi-log y**: exponential growth is a straight line.
3. **semi-log x**: the frequency response of a low-pass filter, the usual
   way to draw one (gain in decibels against frequency).
4. **linear, from ranges**: each axis fills the range it was given, like a
   graphing calculator's window settings, so the axes have different scales.

Each plane is created from ``x_range`` and ``y_range``. Press ``X`` or ``Y``
to switch the current plane's x or y axis between linear and log, and
``Home`` to return to its starting view. Pan and zoom as usual; zooming far
into a log axis reveals finer and finer subdivisions until it looks linear.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

RNG = np.random.default_rng(7)
DATA_X = np.geomspace(0.03, 300, 24)
DATA_Y = 0.8 * DATA_X**1.5 * RNG.lognormal(0.0, 0.15, DATA_X.size)
"""Noisy measurements that follow a power law with exponent 1.5."""


@dataclass
class Example:
    """One of the planes the example switches between.

    Attributes:
        title: Caption shown in the corner.
        grid: The plane, configured from ranges.
        draw: Draws the plots on top of the plane's grid.
    """

    title: str
    grid: pgp.Grid
    draw: Callable[[pgp.Grid, pygame.Surface], None]


def draw_power_laws(grid: pgp.Grid, screen: pygame.Surface) -> None:
    """Draw power laws with the measured data on top."""
    for power in (-1.0, 0.5, 1.0, 2.0, 3.0):
        grid.plot(screen, lambda x, k=power: x**k, width=2)
    grid.plot(screen, lambda x: 0.8 * x**1.5, color="black", width=1.5, opacity=0.6)
    grid.points(screen, np.column_stack((DATA_X, DATA_Y)), edge_color="white")


def draw_growth(grid: pgp.Grid, screen: pygame.Surface) -> None:
    """Draw exponential growth at three rates, and a power law for contrast."""
    for rate in (0.5, 1.0, 2.0):
        grid.plot(screen, lambda x, r=rate: np.exp(r * x))
    grid.plot(screen, lambda x: x**3, width=1.5, opacity=0.7)
    days = np.arange(0, 11)
    grid.points(screen, np.column_stack((days, 3 * 1.9**days)), marker="square")


def draw_filter(grid: pgp.Grid, screen: pygame.Surface) -> None:
    """Draw the gain of first- and second-order low-pass filters, in decibels."""
    cutoff = 1000.0
    for order in (1, 2):
        grid.plot(
            screen,
            lambda f, n=order: -10 * n * np.log10(1 + (f / cutoff) ** 2),
        )
    grid.polyline(screen, [(10, -3), (1e5, -3)], color="gray", width=1)
    grid.points(screen, (cutoff, -3.0), edge_color="white", radius=5)


def draw_waves(grid: pgp.Grid, screen: pygame.Surface) -> None:
    """Draw a few waves on a plane whose axes have different scales."""
    grid.plot(screen, np.sin)
    grid.plot(screen, lambda x: 0.5 * np.cos(3 * x))
    grid.plot(screen, lambda x: np.exp(-x / 4) * np.sin(4 * x), width=1.5)


def make_examples() -> list[Example]:
    """Create the planes, each set up from its x and y ranges."""
    return [
        Example(
            "log-log: power laws are straight lines",
            pgp.Grid(x_range=(0.01, 1000), y_range=(0.01, 1e4), log_x=True, log_y=True),
            draw_power_laws,
        ),
        Example(
            "semi-log y: exponential growth is a straight line",
            pgp.Grid(x_range=(-1, 11), y_range=(0.5, 1e6), log_y=True),
            draw_growth,
        ),
        Example(
            "semi-log x: a low-pass filter's gain in dB",
            pgp.Grid(x_range=(10, 1e5), y_range=(-70, 10), log_x=True),
            draw_filter,
        ),
        Example(
            "linear, set up from x and y ranges",
            pgp.Grid(x_range=(-1, 13), y_range=(-1.5, 1.5)),
            draw_waves,
        ),
    ]


def draw_caption(
    screen: pygame.Surface, font: pygame.font.Font, example: Example
) -> None:
    """Draw the example's title and the keys that control it."""
    grid = example.grid
    kinds = ["log" if log else "linear" for log in (grid.log_x, grid.log_y)]
    lines = [
        example.title,
        f"x {kinds[0]}, y {kinds[1]}   (Space: next   X / Y: toggle log)",
    ]
    rendered = [font.render(text, True, (40, 40, 40)) for text in lines]
    panel = pygame.Surface(
        (max(r.get_width() for r in rendered) + 20, 12 + 20 * len(rendered)),
        pygame.SRCALPHA,
    )
    panel.fill((255, 255, 255, 225))
    for i, text in enumerate(rendered):
        panel.blit(text, (10, 6 + 20 * i))
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: log axes and ranges")
    font = pygame.font.Font(bundled_font_path(), 14)
    examples = make_examples()
    clock = pygame.time.Clock()
    current = 0

    running = True
    while running:
        example = examples[current]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(examples)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                example.grid.log_x = not example.grid.log_x
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_y:
                example.grid.log_y = not example.grid.log_y
            example.grid.handle_event(event)

        example = examples[current]
        example.grid.draw(screen)
        example.draw(example.grid, screen)
        draw_caption(screen, font, example)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
