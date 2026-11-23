"""Sequences: limits, partial sums and a recurrence, drawn with ``plane.sequence``.

Press ``Space`` to switch between three sets:

1. **Limits**: ``(1 + 1/n)ⁿ`` creeping up to ``e``, the alternating
   ``(-1)ⁿ / n`` closing in on 0 from both sides, and ``sin(n) / √n``, each
   with its limit as a faded line, the last two shifted down, with stems
   from their limits (``baseline=``). They are functions of ``n``: zoom out
   and they keep going, however far, with the terms thinned smoothly once
   they are closer than a quarter pixel.
2. **Partial sums of series**: the harmonic series ``Σ 1/k``, which
   diverges like ``log n + 0.5772`` (the faded curve), against ``Σ 1/k²``, which
   converges to ``π²/6``, and the alternating ``Σ (-1)ᵏ⁺¹/k``, to ``log 2``.
   They are arrays, built with ``np.cumsum``. On a log x axis (``X``) the
   harmonic series grows along a straight line.
3. **The logistic map** ``xₙ₊₁ = r xₙ (1 - xₙ)``: drag the slider for
   ``r``. Below 3 the sequence settles on one value, then on 2, 4, 8, ...
   in turn (period doubling), and beyond about 3.57 it is chaotic. A
   recurrence is a few lines of Python building a list.
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
GREEN = (40, 150, 90)
FADED = pgp.LineStyle(width=1.5, opacity=0.5)
TERMS = 2000
"""Terms of the partial sums."""


def level(value: float) -> Callable[[np.ndarray], np.ndarray]:
    """A horizontal line at ``value``, as a function to plot."""
    return lambda x: 0 * x + value


def draw_limits(plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
    """Draw three sequences converging to their limits."""
    plane.plot(screen, level(math.e), style=FADED, color=BLUE)
    plane.sequence(screen, lambda n: (1 + 1 / n) ** n, start=1, color=BLUE)
    # Stems from the limit show the terms closing in on it.
    plane.sequence(screen, lambda n: (-1) ** n / n - 2, start=1, baseline=-2, color=RED)
    plane.sequence(
        screen,
        lambda n: np.sin(n) / np.sqrt(n) - 4,
        start=1,
        baseline=-4,
        color=GREEN,
        radius=2.5,
    )
    return [
        "blue: (1 + 1/n)ⁿ → e    red: (-1)ⁿ/n - 2 → -2    green: sin(n)/√n - 4 → -4",
        "stems from each limit; zoom out: the terms go on, thinned smoothly",
    ]


class PartialSums:
    """The partial sums of three series, as arrays."""

    def __init__(self) -> None:
        k = np.arange(1, TERMS + 1, dtype=float)
        self.harmonic = np.cumsum(1 / k)
        self.squares = np.cumsum(1 / k**2)
        self.alternating = np.cumsum((-1) ** (k + 1) / k)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        gamma = 0.5772156649015329
        plane.plot(screen, lambda x: np.log(x) + gamma, style=FADED, color=BLUE)
        plane.sequence(screen, self.harmonic, start=1, color=BLUE, width=0)
        plane.plot(screen, level(math.pi**2 / 6), style=FADED, color=RED)
        plane.sequence(screen, self.squares, start=1, color=RED, width=0)
        plane.plot(screen, level(math.log(2)), style=FADED, color=GREEN)
        plane.sequence(screen, self.alternating, start=1, color=GREEN, width=0)
        return [
            "blue: Σ 1/k diverges like log n + 0.5772    red: Σ 1/k² → π²/6"
            "    green: Σ (-1)ᵏ⁺¹/k → log 2",
            "press X: on a log x axis the harmonic series grows along a line",
        ]


class Logistic:
    """The logistic map, with a slider for its parameter."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.slider = plane.slider("r", (2.5, 4.0), value=3.46, corner="top-right")
        self.points = (self.slider,)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        r, x = self.slider.value, [0.2]
        for _ in range(300):
            x.append(r * x[-1] * (1 - x[-1]))
        plane.sequence(screen, [value * 4 for value in x], color=BLUE, width=1)
        plane.draw_draggables(screen, *self.points)
        return [
            f"xₙ₊₁ = r xₙ (1 - xₙ) with r = {r:.4f}, from x₀ = 0.2 (drawn ×4)",
            "drag the slider for r: one value, then 2, 4, 8 ... (period doubling), "
            "then chaos",
        ]


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
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: sequences")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-2, 40), y_range=(-5, 5))
    sums, logistic = PartialSums(), Logistic(plane)
    sets = [draw_limits, sums.draw, logistic.draw]
    current = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(sets)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                plane.log_x = not plane.log_x
            plane.handle_event(event)

        plane.draw_grid(screen)
        lines = sets[current](plane, screen)
        draw_caption(screen, font, [*lines, "Space: next set   X: log x axis"])
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
