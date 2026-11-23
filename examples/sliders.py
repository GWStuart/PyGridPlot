"""Sliders: values the user sets, in panels that stay put while you pan and zoom.

Press ``Space`` to switch between two sets:

1. **A wave** ``y = a sin(k x + φ) + c``: ``a`` takes any value, ``k`` is
   on a geometric track (from 0.1 to 10, as much room for 0.1 to 1 as for
   1 to 10), and the phase ``φ`` moves in steps of π/12. They are in the
   bottom-left corner; ``c`` is on its own in the top-right.
2. **Lissajous figures** ``(sin(p t + δ), sin(q t))``: ``p`` and ``q`` are
   whole numbers (their sliders have ``step=1``, so their values are
   integers), and ``δ`` shifts the phase. The caption counts the frames in
   which a slider moved, from their ``moved`` flags.

Press a track to move the handle there, drag the handle, or turn the mouse
wheel over a slider to move it a step. Pan and zoom as usual: the sliders
stay where they are, and the plane never pans from under their panels.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

BLUE = (40, 100, 200)
RED = (200, 50, 60)


class Wave:
    """A sine wave set by four sliders in two corners."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.a = plane.slider("a", (0.0, 3.0), value=1.5, color=BLUE)
        self.k = plane.slider("k", (0.1, 10.0), value=1.0, log=True, color=BLUE)
        self.phase = plane.slider(
            "φ", (-math.pi, math.pi), value=0.0, step=math.pi / 12, color=BLUE
        )
        self.c = plane.slider(
            "c", (-3.0, 3.0), value=0.0, color=RED, corner="top-right"
        )
        self.sliders = (self.a, self.k, self.phase, self.c)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a, k, phase, c = (s.value for s in self.sliders)
        plane.plot(screen, lambda x: a * np.sin(k * x + phase) + c, color=BLUE)
        plane.plot(screen, lambda x: 0 * x + c, color=RED, width=1.5, opacity=0.6)
        plane.draw_draggables(screen, *self.sliders)
        return [f"y = {a:g} sin({k:g} x + {phase / math.pi:.3g}π) + {c:g}"]


class Lissajous:
    """A Lissajous figure with whole-number frequencies."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.p = plane.slider("p", (1, 9), value=3, step=1)
        self.q = plane.slider("q", (1, 9), value=2, step=1)
        self.delta = plane.slider("δ", (0.0, math.pi), value=math.pi / 2)
        self.sliders = (self.p, self.q, self.delta)
        self.changes = 0  # how many frames a slider had moved in

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        p, q, delta = self.p.value, self.q.value, self.delta.value
        if any(s.moved for s in self.sliders):  # true for one frame after a move
            self.changes += 1
        plane.parametric(
            screen,
            lambda t: 4 * np.sin(p * t + delta),
            lambda t: 4 * np.sin(q * t),
            (0, 2 * math.pi),
            color=RED,
        )
        plane.draw_draggables(screen, *self.sliders)
        ratio = math.gcd(p, q)  # p and q are ints, ready for math.gcd
        return [
            f"x = sin({p} t + {delta:.3g}), y = sin({q} t): "
            f"{q // ratio} loops across, {p // ratio} up and down",
            f"the sliders have moved in {self.changes} frames (slider.moved)",
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
    pygame.display.set_caption("PyGridPlot: sliders")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-8, 8))
    sets: list[Wave | Lissajous] = [Wave(plane), Lissajous(plane)]
    current = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(sets)
            plane.handle_event(event)  # sliders first, then pan and zoom

        plane.draw_grid(screen)
        lines = sets[current].draw(plane, screen)  # draws only its own sliders
        draw_caption(screen, font, [*lines, "Space: next set"])
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
