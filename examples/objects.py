"""A picture made of objects: made once, changed through their attributes.

* **A wave you can shape**: drag the red point to move its crest, and the
  blue slider on the left to change its height. The wave is a
  ``pgp.Function`` whose Python function reads the two points.
* **A tangent**: the green point runs along the wave (``curve=wave``), and
  the tangent through it uses the wave's exact derivative.
* **A point on a circle**, with its radius drawn as a polyline whose end
  follows the point, and its angle in the caption.

Press ``Space`` to hide or show the circle (``visible``), ``W`` to switch
the wave between a sine and a triangle wave (``wave.func = ...``). Pan and
zoom as usual; a press on a point drags it.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path


class Picture:
    """The objects of the picture, made once."""

    def __init__(self) -> None:
        self.crest = pgp.Point((math.pi / 2, 2.0), draggable=True, color=(200, 60, 60))
        self.height = pgp.Point(
            (-7.0, 1.5),
            draggable=True,
            lock="x",
            y_range=(0.25, 4.0),
            color=(60, 90, 200),
        )
        self.track = pgp.Polyline(
            [(-7, 0.25), (-7, 4)], width=4, opacity=0.35, color="gray"
        )
        self.wave = pgp.Function(self.sine, width=3)
        self.runner = pgp.Point(
            curve=self.wave, t=0.0, draggable=True, color=(40, 150, 90)
        )
        slope = self.wave.derivative()  # follows the wave, whatever its func
        self.tangent = pgp.Function(
            lambda x: slope(self.runner.x) * (x - self.runner.x) + self.runner.y,
            width=1.5,
            color="gray",
        )
        self.circle = pgp.ParametricCurve(
            lambda t: 5 + 1.5 * np.cos(t), lambda t: -3 + 1.5 * np.sin(t), color="gray"
        )
        self.on_circle = pgp.Point(
            curve=self.circle, t=0.8, draggable=True, color=(150, 70, 170)
        )
        self.radius = pgp.Polyline(
            [(5, -3), (lambda: self.on_circle.x, lambda: self.on_circle.y)]
        )
        self.centre = pgp.Point((5, -3), radius=3, color="gray")

    def sine(self, x: np.ndarray) -> np.ndarray:
        """A sine wave with its crest at the red point and the slider's height."""
        a, crest = self.height.y, self.crest
        return a * np.sin(x - crest.x + math.pi / 2) + crest.y - a

    def triangle(self, x: np.ndarray) -> np.ndarray:
        """A triangle wave with the same crest and height."""
        a, crest = self.height.y, self.crest
        phase = (x - crest.x) / (2 * math.pi)
        shape = 1 - 4 * np.abs(phase - np.round(phase))  # 1 at the crest
        return a * shape + crest.y - a

    def objects(self) -> list[pgp.objects.PlotObject]:
        """Return the objects in drawing order (points last, on top)."""
        return [
            self.track,
            self.wave,
            self.tangent,
            self.circle,
            self.radius,
            self.centre,
            self.crest,
            self.height,
            self.runner,
            self.on_circle,
        ]

    def toggle_circle(self) -> None:
        """Hide or show the circle and what belongs to it."""
        for obj in (self.circle, self.radius, self.centre, self.on_circle):
            obj.visible = not obj.visible

    def switch_wave(self) -> None:
        """Switch the wave between a sine and a triangle wave."""
        self.wave.func = self.triangle if self.wave.func == self.sine else self.sine

    def caption(self) -> list[str]:
        """Return the caption lines, read from the objects."""
        point = self.on_circle
        angle = math.degrees(math.atan2(point.y + 3, point.x - 5)) % 360
        return [
            f"crest ({self.crest.x:.2f}, {self.crest.y:.2f}), "
            f"height {self.height.y:.2f}",
            f"tangent at x = {self.runner.x:.2f}",
            f"angle on the circle: {angle:.0f}°",
            "Space: hide/show the circle   W: switch the wave",
        ]


def draw_caption(
    screen: pygame.Surface, font: pygame.font.Font, lines: list[str]
) -> None:
    """Draw caption lines in a panel at the top left."""
    texts = [font.render(line, True, (40, 40, 40)) for line in lines]
    width = max(text.get_width() for text in texts) + 20
    panel = pygame.Surface((width, 22 * len(texts) + 12), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 225))
    for number, text in enumerate(texts):
        panel.blit(text, (10, 6 + 22 * number))
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: objects")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-8, 8))
    picture = Picture()
    plane.add(*picture.objects())
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                picture.toggle_circle()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                picture.switch_wave()
            plane.handle_event(event)
        plane.draw(screen)
        draw_caption(screen, font, picture.caption())
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
