"""Plotting data: scatter plots, error bars, steps, bars, histograms, box plots.

Press ``Space`` to switch between sets:

1. **Measurements and a model**: a pendulum's period against its length,
   with error bars on both, and the model ``T = 2π √(L / g)`` drawn over
   them. Drag the slider to change ``g`` and watch the model leave the data.
2. **Particles**: two hundred particles orbiting, each coloured by its
   speed (``pgp.color_map``, with a colour bar to read it by) and sized by
   its mass, every frame.
3. **A histogram that fills in**: the mean of ``n`` uniform random numbers,
   sampled over and over. Its histogram (a density) grows frame by frame
   in bins that never move, and approaches the normal density drawn over
   it, as the central limit theorem says. The slider sets ``n``.
4. **Dice**: the sum of two dice, rolled again and again. Bars show how
   often each sum came up, points with error bars (one standard deviation)
   what to expect, and the steps the cumulative fraction.
5. **Groups compared**: reaction times in four conditions, as violins
   with a narrow box plot inside each, the conditions named on the x axis.
6. **Stacked bars**: where electricity came from, year by year, each
   source standing on the ones below, the years named on the axis.

``P`` pauses the moving sets. Pan and zoom as usual.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path
from pygridplot.render.pygame_renderer import PygameRenderer

BLUE = (40, 100, 200)
RED = (200, 50, 60)
GREEN = (40, 150, 90)
GRAY = (90, 90, 100)


class Demo(Protocol):
    """One set of the example."""

    grid: pgp.Grid

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set (advancing it if ``running``); return its caption."""
        ...


class Pendulum:
    """Measured periods of a pendulum, with the model through them."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(
            x_range=(-0.1, 1.6), y_range=(-0.3, 2.8), renderer=renderer
        )
        rng = np.random.default_rng(1)
        self.lengths = np.linspace(0.2, 1.4, 9)
        self.length_error = np.full(9, 0.03)
        true = 2 * np.pi * np.sqrt(self.lengths / 9.81)
        self.periods = true + rng.normal(0, 0.04, 9)
        self.period_error = 0.03 + 0.03 * rng.random(9)
        self.g = self.grid.slider("g", (5.0, 15.0), value=9.81, step=0.01, color=RED)

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        grid = self.grid
        grid.draw(screen, x_title="length L (m)", y_title="period T (s)")
        g = self.g.value
        grid.plot(
            screen,
            lambda length: 2 * np.pi * np.sqrt(length / g),
            domain=(0, 2),
            color=RED,
            width=2,
            label=f"T = 2π √(L / {g:.2f})",
        )
        grid.points(
            screen,
            np.column_stack((self.lengths, self.periods)),
            x_error=self.length_error,
            y_error=self.period_error,
            color=BLUE,
            edge_color="white",
            label="measured",
        )
        grid.legend(screen, corner="top-left")
        grid.draw_controls(screen, self.g)
        residuals = self.periods - 2 * np.pi * np.sqrt(self.lengths / g)
        return f"rms residual {math.sqrt(np.mean(residuals**2)) * 1000:.0f} ms"


class Particles:
    """Particles orbiting the origin, coloured by speed and sized by mass."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(x_range=(-6, 6), renderer=renderer)
        rng = np.random.default_rng(2)
        self.radius = rng.uniform(1.0, 5.0, 200)
        self.angle = rng.uniform(0, 2 * np.pi, 200)
        self.mass = rng.uniform(0.5, 3.0, 200)
        self.speed = 1.2 / np.sqrt(self.radius)  # faster nearer the middle

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        if running:
            self.angle += self.speed / 60
        grid = self.grid
        grid.draw(screen)
        x, y = self.radius * np.cos(self.angle), self.radius * np.sin(self.angle)
        grid.points(
            screen,
            np.column_stack((x, y)),
            color=pgp.color_map(self.speed, "plasma", SPEEDS),
            radius=2 + 2 * self.mass,
            edge_color="white",
            edge_width=1,
        )
        grid.color_bar(screen, "plasma", SPEEDS, title="speed")
        grid.text(screen, "the bigger, the heavier", (0, -5.5))
        return "Particles, coloured by speed and sized by mass"


SPEEDS = (0.5, 1.2)
"""The range of the particles' speeds, fixed so a colour keeps its meaning."""


class Groups:
    """Reaction times in four conditions, as violins and box plots."""

    NAMES = ("rested", "tired", "coffee", "noise")

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(x_range=(-0.7, 3.7), y_range=(0, 900), renderer=renderer)
        rng = np.random.default_rng(5)
        self.groups = [
            rng.lognormal(math.log(300), 0.2, 400),
            rng.lognormal(math.log(380), 0.3, 400),
            rng.lognormal(math.log(280), 0.15, 400),
            np.r_[rng.lognormal(math.log(320), 0.2, 300), rng.normal(650, 60, 60)],
        ]

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        grid = self.grid
        grid.draw(screen, x_labels=self.NAMES, y_title="reaction time (ms)")
        grid.violin_plot(screen, self.groups, color=BLUE, label="density")
        grid.box_plot(
            screen, self.groups, width=0.1, color=GRAY, opacity=0.9, label="quartiles"
        )
        grid.legend(screen)
        return "Groups compared: violins with box plots inside"


class Energy:
    """Where electricity came from, year by year, as stacked bars."""

    SOURCES = (("coal", GRAY), ("gas", RED), ("wind", BLUE), ("solar", GREEN))

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(x_range=(-0.8, 6.8), y_range=(-8, 110), renderer=renderer)
        years = np.arange(7)
        self.years = [str(2018 + k) for k in years]
        self.shares = np.array(
            [
                60 - 6 * years,
                25 + 0.5 * years,
                10 + 3 * years,
                5 + 2.5 * years,
            ],
            dtype=np.float64,
        )

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        grid = self.grid
        grid.draw(screen, x_labels=self.years, y_title="share (%)")
        below = np.zeros(self.shares.shape[1])
        for (name, color), share in zip(self.SOURCES, self.shares, strict=True):
            grid.bars(screen, below + share, baseline=below, color=color, label=name)
            below = below + share
        grid.legend(screen, corner="top-left")
        return "Stacked bars, each source on the ones below"


class Means:
    """A histogram of sample means, filling in frame by frame."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(
            x_range=(-0.05, 1.05), y_range=(-0.3, 7), renderer=renderer
        )
        self.n = self.grid.slider("n", (1, 30), value=4, step=1, color=GREEN)
        self.rng = np.random.default_rng(3)
        self.samples = np.zeros(0)
        self.shown_n = self.n.value

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        n = self.n.value
        if n != self.shown_n:  # a new n: start again
            self.samples, self.shown_n = np.zeros(0), n
        if running and self.samples.shape[0] < 50_000:
            means = self.rng.random((200, n)).mean(axis=1)
            self.samples = np.append(self.samples, means)
        grid = self.grid
        grid.draw(screen, x_title="mean of n uniform numbers", y_title="density")
        grid.histogram(
            screen, self.samples, width=0.02, density=True, color=GREEN, label="sampled"
        )
        sd = math.sqrt(1 / (12 * n))
        grid.plot(
            screen,
            lambda x: (
                np.exp(-(((x - 0.5) / sd) ** 2) / 2) / (sd * math.sqrt(2 * math.pi))
            ),
            color=GRAY,
            width=2,
            label="normal",
        )
        grid.legend(screen)
        grid.draw_controls(screen, self.n)
        return f"{self.samples.shape[0]} means of {n} numbers"


class Dice:
    """The sum of two dice: observed, expected and cumulative."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.grid = pgp.Grid(x_range=(1, 13), y_range=(-0.08, 1.1), renderer=renderer)
        self.rng = np.random.default_rng(4)
        self.counts = np.zeros(13)
        ways = np.array([0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        self.chance = ways / 36

    def draw(self, screen: pygame.Surface, running: bool) -> str:
        """Draw the set; return its caption."""
        if running:
            sums = self.rng.integers(1, 7, (20, 2)).sum(axis=1)
            self.counts += np.bincount(sums, minlength=13)
        rolls = self.counts.sum()
        grid = self.grid
        grid.draw(screen, x_title="sum of two dice", y_title="fraction")
        sums = np.arange(2, 13)
        observed = self.counts[2:] / max(rolls, 1)
        grid.bars(screen, observed, start=2, color=BLUE, label="observed")
        spread = np.sqrt(self.chance[2:] * (1 - self.chance[2:]) / max(rolls, 1))
        grid.points(
            screen,
            np.column_stack((sums, self.chance[2:])),
            y_error=spread,
            color=RED,
            radius=3.5,
            label="expected ± 1 sd",
        )
        cumulative = np.cumsum(observed)
        grid.polyline(
            screen,
            np.column_stack((np.r_[1.5, sums + 0.5], np.r_[0, cumulative])),
            step="pre",
            color=GRAY,
            width=1.5,
            label="cumulative",
        )
        grid.legend(screen, corner="top-left")
        return f"{rolls:.0f} rolls"


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, caption: str) -> None:
    """Draw the set's caption and the keys at the top, in the middle."""
    text = font.render(f"{caption}   (Space: next set, P: pause)", True, (40, 40, 40))
    panel = pygame.Surface(
        (text.get_width() + 20, text.get_height() + 12), pygame.SRCALPHA
    )
    panel.fill((255, 255, 255, 225))
    panel.blit(text, (10, 6))
    screen.blit(panel, ((screen.get_width() - panel.get_width()) // 2, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: data plots")
    font = pygame.font.Font(bundled_font_path(), 14)
    renderer = PygameRenderer()  # the planes share it, and its caches
    sets: list[Demo] = [
        Pendulum(renderer),
        Particles(renderer),
        Means(renderer),
        Dice(renderer),
        Groups(renderer),
        Energy(renderer),
    ]
    clock = pygame.time.Clock()
    current = 0
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(sets)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            sets[current].grid.handle_event(event)
        caption = sets[current].draw(screen, not paused)
        draw_caption(screen, font, caption)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
