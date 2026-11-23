"""Histograms: filling up as samples arrive.

Each frame adds 40 samples. ``histogram(samples, density=True)`` counts
them in bins anchored at multiples of the bin width, so the bins never
shift as samples arrive, and it picks the width itself: narrower as the
samples pile up. The curve is the density they are drawn from.
"""

import numpy as np
import pygame

import pygridplot as pgp


def normal(x: float, mean: float, sd: float) -> float:
    """The density of a normal distribution."""
    return np.exp(-(((x - mean) / sd) ** 2) / 2) / (sd * np.sqrt(2 * np.pi))


def density(x: float) -> float:
    """The density of the samples: two overlapping bell curves."""
    return 0.6 * normal(x, -1, 0.6) + 0.4 * normal(x, 1.5, 0.8)


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: histogram")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-4, 5), y_range=(-0.04, 0.5))
    rng = np.random.default_rng(1)
    samples = np.empty(0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        if samples.size < 50_000:
            left = rng.random(40) < 0.6
            new = np.where(left, rng.normal(-1, 0.6, 40), rng.normal(1.5, 0.8, 40))
            samples = np.concatenate([samples, new])

        grid.draw(screen, x_title="x", y_title="density")
        grid.histogram(screen, samples, density=True, label=f"{samples.size} samples")
        grid.plot(screen, density, color="black", label="their density")
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
