"""The built-in dark theme, plus a customised theme derived from it.

Themes are frozen dataclasses: derive variations with
:func:`dataclasses.replace`. Press ``Space`` to switch between the plain
dark theme and the customised one.
"""

from __future__ import annotations

from dataclasses import replace

import pygame

import pygridplot as pgp

CUSTOM = replace(
    pgp.DARK,
    background=(14, 20, 34),
    minor_color=(26, 36, 56),
    major_color=(44, 62, 94),
    axis_color=(120, 190, 255),
    label_color=(170, 205, 245),
    axis_width=2,
    target_major_px=120,
)
"""A blueprint-style variation of the dark theme."""


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: dark theme (Space to switch)")
    grid = pgp.Grid(theme=pgp.DARK)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                grid.theme = CUSTOM if grid.theme is pgp.DARK else pgp.DARK
            grid.handle_event(event)
        grid.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
