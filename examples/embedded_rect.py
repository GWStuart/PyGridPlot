"""A plane embedded in part of a window, next to other content.

The plane only reacts to mouse input inside its own rectangle, and all of
its drawing is clipped to it. The rest of the window (here a simple sidebar
showing the world coordinates under the cursor) belongs to the program.
"""

from __future__ import annotations

import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

SIDEBAR_WIDTH = 260
MARGIN = 16


def grid_rect(screen: pygame.Surface) -> pygame.Rect:
    """Return the rectangle the plane occupies for the current window size.

    Args:
        screen: The display surface.

    Returns:
        Everything right of the sidebar, inset by a margin.
    """
    width, height = screen.get_size()
    return pygame.Rect(
        SIDEBAR_WIDTH + MARGIN,
        MARGIN,
        max(0, width - SIDEBAR_WIDTH - 2 * MARGIN),
        max(0, height - 2 * MARGIN),
    )


def draw_sidebar(
    screen: pygame.Surface, font: pygame.font.Font, grid: pgp.Grid
) -> None:
    """Draw the sidebar with the cursor's world coordinates.

    Args:
        screen: The display surface.
        font: Font for the sidebar text.
        grid: The embedded plane.
    """
    mx, my = pygame.mouse.get_pos()
    lines = ["Embedded plane", ""]
    if grid.rect.contains_point(mx, my):
        x, y = grid.to_world(mx, my)
        lines += [f"x = {x:.6g}", f"y = {y:.6g}"]
    else:
        lines += ["(move the mouse", " over the plane)"]
    y_pos = MARGIN
    for text in lines:
        screen.blit(font.render(text, True, (225, 228, 235)), (MARGIN, y_pos))
        y_pos += font.get_linesize()


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 680), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: embedded plane")
    font = pygame.font.Font(bundled_font_path(), 16)
    grid = pgp.Grid(grid_rect(screen))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWSIZECHANGED:
                grid.rect = grid_rect(screen)
            grid.handle_event(event)

        screen.fill((48, 52, 62))
        draw_sidebar(screen, font, grid)
        grid.draw(screen)
        r = grid.rect
        border = pygame.Rect(r.x - 1, r.y - 1, r.width + 2, r.height + 2)
        pygame.draw.rect(screen, (24, 26, 31), border, 1)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
