"""An interactive playground for tuning the grid, with a debug overlay.

Pan and zoom as usual, and use these keys to change settings live:

======  ==============================================
Key     Effect
======  ==============================================
T       toggle light / dark theme
F       toggle spacing transitions: fade / snap
R       toggle spacing rule: nearest / ceil
N       toggle minor lines
J       toggle major lines
A       toggle axes
L       toggle labels
H       toggle label halos
S       toggle eased (smooth) zooming
[ / ]   decrease / increase the target major spacing
F1      show or hide this overlay
Esc     quit
======  ==============================================

The overlay shows the current scale, centre, chosen spacings (including
cross-fade weights), frame time and the time spent drawing the grid.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pygame

import pygridplot as pgp
from pygridplot.core.spacing import SpacingChoice
from pygridplot.render.fonts import bundled_font_path

HELP = (
    "T theme  F fade/snap  R rule  N minor  J major  A axes  L labels  "
    "H halo  S smooth  [ ] spacing  F1 overlay"
)


def describe_spacing(choice: SpacingChoice) -> str:
    """Describe a spacing choice for the overlay.

    Args:
        choice: The spacing chosen for one axis.

    Returns:
        For example ``"2"`` or ``"2 -> 5 (37%)"`` during a cross-fade.
    """
    if choice.coarse is None:
        return str(choice.fine)
    return f"{choice.fine} -> {choice.coarse} ({choice.blend:.0%})"


def toggle_theme_setting(plane: pgp.Plane, key: int) -> bool:
    """Apply a theme-changing hotkey to the plane.

    Args:
        plane: The plane to restyle.
        key: The pressed key.

    Returns:
        ``True`` if the key changed the theme.
    """
    theme = plane.theme
    flags = {
        pygame.K_n: "show_minor",
        pygame.K_j: "show_major",
        pygame.K_a: "show_axes",
        pygame.K_l: "show_labels",
    }
    if key in flags:
        name = flags[key]
        plane.theme = replace(theme, **{name: not getattr(theme, name)})
    elif key == pygame.K_t:
        base = pgp.LIGHT if theme.background == pgp.DARK.background else pgp.DARK
        plane.theme = replace(
            base,
            transition=theme.transition,
            spacing_rule=theme.spacing_rule,
            target_major_px=theme.target_major_px,
        )
    elif key == pygame.K_f:
        mode = "snap" if theme.transition == "fade" else "fade"
        plane.theme = replace(theme, transition=mode)
    elif key == pygame.K_r:
        rule = "ceil" if theme.spacing_rule == "nearest" else "nearest"
        plane.theme = replace(theme, spacing_rule=rule)
    elif key == pygame.K_h:
        plane.theme = replace(
            theme, label_halo_width=0 if theme.label_halo_width else 2
        )
    elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
        factor = 1.1 if key == pygame.K_RIGHTBRACKET else 1 / 1.1
        target = min(max(theme.target_major_px * factor, 30.0), 400.0)
        plane.theme = replace(theme, target_major_px=target)
    else:
        return False
    return True


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    plane: pgp.Plane,
    frame_ms: float,
    grid_ms: float,
) -> None:
    """Draw the debug overlay in the top-left corner.

    Args:
        screen: The display surface.
        font: Font for the overlay text.
        plane: The plane being inspected.
        frame_ms: Smoothed total frame time in milliseconds.
        grid_ms: Smoothed time spent in ``draw_grid`` in milliseconds.
    """
    view = plane.view
    grid = plane.last_grid
    theme = plane.theme
    lines = [
        f"scale  {view.scale_x:.6g} px/unit",
        f"centre ({view.center_x:.10g}, {view.center_y:.10g})",
    ]
    if grid is not None:
        lines.append(f"x step {describe_spacing(grid.x_spacing)}")
        lines.append(f"y step {describe_spacing(grid.y_spacing)}")
    lines += [
        f"mode   {theme.transition} / {theme.spacing_rule}"
        f" / target {theme.target_major_px:.0f}px"
        f" / smooth {'on' if plane.interaction.smoothing else 'off'}",
        f"frame  {frame_ms:5.2f} ms   grid {grid_ms:5.2f} ms",
        HELP,
    ]
    rendered = [font.render(text, True, (240, 240, 240)) for text in lines]
    width = max(r.get_width() for r in rendered) + 16
    height = sum(r.get_height() for r in rendered) + 12
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((20, 22, 28, 200))
    y = 6
    for r in rendered:
        panel.blit(r, (8, y))
        y += r.get_height()
    screen.blit(panel, (10, 10))


def main() -> None:
    """Run the playground."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: grid playground")
    plane = pgp.Plane()
    font = pygame.font.Font(bundled_font_path(), 13)
    clock = pygame.time.Clock()
    show_overlay = True
    frame_ms = grid_ms = 0.0
    last = time.perf_counter()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    show_overlay = not show_overlay
                elif event.key == pygame.K_s:
                    smoothing = 0.0 if plane.interaction.smoothing else 0.06
                    plane.interaction = replace(plane.interaction, smoothing=smoothing)
                elif toggle_theme_setting(plane, event.key):
                    pass
            plane.handle_event(event)

        start = time.perf_counter()
        plane.draw_grid(screen)
        grid_ms = 0.9 * grid_ms + 0.1 * (time.perf_counter() - start) * 1000
        if show_overlay:
            draw_overlay(screen, font, plane, frame_ms, grid_ms)
        pygame.display.flip()
        clock.tick(120)
        now = time.perf_counter()
        frame_ms = 0.9 * frame_ms + 0.1 * (now - last) * 1000
        last = now
    pygame.quit()


if __name__ == "__main__":
    main()
