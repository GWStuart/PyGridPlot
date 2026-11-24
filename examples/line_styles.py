"""Line styles: dashed asymptotes and a dashed tangent, with dotted guides.

Drag the point along the curve. Its tangent is dashed, and dotted lines
drop from it to the axes; the curve's two asymptotes are dashed too. Pan
the plane: the dashes are fixed to it, so they move with the lines.
"""

import pygame

import pygridplot as pgp


def f(x: float) -> float:
    """A rational function: a pole at x = 1, and y = x + 1 far away."""
    return (x**2 + 1) / (x - 1)


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: line styles")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-6, 8), center=(1.0, 2.0))
    df = pgp.derivative(f)
    point = grid.draggable(curve=(lambda t: t, f, (-6, 8)), t=3.0)
    guide = pgp.LineStyle(width=1.5, color="gray")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        x0, y0 = point.x, point.y
        slope = df(x0)
        grid.draw(screen)
        grid.polyline(
            screen,
            [(1, -1e6), (1, 1e6)],
            style=guide,
            dash="dashed",
            label="asymptotes",
        )
        grid.plot(screen, lambda x: x + 1, style=guide, dash="dashed")
        grid.plot(screen, f, label="f(x) = (x² + 1) / (x − 1)")
        grid.plot(
            screen,
            lambda x: y0 + slope * (x - x0),
            dash="dashed",
            label=f"tangent, slope {slope:.2f}",
        )
        grid.polyline(screen, [(x0, 0), (x0, y0), (0, y0)], color="gray", dash="dotted")
        grid.draw_controls(screen)
        grid.legend(screen, corner="top-left")
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
