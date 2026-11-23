"""Calculus: a tangent line and the area under a curve.

Drag the point along the curve. The tangent comes from the exact
derivative, ``pgp.derivative(f)``, and the shaded area and its value from
``pgp.integral(f, 0, x)``. The derivative is drawn too.
"""

from dataclasses import replace

import pygame

import pygridplot as pgp


def f(x: float) -> float:
    """The function: a cubic with a hump and a dip."""
    return x**3 / 6 - x


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: calculus")
    clock = pygame.time.Clock()
    theme = replace(pgp.LIGHT, show_arrows=True)  # arrowheads on the axes
    grid = pgp.Grid(x_range=(-4.5, 4.5), theme=theme)
    df = pgp.derivative(f)
    point = grid.draggable(curve=(lambda t: t, f, (-4, 4)), t=2.2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        x0, y0 = point.x, point.y
        slope = df(x0)
        area = pgp.integral(f, 0, x0)

        grid.draw(screen, x_title="x", y_title="y")
        grid.fill_between(
            screen,
            f,
            0,
            domain=(min(0, x0), max(0, x0)),
            label=f"∫ f dx from 0 to {x0:.2f} = {area:.3f}",
        )
        grid.plot(screen, f, label="f(x) = x³/6 − x")
        grid.plot(screen, df, width=1.5, label="f′(x)")
        grid.plot(
            screen,
            lambda x: y0 + slope * (x - x0),
            width=1.5,
            label=f"tangent, slope {slope:.2f}",
        )
        grid.draw_controls(screen)
        grid.legend(screen, corner="top-left")
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
