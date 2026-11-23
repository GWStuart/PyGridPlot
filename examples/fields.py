"""Vector fields: the electric field of two charges.

``vector_field(F)`` draws an arrow in the direction of ``F(x, y)`` at the
points of a lattice that stays put while you pan and keeps its density as
you zoom. Drag the two charges: the field, and the equipotential lines
drawn with ``implicit``, follow them.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: vector fields")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-8, 8))
    plus = grid.draggable((-2.5, 0.0), color="red")
    minus = grid.draggable((2.5, 0.0), color="blue")
    charges = [(plus, 1.0), (minus, -1.0)]

    def field(x: float, y: float) -> tuple[float, float]:
        """The electric field at (x, y): the sum of each charge's."""
        ex, ey = 0.0, 0.0
        for point, charge in charges:
            dx, dy = x - point.x, y - point.y
            r3 = (dx**2 + dy**2) ** 1.5
            ex, ey = ex + charge * dx / r3, ey + charge * dy / r3
        return ex, ey

    def potential(x: float, y: float) -> float:
        """The electric potential at (x, y)."""
        return sum(charge / np.hypot(x - p.x, y - p.y) for p, charge in charges)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        grid.draw(screen)
        grid.vector_field(screen, field, color="steelblue", label="electric field")
        grid.implicit(
            screen,
            potential,
            [-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1],
            color="gray",
            width=1.5,
            label="equipotential lines",
        )
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
