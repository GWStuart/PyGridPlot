"""A plane inside a game: a bouncing ball, and a live graph of its height.

The left of the window is ordinary pygame drawing. The right is a
``Grid`` given a rectangle of the window, plotting the ball's height as
it falls and bounces; ``set_view`` keeps the last ten seconds in view.
"""

import pygame

import pygridplot as pgp

GRAVITY = 9.81  # m/s²
BOUNCE = 0.85  # the speed kept at each bounce
PIXELS_PER_METRE = 52
BALL = (235, 95, 70)  # the ball's colour, in the game and on the graph


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 640))
    pygame.display.set_caption("PyGridPlot: a plane inside a game")
    clock = pygame.time.Clock()
    graph = pgp.Grid(
        pygame.Rect(400, 0, 700, 640),
        x_range=(0, 10),
        y_range=(-0.5, 11),
        interactive=False,  # the view follows the data instead
    )
    t, height, speed = 0.0, 10.0, 0.0
    history: list[tuple[float, float]] = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            graph.handle_event(event)

        # The simulation: one step of a sixtieth of a second.
        dt = 1 / 60
        t += dt
        speed -= GRAVITY * dt
        height += speed * dt
        if height < 0:  # a bounce
            height, speed = -height, -BOUNCE * speed
            if speed < 1:  # it has nearly stopped: drop it again
                height, speed = 10.0, 0.0
        history.append((t, height))
        history = history[-700:]

        # The game: plain pygame.
        ground = 600
        screen.fill((32, 34, 44), pygame.Rect(0, 0, 400, 640))
        pygame.draw.line(screen, (200, 200, 210), (30, ground), (370, ground), 3)
        ball_y = ground - 16 - height * PIXELS_PER_METRE
        pygame.draw.circle(screen, BALL, (200, round(ball_y)), 16)

        # The graph.
        graph.set_view(x_range=(max(0, t - 10), max(10, t)), y_range=(-0.5, 11))
        graph.draw(screen, x_title="time (s)", y_title="height (m)")
        graph.polyline(
            screen, history, color=BALL, width=2.5, label="height of the ball"
        )
        graph.points(screen, history[-1:], color=BALL, radius=5)
        graph.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
