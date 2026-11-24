"""Recording: press S to save a picture, and R to start and stop a video.

A Lissajous figure turns as its phase plays. S saves the plane as a PNG
at twice the size (``grid.save``); R starts recording a video of it, and R
again finishes it (``pgp.Recorder``, which needs ffmpeg: see the user
guide's page on saving pictures and videos). The files go in the folder the
program runs from.
"""

import numpy as np
import pygame

import pygridplot as pgp


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: S saves a picture, R records")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-3.6, 3.6))
    phase = grid.slider("phase", (0, 2 * np.pi), play="loop", duration=6)
    phase.playing = True
    video: pgp.Recorder | None = None
    saves = 0

    running = True
    while running:
        save = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                save = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if video is None:
                    video = pgp.Recorder("lissajous.mp4", fps=60)
                else:
                    video.close()
                    video = None
            grid.handle_event(event)

        p = phase.value
        grid.draw(screen)
        grid.parametric(
            screen,
            lambda t: 2 * np.sin(3 * t + p),
            lambda t: 2 * np.sin(2 * t),
            (0, 2 * np.pi),
        )
        grid.draw_controls(screen)
        if save:
            saves += 1
            grid.save(f"lissajous_{saves}.png", scale=2)
        if video is not None:
            video.add(grid)
        pygame.display.flip()
        clock.tick(60)
    if video is not None:
        video.close()
    pygame.quit()


if __name__ == "__main__":
    main()
