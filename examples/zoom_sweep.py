"""A visual stress test: sweep the zoom through many decades and pan far away.

Watch it for flicker, popping, uneven spacing or jittering labels. The sweep
zooms about a point that is *not* the centre of the window, so it also tests
zooming about the cursor.

Interactive use::

    python examples/zoom_sweep.py            # Space pauses, Esc quits

Headless frame dumps (for comparing frames one by one)::

    python examples/zoom_sweep.py --save-frames frames/ --first 0 --count 120

Keys while running: ``Space`` pause/resume, ``T`` light/dark, ``F`` fade/snap.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pygame

import pygridplot as pgp


@dataclass(frozen=True)
class Segment:
    """One part of the sweep.

    Attributes:
        kind: ``"zoom"`` or ``"pan"``.
        amount: Decades of zoom (positive zooms in) or a pan velocity in
            pixels per second ``(dx, dy)``.
        seconds: Duration of the segment.
    """

    kind: str
    amount: float | tuple[float, float]
    seconds: float


SWEEP: tuple[Segment, ...] = (
    Segment("zoom", -6.0, 15.0),
    Segment("zoom", 14.0, 35.0),
    Segment("pan", (700.0, 300.0), 6.0),
    Segment("zoom", -8.0, 20.0),
    Segment("pan", (-900.0, -500.0), 6.0),
)


def frame_actions(fps: int) -> Iterator[tuple[str, object]]:
    """Yield one ``(kind, amount)`` action per frame of the sweep, forever.

    Args:
        fps: Frames per second the sweep is timed for.

    Yields:
        ``("zoom", factor)`` or ``("pan", (dx, dy))`` for each frame.
    """
    while True:
        for segment in SWEEP:
            frames = max(1, round(segment.seconds * fps))
            for _ in range(frames):
                if segment.kind == "zoom":
                    assert isinstance(segment.amount, float)
                    yield ("zoom", 10 ** (segment.amount / frames))
                else:
                    assert isinstance(segment.amount, tuple)
                    dx, dy = segment.amount
                    yield ("pan", (dx / fps, dy / fps))


def apply_action(plane: pgp.Plane, action: tuple[str, object]) -> None:
    """Apply one frame of the sweep to the plane's view.

    Args:
        plane: The plane being swept.
        action: The action from :func:`frame_actions`.
    """
    kind, amount = action
    view = plane.view
    if kind == "zoom":
        rect = view.rect
        anchor = (rect.x + rect.width * 0.62, rect.y + rect.height * 0.38)
        view.zoom_at(*anchor, float(amount))  # type: ignore[arg-type]
    else:
        dx, dy = amount  # type: ignore[misc]
        view.pan_pixels(dx, dy)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv: Arguments without the program name.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Zoom/pan sweep stress test.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--save-frames", type=Path, help="write frames here, headless")
    parser.add_argument("--first", type=int, default=0, help="first frame to save")
    parser.add_argument("--count", type=int, default=120, help="frames to save")
    parser.add_argument("--every", type=int, default=1, help="save every Nth frame")
    parser.add_argument("--size", default="900x600")
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--snap", action="store_true", help="snap instead of fade")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the sweep, interactively or as a headless frame dump.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    headless = args.save_frames is not None
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        args.save_frames.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.size.lower().split("x"))
    pygame.init()
    screen = pygame.display.set_mode(
        (width, height), 0 if headless else pygame.RESIZABLE
    )
    pygame.display.set_caption("PyGridPlot: zoom sweep")
    theme = pgp.DARK if args.dark else pgp.LIGHT
    if args.snap:
        theme = replace(theme, transition="snap")
    plane = pgp.Plane(theme=theme, interaction=pgp.InteractionConfig(smoothing=0.0))
    clock = pygame.time.Clock()
    actions = frame_actions(args.fps)
    paused = False
    frame = 0
    last_frame = args.first + args.count * args.every

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    dark = plane.theme.background == pgp.DARK.background
                    plane.theme = pgp.LIGHT if dark else pgp.DARK
                elif event.key == pygame.K_f:
                    mode = "snap" if plane.theme.transition == "fade" else "fade"
                    plane.theme = replace(plane.theme, transition=mode)
            plane.handle_event(event)
        if not paused:
            apply_action(plane, next(actions))
        plane.draw_grid(screen)
        if headless:
            offset = frame - args.first
            if offset >= 0 and offset % args.every == 0:
                pygame.image.save(screen, args.save_frames / f"frame_{frame:05d}.png")
            running = running and frame + 1 < last_frame
        else:
            pygame.display.flip()
            clock.tick(args.fps)
        frame += 1
    pygame.quit()
    if headless:
        print(f"saved {args.count} frames to {args.save_frames}")


if __name__ == "__main__":
    main()
