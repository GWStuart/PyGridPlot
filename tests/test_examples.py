"""Smoke tests: every example runs headless for a few frames.

Each example's ``main`` loop is driven by a fake event queue that presses a
few of the keys the examples use (switching sets, toggling log axes, cycling
plots, zooming) and then closes the window.
"""

from __future__ import annotations

import runpy
from collections.abc import Iterator
from pathlib import Path

import pygame
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(
    path for path in (ROOT / "examples").glob("*.py") if path.stem != "zoom_sweep"
)


def scripted_events() -> Iterator[list[pygame.event.Event]]:
    """Yield the events of each frame: some key presses, a zoom, then quit."""

    def key(code: int) -> pygame.event.Event:
        return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0, unicode="")

    yield []
    yield [key(pygame.K_SPACE), key(pygame.K_x)]
    yield [
        pygame.event.Event(
            pygame.MOUSEMOTION, pos=(300, 200), rel=(0, 0), buttons=(0, 0, 0)
        ),
        pygame.event.Event(
            pygame.MOUSEWHEEL, x=0, y=1, precise_x=0.0, precise_y=1.0, flipped=False
        ),
    ]
    yield [key(pygame.K_y), key(pygame.K_p)]
    yield []
    while True:
        yield [pygame.event.Event(pygame.QUIT)]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs_a_few_frames(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = scripted_events()
    monkeypatch.setattr(pygame.event, "get", lambda *args, **kwargs: next(frames))
    namespace = runpy.run_path(str(path), run_name="example")
    namespace["main"]()
