"""Render the examples' pictures, the README's montage and the examples page.

Each example's ``main()`` runs headless with a scripted event queue and a
clock driven by the frame count (frame ``k`` is at ``k / 60`` seconds), so
the pictures are the same on every run and nothing waits for real time.
The window is saved at a chosen frame (later for examples that build up
over time), and the example is then closed.

Usage::

    python scripts/render_examples.py          # everything
    python scripts/render_examples.py fields   # these pictures only
    python scripts/render_examples.py --page   # the page only (no rendering)

Pictures go to ``docs/images/examples/<name>.png`` and the montage to
``docs/images/montage.png``. The page, ``docs/user/examples.md``, is built
from :data:`SECTIONS` and the examples' docstrings (whose first line is
``Title: summary.``); ``tests/test_docs.py`` fails if it is out of date or
an example is missing from it.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import runpy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "docs" / "images" / "examples"
MONTAGE_OUT = ROOT / "docs" / "images" / "montage.png"
PAGE = ROOT / "docs" / "user" / "examples.md"

FRAME_MS = 1000 / 60
"""The time between frames, as the examples' clocks tick at 60 frames a second."""

FRAMES: dict[str, int] = {
    "animation": 200,
    "controls": 150,
    "recording": 60,
    "curves": 50,
    "embedded": 420,
    "histogram": 360,
}
"""The frame to save, for examples that change over time (default 3)."""

MONTAGE = [
    "calculus",
    "fields",
    "scatter",
    "curves",
    "geometry",
    "distributions",
]
"""The examples in the README's montage, in reading order (three a row)."""

SECTIONS: list[tuple[str, list[str]]] = [
    ("Getting started", ["quickstart", "quick_look"]),
    (
        "Functions and calculus",
        ["functions", "calculus", "line_styles", "taylor", "fourier"],
    ),
    ("Curves and regions", ["curves", "implicit", "regions"]),
    ("Fields", ["fields", "slope_field"]),
    ("Points, sliders and shapes", ["sliders", "controls", "geometry", "sequences"]),
    ("Data", ["scatter", "histogram", "bars", "distributions"]),
    ("Axes", ["log_axes"]),
    ("Animation, games and recording", ["animation", "embedded", "recording"]),
]
"""The examples page's sections, and the order of the examples in them."""

TILE = (400, 266)
GAP = 6


def scripted_events(frames: int) -> Iterator[list[pygame.event.Event]]:
    """Yield no events for ``frames`` frames, then a request to quit."""
    for _ in range(frames):
        yield []
    while True:
        yield [pygame.event.Event(pygame.QUIT)]


class FakeClock:
    """A ``pygame.time.Clock`` that never sleeps."""

    def tick(self, framerate: float = 0) -> int:
        """Return the time of one frame, without waiting."""
        return round(FRAME_MS)

    def get_fps(self) -> float:
        """Return the nominal frame rate."""
        return 60.0


def render(name: str) -> pygame.Surface:
    """Run one example and return its window at the chosen frame.

    Args:
        name: The example's file name without ``.py``.

    Returns:
        A copy of the window.
    """
    wanted = FRAMES.get(name, 3)
    events = scripted_events(wanted + 1)
    frame = 0
    shot: list[pygame.Surface] = []
    flip = pygame.display.flip

    def next_events(*args: Any, **kwargs: Any) -> list[pygame.event.Event]:
        return next(events)

    def ticks() -> int:
        return round(frame * FRAME_MS)

    def counting_flip() -> None:
        nonlocal frame
        frame += 1
        if frame == wanted:
            surface = pygame.display.get_surface()
            if surface is not None:
                shot.append(surface.copy())
        flip()

    saved = (pygame.event.get, pygame.time.Clock, pygame.time.get_ticks, flip)
    pygame.event.get = next_events  # type: ignore[assignment]
    pygame.time.Clock = FakeClock  # type: ignore[assignment, misc]
    pygame.time.get_ticks = ticks  # type: ignore[assignment]
    pygame.display.flip = counting_flip  # type: ignore[assignment]
    try:
        namespace = runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="example")
        namespace["main"]()
    finally:
        (
            pygame.event.get,
            pygame.time.Clock,
            pygame.time.get_ticks,
            pygame.display.flip,
        ) = saved  # type: ignore[assignment]
    if not shot:
        msg = f"{name} ended before frame {wanted}"
        raise RuntimeError(msg)
    return shot[0]


def montage(pictures: dict[str, pygame.Surface]) -> pygame.Surface:
    """Return the README's montage: the pictures of :data:`MONTAGE`, tiled.

    Args:
        pictures: Each example's picture, by name.

    Returns:
        Three tiles a row, separated by :data:`GAP` white pixels.
    """
    columns = 3
    rows = -(-len(MONTAGE) // columns)
    width, height = TILE
    sheet = pygame.Surface(
        (columns * width + (columns - 1) * GAP, rows * height + (rows - 1) * GAP)
    )
    sheet.fill((255, 255, 255))
    for k, name in enumerate(MONTAGE):
        tile = pygame.transform.smoothscale(pictures[name], TILE)
        sheet.blit(
            tile, ((k % columns) * (width + GAP), (k // columns) * (height + GAP))
        )
    return sheet


def describe(name: str) -> tuple[str, str, str]:
    """Return an example's title, summary and description, from its docstring.

    Args:
        name: The example's file name without ``.py``.

    Returns:
        The first line's title (before the colon) and summary (after it,
        capitalised), and the rest of the docstring with ``code`` marked as
        in Markdown.
    """
    source = (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    first, _, rest = docstring.partition("\n")
    title, _, summary = first.partition(": ")
    summary = summary[:1].upper() + summary[1:]
    body = re.sub(r"``(.+?)``", r"`\1`", rest.strip(), flags=re.S)
    return title, summary, body


def examples_page() -> str:
    """Return the examples page, ``docs/user/examples.md``."""
    lines = [
        "# Examples",
        "",
        "Every program in the repository's `examples/` folder, with what it shows.",
        "Each is one short file using only `import pygridplot as pgp`, written",
        "the way your own program would be: copy one to start from. To run them,",
        "clone or download the repository, `pip install pygridplot`, and run",
        "`python examples/calculus.py`, say. In every one you can drag to pan,",
        "scroll to zoom and press Home to go back to the start.",
        "",
        "<!-- Generated by scripts/render_examples.py from the examples'",
        "     docstrings and pictures: edit those, then run it again. -->",
        "",
    ]
    for section, names in SECTIONS:
        lines += [f"## {section}", ""]
        for name in names:
            title, summary, body = describe(name)
            alt = f"{title}: {summary[:1].lower()}{summary[1:]}".rstrip(".")
            lines += [
                f"### {title}",
                "",
                f"{summary}",
                "",
                f"![{alt}](../images/examples/{name}.png)",
                "",
                body,
                "",
                f"`python examples/{name}.py` ([source](../../examples/{name}.py))",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def listed() -> list[str]:
    """Return every example named in :data:`SECTIONS`, in order."""
    return [name for _, names in SECTIONS for name in names]


def main() -> None:
    """Render the pictures named on the command line (or all), and the page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="pictures to render (default: all)")
    parser.add_argument("--page", action="store_true", help="only write the page")
    args = parser.parse_args()
    missing = sorted({path.stem for path in EXAMPLES.glob("*.py")} - set(listed()))
    if missing:
        parser.error(f"add these examples to SECTIONS: {', '.join(missing)}")
    if not args.page:
        names = args.names or listed()
        OUT.mkdir(parents=True, exist_ok=True)
        pictures = {}
        for name in names:
            pictures[name] = render(name)
            pygame.image.save(pictures[name], OUT / f"{name}.png")
            print(f"rendered {name}")
        if set(MONTAGE) <= set(pictures):
            pygame.image.save(montage(pictures), MONTAGE_OUT)
            print(f"wrote {MONTAGE_OUT.relative_to(ROOT)}")
    PAGE.write_text(examples_page(), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
