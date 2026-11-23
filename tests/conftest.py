"""Shared pytest configuration and fixtures.

SDL is switched to its dummy drivers *before* pygame is imported anywhere,
so the whole suite runs headless (on CI, over SSH, ...).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from pygridplot.core.primitives import FontSpec, LabelText, TextItem, TextSize

if TYPE_CHECKING:
    from pygridplot.core.grid.scene import GridScene

CHAR_WIDTH = 7
SUPERSCRIPT_CHAR_WIDTH = 5
DIGIT_HEIGHT = 9
SUPERSCRIPT_OVERFLOW = 3
DESCENT = 3


class FakeMeasurer:
    """A deterministic text measurer for testing the core without fonts.

    Every normal character is 7 px wide, superscript characters 5 px, and the
    digit box is 9 px tall with 3 px of superscript overflow, and letters
    reach 3 px below the baseline.
    """

    def measure_text(self, text: LabelText, font: FontSpec) -> TextSize:
        """Measure ``text`` with fixed per-character widths."""
        width = 0
        sign = 0
        overflow = 0
        for index, run in enumerate(text.runs):
            if run.role == "superscript":
                width += SUPERSCRIPT_CHAR_WIDTH * len(run.text)
                overflow = SUPERSCRIPT_OVERFLOW
            else:
                run_width = CHAR_WIDTH * len(run.text)
                if index == 0 and run.role == "sign":
                    sign = run_width
                width += run_width
        return TextSize(width, DIGIT_HEIGHT, sign, overflow, DESCENT)


def labels_by_axis(scene: GridScene) -> list[tuple[tuple[str, str], TextItem]]:
    """Return every label of a grid with its key: its axis, and its text.

    Tests that follow labels from frame to frame match them by this key.
    The text alone is not enough: both axes can show the same text (a view
    centred at ``(c, c)``), and the two labels then move independently.

    Args:
        scene: The grid.

    Returns:
        ``((axis, text), label)`` for every label, where the axis is
        ``"x"``, ``"zero"`` (the origin's ``0``), ``"y"``, ``"x_title"`` or
        ``"y_title"``.
    """
    groups = (
        ("x", scene.axis_labels.x),
        ("zero", scene.axis_labels.zero),
        ("y", scene.axis_labels.y),
        ("x_title", scene.axis_labels.x_title),
        ("y_title", scene.axis_labels.y_title),
    )
    return [((axis, item.text.plain), item) for axis, items in groups for item in items]


@pytest.fixture
def measurer() -> FakeMeasurer:
    """A deterministic :class:`FakeMeasurer`."""
    return FakeMeasurer()


@pytest.fixture
def pygame_display() -> Iterator[object]:
    """Initialise pygame with a small dummy display for rendering tests."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((320, 240))
    yield screen
    pygame.quit()
