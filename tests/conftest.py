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

import pytest

from pygridplot.core.primitives import FontSpec, LabelText, TextSize

CHAR_WIDTH = 7
SUPERSCRIPT_CHAR_WIDTH = 5
DIGIT_HEIGHT = 9
SUPERSCRIPT_OVERFLOW = 3


class FakeMeasurer:
    """A deterministic text measurer for testing the core without fonts.

    Every normal character is 7 px wide, superscript characters 5 px, and the
    digit box is 9 px tall with 3 px of superscript overflow.
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
        return TextSize(width, DIGIT_HEIGHT, sign, overflow)


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
