"""Typesetting tick labels for the pygame backend.

A :class:`~pygridplot.core.primitives.LabelText` is typeset run by run. Normal
runs sit on a shared baseline, and superscript runs are drawn at a smaller
size and raised. Every measurement is relative to the label's **digit box**
(top of the digits to the baseline), which is what layout aligns. This means
labels line up exactly with their grid lines whatever characters they
contain.

Finished labels are cached as single surfaces, halo included, so drawing
one costs a single blit per frame. Label text depends only on the label's
value, so cache hits are the norm while panning and zooming.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Final

import pygame

from pygridplot.core.coords import ColorLike
from pygridplot.core.grid.formatting import MINUS_SIGN, TIMES_SIGN
from pygridplot.core.primitives import FontSpec, LabelText, TextSize
from pygridplot.render.colors import ColorResolver
from pygridplot.render.fonts import FontLibrary

SUPERSCRIPT_SCALE: Final = 0.72
"""Superscript font size relative to the label font size."""

SUPERSCRIPT_RAISE: Final = 0.5
"""How far a superscript's baseline is raised, as a fraction of digit height."""

_FALLBACK_GLYPHS: Final = {MINUS_SIGN: "-", TIMES_SIGN: "x"}
"""ASCII stand-ins for characters a font might be missing."""


@dataclass(frozen=True, slots=True)
class _RunLayout:
    """Where one run of a label goes, relative to the digit box's top-left.

    Attributes:
        text: The characters to draw (after glyph fallbacks).
        size: Font size of the run in points.
        x: Left edge of the run.
        baseline: Baseline of the run (the digit box's baseline is at
            ``TextSize.height``).
    """

    text: str
    size: int
    x: int
    baseline: int


@dataclass(frozen=True, slots=True)
class _LabelLayout:
    """The typeset form of a label: positioned runs plus the overall size.

    Attributes:
        runs: Positioned runs in drawing order.
        size: Measured size of the whole label.
    """

    runs: tuple[_RunLayout, ...]
    size: TextSize


@dataclass(frozen=True, slots=True)
class RenderedText:
    """A label rendered to a surface, ready to blit.

    Attributes:
        surface: The rendered label, halo included, with per-pixel alpha.
        offset: Position of the digit box's top-left corner within ``surface``.
            Blit the surface at ``(left - offset[0], top - offset[1])`` to put
            the digit box at ``(left, top)``.
    """

    surface: pygame.Surface
    offset: tuple[int, int]


class TextRenderer:
    """Measures and renders labels, with bounded caches for both.

    Args:
        fonts: Where fonts are loaded from.
        colors: Resolves colour values.
        cache_size: Maximum number of rendered labels kept in memory.
    """

    def __init__(
        self, fonts: FontLibrary, colors: ColorResolver, cache_size: int = 2048
    ) -> None:
        self._fonts = fonts
        self._colors = colors
        self._cache_size = max(16, cache_size)
        self._layouts: OrderedDict[tuple[LabelText, FontSpec], _LabelLayout] = (
            OrderedDict()
        )
        self._rendered: OrderedDict[tuple[object, ...], RenderedText] = OrderedDict()

    def measure(self, text: LabelText, font: FontSpec) -> TextSize:
        """Return the size of a label without rendering it.

        Args:
            text: The label.
            font: The font it will be drawn with.

        Returns:
            The label's digit-box size.
        """
        return self._layout(text, font).size

    def render(
        self,
        text: LabelText,
        font: FontSpec,
        color: ColorLike,
        halo_color: ColorLike | None = None,
        halo_width: int = 0,
    ) -> RenderedText:
        """Render a label (cached).

        Args:
            text: The label.
            font: The font to draw with.
            color: Text colour.
            halo_color: Colour of the outline around the glyphs, or ``None``.
            halo_width: Thickness of the outline in pixels.

        Returns:
            The rendered label.
        """
        rgb = self._colors.rgba(color)
        halo_rgb = None
        if halo_color is not None and halo_width > 0:
            halo_rgb = self._colors.rgba(halo_color)
        else:
            halo_width = 0
        key = (text, font, rgb, halo_rgb, halo_width)
        cached = self._rendered.get(key)
        if cached is not None:
            self._rendered.move_to_end(key)
            return cached
        rendered = self._render_uncached(
            self._layout(text, font), font, rgb, halo_rgb, halo_width
        )
        self._rendered[key] = rendered
        if len(self._rendered) > self._cache_size:
            self._rendered.popitem(last=False)
        return rendered

    def clear(self) -> None:
        """Drop all cached layouts and rendered labels."""
        self._layouts.clear()
        self._rendered.clear()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _layout(self, text: LabelText, font: FontSpec) -> _LabelLayout:
        """Typeset a label into positioned runs (cached).

        Args:
            text: The label.
            font: The base font.

        Returns:
            The label's layout.
        """
        key = (text, font)
        cached = self._layouts.get(key)
        if cached is not None:
            self._layouts.move_to_end(key)
            return cached
        layout = self._layout_uncached(text, font)
        self._layouts[key] = layout
        if len(self._layouts) > self._cache_size:
            self._layouts.popitem(last=False)
        return layout

    def _layout_uncached(self, text: LabelText, font: FontSpec) -> _LabelLayout:
        """Compute run positions and the overall size of a label.

        Args:
            text: The label.
            font: The base font.

        Returns:
            The label's layout.
        """
        base = self._fonts.metrics(font.name, font.size)
        sup_size = max(1, round(font.size * SUPERSCRIPT_SCALE))
        sup = self._fonts.metrics(font.name, sup_size)
        sup_baseline = base.digit_height - round(base.digit_height * SUPERSCRIPT_RAISE)
        runs: list[_RunLayout] = []
        x = 0
        sign_width = 0
        overflow = 0
        for index, run in enumerate(text.runs):
            is_sup = run.role == "superscript"
            run_size = sup_size if is_sup else font.size
            chars = self._with_fallbacks(run.text, font.name, run_size)
            width = self._fonts.get(font.name, run_size).size(chars)[0]
            baseline = sup_baseline if is_sup else base.digit_height
            runs.append(_RunLayout(chars, run_size, x, baseline))
            if index == 0 and run.role == "sign":
                sign_width = width
            if is_sup:
                overflow = max(overflow, sup.digit_height - sup_baseline)
            x += width
        size = TextSize(x, base.digit_height, sign_width, overflow, base.descent)
        return _LabelLayout(tuple(runs), size)

    def _with_fallbacks(self, chars: str, name: str | None, size: int) -> str:
        """Replace characters the font cannot draw with ASCII stand-ins.

        Args:
            chars: Text of a run.
            name: Font name.
            size: Font size.

        Returns:
            ``chars`` with unsupported minus/times signs replaced.
        """
        for special, fallback in _FALLBACK_GLYPHS.items():
            if special in chars and not self._fonts.has_glyph(name, size, special):
                chars = chars.replace(special, fallback)
        return chars

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_uncached(
        self,
        layout: _LabelLayout,
        font: FontSpec,
        rgb: tuple[int, int, int, int],
        halo_rgb: tuple[int, int, int, int] | None,
        halo: int,
    ) -> RenderedText:
        """Render a typeset label, halo first and glyphs on top.

        Args:
            layout: The label's layout.
            font: The base font.
            rgb: Resolved text colour.
            halo_rgb: Resolved halo colour, or ``None``.
            halo: Halo thickness in pixels.

        Returns:
            The rendered label.
        """
        placed = []
        for run in layout.runs:
            run_font = self._fonts.get(font.name, run.size)
            glyphs = run_font.render(run.text, True, rgb[:3])
            top = run.baseline - run_font.get_ascent()
            placed.append((run, run_font, glyphs, top))

        left = -halo
        top = min(t for *_, t in placed) - halo if placed else 0
        right = max((r.x + g.get_width() for r, _, g, _ in placed), default=0) + halo
        bottom = max((t + g.get_height() for *_, g, t in placed), default=0) + halo
        surface = pygame.Surface(
            (max(1, right - left), max(1, bottom - top)), pygame.SRCALPHA
        )
        # Fill transparent pixels with the colour drawn first so that blending
        # antialiased edges onto them never produces a dark fringe.
        first = halo_rgb if halo_rgb is not None else rgb
        surface.fill((*first[:3], 0))

        if halo_rgb is not None and halo > 0:
            for run, run_font, _, run_top in placed:
                outline = _render_outline(run_font, run.text, halo_rgb, halo)
                surface.blit(outline, (run.x - halo - left, run_top - halo - top))
        for run, _, glyphs, run_top in placed:
            surface.blit(glyphs, (run.x - left, run_top - top))
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            surface = surface.convert_alpha()
        return RenderedText(surface, (-left, -top))


def _render_outline(
    font: pygame.font.Font, text: str, rgb: tuple[int, int, int, int], width: int
) -> pygame.Surface:
    """Render the outline (halo) of some text.

    Uses SDL_ttf's outline rendering. The result is ``2 * width`` pixels
    larger than the plain text in each direction, and plain text blitted at
    ``(width, width)`` lines up with it.

    Args:
        font: The font (its outline setting is restored afterwards).
        text: The characters to outline.
        rgb: Outline colour.
        width: Outline thickness in pixels.

    Returns:
        The rendered outline with per-pixel alpha.
    """
    previous = font.outline
    font.outline = width
    try:
        return font.render(text, True, rgb[:3])
    finally:
        font.outline = previous
