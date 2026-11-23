"""Loading and caching fonts for the pygame backend.

Fonts are resolved from a :class:`~pygridplot.core.primitives.FontSpec`:

* ``name=None`` uses the bundled DejaVu Sans, so labels look the same on
  every platform without any configuration.
* A path to an existing ``.ttf``/``.otf`` file loads that file.
* Anything else is looked up as a system font name with
  ``pygame.font.match_font``. If nothing matches, a warning is issued once
  and the bundled font is used instead.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pygame

_BUNDLED_FONT = "DejaVuSans.ttf"
_DIGITS = "0123456789"


def bundled_font_path() -> Path:
    """Return the filesystem path of the bundled default font.

    Returns:
        Path to ``DejaVuSans.ttf`` inside the installed package.
    """
    resource = resources.files("pygridplot") / "assets" / "fonts" / _BUNDLED_FONT
    return Path(str(resource))


@dataclass(frozen=True, slots=True)
class FontMetrics:
    """Measurements of a loaded font that label layout relies on.

    Attributes:
        ascent: Distance in pixels from the top of a rendered line of text to
            the baseline.
        digit_height: Height in pixels of the digits above the baseline (all
            digits in a lining font share one height).
        descent: Distance in pixels from the baseline to the bottom of the
            lowest letters (g, y, brackets).
    """

    ascent: int
    digit_height: int
    descent: int = 0


class FontLibrary:
    """Loads pygame fonts on demand and caches them by name and size."""

    def __init__(self) -> None:
        self._fonts: dict[tuple[str | None, int], pygame.font.Font] = {}
        self._metrics: dict[tuple[str | None, int], FontMetrics] = {}
        self._glyphs: dict[tuple[str | None, int, str], bool] = {}
        self._paths: dict[str | None, str] = {}
        self._warned: set[str] = set()

    def get(self, name: str | None, size: int) -> pygame.font.Font:
        """Return the font ``name`` at ``size`` points, loading it if needed.

        Args:
            name: ``None`` for the bundled font, a font file path, or a system
                font name.
            size: Size in points.

        Returns:
            The loaded ``pygame.font.Font``.
        """
        key = (name, size)
        font = self._fonts.get(key)
        if font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.Font(self._resolve_path(name), size)
            self._fonts[key] = font
        return font

    def metrics(self, name: str | None, size: int) -> FontMetrics:
        """Return the layout metrics of a font.

        Args:
            name: Font name as accepted by :meth:`get`.
            size: Size in points.

        Returns:
            The font's :class:`FontMetrics`.
        """
        key = (name, size)
        metrics = self._metrics.get(key)
        if metrics is None:
            font = self.get(name, size)
            tops = [m[3] for m in font.metrics(_DIGITS) if m is not None]
            digit_height = max(tops) if tops else font.get_ascent()
            metrics = FontMetrics(
                font.get_ascent(), digit_height, max(0, -font.get_descent())
            )
            self._metrics[key] = metrics
        return metrics

    def has_glyph(self, name: str | None, size: int, char: str) -> bool:
        """Return whether a font can draw a character.

        Args:
            name: Font name as accepted by :meth:`get`.
            size: Size in points.
            char: A single character.

        Returns:
            ``True`` if the font has a glyph for ``char``.
        """
        key = (name, size, char)
        known = self._glyphs.get(key)
        if known is None:
            known = self.get(name, size).metrics(char)[0] is not None
            self._glyphs[key] = known
        return known

    def _resolve_path(self, name: str | None) -> str:
        """Turn a font name into a file path, with a fallback to the bundled font.

        Args:
            name: Font name as accepted by :meth:`get`.

        Returns:
            Path of the font file to load.
        """
        cached = self._paths.get(name)
        if cached is not None:
            return cached
        path: str | None = None
        if name is None:
            path = str(bundled_font_path())
        elif os.path.isfile(name):
            path = name
        else:
            path = pygame.font.match_font(name)
            if path is None:
                if name not in self._warned:
                    self._warned.add(name)
                    warnings.warn(
                        f"font {name!r} not found; using the bundled default font",
                        stacklevel=4,
                    )
                path = str(bundled_font_path())
        self._paths[name] = path
        return path
