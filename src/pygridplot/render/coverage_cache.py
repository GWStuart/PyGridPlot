"""Keeping the rasterised coverage of primitives drawn again.

Rasterising a stroke or a fill (working out how much of each pixel it
covers) is most of the cost of drawing it; blending that coverage onto the
surface is cheap. When the plane reuses a plot between frames (see
:mod:`pygridplot.core.plot.memo`), it hands the renderer the *same*
primitive object again, so the renderer can keep the coverage it computed
and only blend it.

Entries are found by the primitive's identity and the clip box, and hold the
primitive itself, so an identity cannot be reused by another object while
its entry exists. The least recently used entries are dropped beyond
:data:`MAX_ENTRIES` entries or :data:`MAX_PIXELS` covered pixels in all.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Final, TypeVar

from pygridplot.render.raster import DenseCoverage, SparseCoverage

MAX_ENTRIES: Final = 256
"""Most primitives whose coverage is kept."""

MAX_PIXELS: Final = 2_000_000
"""Most covered pixels kept in all (about 30 MB of coverage)."""

CoverageT = TypeVar("CoverageT", SparseCoverage, DenseCoverage)


class CoverageCache:
    """The coverage of recently drawn primitives, for drawing them again."""

    def __init__(self) -> None:
        self._entries: OrderedDict[
            int, tuple[object, tuple[float, ...], SparseCoverage | DenseCoverage, int]
        ] = OrderedDict()
        self._pixels = 0

    def get(
        self,
        primitive: object,
        clip: tuple[float, ...],
        compute: Callable[[], CoverageT | None],
    ) -> CoverageT | None:
        """Return a primitive's coverage within a clip box, computing it if new.

        Args:
            primitive: The stroke or fill.
            clip: The clip box it is rasterised to.
            compute: Rasterises it.

        Returns:
            The coverage (``None`` if nothing is covered).
        """
        key = id(primitive)
        entry = self._entries.get(key)
        if entry is not None and entry[0] is primitive and entry[1] == clip:
            self._entries.move_to_end(key)
            return entry[2]  # type: ignore[return-value]
        coverage = compute()
        if coverage is None:
            return None
        if entry is not None:
            self._drop(key)
        size = (
            len(coverage)
            if isinstance(coverage, SparseCoverage)
            else coverage.alpha.size
        )
        if size <= MAX_PIXELS:
            self._entries[key] = (primitive, clip, coverage, size)
            self._pixels += size
            while len(self._entries) > MAX_ENTRIES or self._pixels > MAX_PIXELS:
                self._drop(next(iter(self._entries)))
        return coverage

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()
        self._pixels = 0

    def __len__(self) -> int:
        """Return the number of primitives whose coverage is kept."""
        return len(self._entries)

    def _drop(self, key: int) -> None:
        """Remove one entry."""
        _, _, _, size = self._entries.pop(key)
        self._pixels -= size
