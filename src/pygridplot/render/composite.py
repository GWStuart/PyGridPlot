"""Blending coverage masks onto pygame surfaces.

The rasterisers in :mod:`pygridplot.render.raster` say how much of each
pixel a shape covers. This module paints a colour into those pixels with
standard "over" alpha blending:

* Sparse coverage (strokes) is blended pixel by pixel through
  ``pygame.surfarray``. That is fast because a stroke touches few pixels.
* Dense coverage (fills) is turned into a translucent patch and blitted, so
  pygame does the blending in C.

Surfaces with per-pixel alpha (for example a transparent layer you composite
later) are blended with the full alpha "over" formula, so antialiased edges
never get a dark fringe. Every function respects the surface's clip area.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pygame

from pygridplot.render.colors import RGBA
from pygridplot.render.raster import DenseCoverage, SparseCoverage

_EPSILON: Final = 1e-6

_DENSE_THRESHOLD: Final = 40_000
"""Above this many covered pixels, sparse coverage is blended as a patch."""


def blend_sparse(
    surface: pygame.Surface, coverage: SparseCoverage, color: RGBA, opacity: float = 1.0
) -> None:
    """Paint ``color`` into individually covered pixels.

    Args:
        surface: The surface to paint on.
        coverage: The covered pixels (each listed once).
        color: The colour; its alpha multiplies the coverage.
        opacity: Extra opacity multiplier in ``[0, 1]``.
    """
    clip = surface.get_clip()
    x, y = coverage.x, coverage.y
    inside = (x >= clip.left) & (x < clip.right) & (y >= clip.top) & (y < clip.bottom)
    if not inside.any():
        return
    x, y = x[inside], y[inside]
    alpha = coverage.alpha[inside] * (color[3] / 255.0) * opacity
    has_alpha = bool(surface.get_flags() & pygame.SRCALPHA)
    if not has_alpha and x.shape[0] > _DENSE_THRESHOLD:
        # Very many pixels (a dense tangle of curve): one blitted patch is
        # cheaper than blending each pixel individually.
        _blit_patch(surface, _densify(x, y, alpha), color, 1.0)
        return
    try:
        if has_alpha:
            _over_with_alpha(surface, x, y, alpha, color)
        else:
            _over_opaque(surface, x, y, alpha, color)
    except ValueError:  # pixel format without direct array access
        _blit_patch(surface, _densify(x, y, alpha), color, 1.0)


def blend_dense(
    surface: pygame.Surface, coverage: DenseCoverage, color: RGBA, opacity: float = 1.0
) -> None:
    """Paint ``color`` into a block of pixels with the given coverage.

    Args:
        surface: The surface to paint on.
        coverage: Coverage of a rectangular block of pixels.
        color: The colour; its alpha multiplies the coverage.
        opacity: Extra opacity multiplier in ``[0, 1]``.
    """
    if surface.get_flags() & pygame.SRCALPHA:
        rows, cols = np.nonzero(coverage.alpha)
        sparse = SparseCoverage(
            (cols + coverage.left).astype(np.intp),
            (rows + coverage.top).astype(np.intp),
            coverage.alpha[rows, cols],
        )
        blend_sparse(surface, sparse, color, opacity)
        return
    _blit_patch(surface, coverage, color, opacity)


def _over_opaque(
    surface: pygame.Surface,
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    color: RGBA,
) -> None:
    """Blend into a surface without per-pixel alpha.

    Args:
        surface: The surface.
        x: Pixel columns.
        y: Pixel rows.
        alpha: Effective opacity per pixel.
        color: The colour.
    """
    pixels = pygame.surfarray.pixels3d(surface)
    try:
        dst = pixels[x, y].astype(np.float64)
        src = np.array(color[:3], dtype=np.float64)
        out = dst + (src - dst) * alpha[:, None]
        pixels[x, y] = (out + 0.5).astype(np.uint8)
    finally:
        del pixels


def _over_with_alpha(
    surface: pygame.Surface,
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    color: RGBA,
) -> None:
    """Blend into a surface with per-pixel alpha using the full "over" formula.

    Args:
        surface: The surface.
        x: Pixel columns.
        y: Pixel rows.
        alpha: Effective opacity per pixel.
        color: The colour.
    """
    rgb = pygame.surfarray.pixels3d(surface)
    channel = pygame.surfarray.pixels_alpha(surface)
    try:
        dst_rgb = rgb[x, y].astype(np.float64)
        dst_a = channel[x, y].astype(np.float64) / 255.0
        keep = dst_a * (1.0 - alpha)
        out_a = alpha + keep
        src = np.array(color[:3], dtype=np.float64)
        out_rgb = (src * alpha[:, None] + dst_rgb * keep[:, None]) / np.maximum(
            out_a, _EPSILON
        )[:, None]
        rgb[x, y] = (out_rgb + 0.5).astype(np.uint8)
        channel[x, y] = (out_a * 255.0 + 0.5).astype(np.uint8)
    finally:
        del rgb, channel


def _densify(x: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> DenseCoverage:
    """Turn sparse coverage into a dense block.

    Args:
        x: Pixel columns.
        y: Pixel rows.
        alpha: Coverage per pixel.

    Returns:
        The same coverage as a rectangular block.
    """
    left, top = int(np.min(x)), int(np.min(y))
    block = np.zeros((int(np.max(y)) - top + 1, int(np.max(x)) - left + 1))
    block[y - top, x - left] = alpha
    return DenseCoverage(left, top, block)


def _blit_patch(
    surface: pygame.Surface, coverage: DenseCoverage, color: RGBA, opacity: float
) -> None:
    """Blend dense coverage by blitting a translucent patch (blending in C).

    Args:
        surface: The surface.
        coverage: The coverage block.
        color: The colour.
        opacity: Extra opacity multiplier.
    """
    height, width = coverage.alpha.shape
    if width == 0 or height == 0:
        return
    patch = pygame.Surface((width, height), pygame.SRCALPHA)
    patch.fill((color[0], color[1], color[2], 0))
    levels = coverage.alpha * (color[3] * opacity) + 0.5  # straight alpha, 8 bits
    channel = pygame.surfarray.pixels_alpha(patch)
    try:
        channel[...] = np.clip(levels, 0, 255).astype(np.uint8).T
    finally:
        del channel
    surface.blit(patch, (coverage.left, coverage.top))
