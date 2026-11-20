"""Colour resolution and compositing for the pygame backend.

The core describes colours as user-supplied values (names, hex strings or
tuples) and line paints as stacks of
:class:`~pygridplot.core.primitives.PaintLayer`. This module turns them into
concrete RGBA tuples.

When the background colour is known, a paint stack is flattened into one
**opaque** colour, so lines can be drawn with fast, perfectly crisp
``Surface.fill`` calls. Otherwise the stack is flattened into a single
translucent colour that the renderer alpha-blends onto whatever is already
on the surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pygame

from pygridplot.core.coords import ColorLike
from pygridplot.core.primitives import PaintLayer

RGBA = tuple[int, int, int, int]
"""A resolved colour with 8-bit channels."""


class ColorResolver:
    """Converts colour values to RGBA tuples and composites paint stacks.

    Results are cached, because the same handful of theme colours are
    resolved every frame.
    """

    def __init__(self) -> None:
        self._rgba: dict[object, RGBA] = {}
        self._paints: dict[tuple[object, ...], RGBA] = {}

    def rgba(self, color: ColorLike) -> RGBA:
        """Resolve any pygame-compatible colour value to an RGBA tuple.

        Args:
            color: A colour name, hex string, RGB/RGBA tuple or
                ``pygame.Color``.

        Returns:
            The colour as ``(r, g, b, a)``.

        Raises:
            ValueError: If pygame does not recognise the colour.
        """
        key = _hashable(color)
        cached = self._rgba.get(key)
        if cached is None:
            c = pygame.Color(color)
            cached = (c.r, c.g, c.b, c.a)
            self._rgba[key] = cached
        return cached

    def flatten(
        self, layers: Sequence[PaintLayer], background: ColorLike | None
    ) -> RGBA:
        """Composite a paint stack into a single colour.

        Args:
            layers: Paint layers, bottom first.
            background: The opaque colour already under the paint, or
                ``None`` if unknown.

        Returns:
            With a background, the final opaque colour the pixels will have.
            Without one, a single colour whose alpha-blend onto any background
            matches blending the layers one after another.
        """
        key = (
            tuple((_hashable(layer.color), layer.opacity) for layer in layers),
            None if background is None else _hashable(background),
        )
        cached = self._paints.get(key)
        if cached is None:
            base = (0.0, 0.0, 0.0, 0.0)
            if background is not None:
                r, g, b, _ = self.rgba(background)
                base = (float(r), float(g), float(b), 1.0)
            for layer in layers:
                base = _over(self.rgba(layer.color), layer.opacity, base)
            cached = _to_rgba(base)
            self._paints[key] = cached
        return cached


def _over(
    color: RGBA, opacity: float, dst: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Composite one colour over another with the standard "over" operator.

    Args:
        color: The source colour (straight, non-premultiplied alpha).
        opacity: Extra opacity multiplier for the source, in ``[0, 1]``.
        dst: The destination as ``(r, g, b, alpha)`` with alpha in ``[0, 1]``.

    Returns:
        The composited colour in the same format as ``dst``.
    """
    src_a = (color[3] / 255.0) * min(max(opacity, 0.0), 1.0)
    dst_r, dst_g, dst_b, dst_a = dst
    out_a = src_a + dst_a * (1.0 - src_a)
    if out_a <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    keep = dst_a * (1.0 - src_a)
    return (
        (color[0] * src_a + dst_r * keep) / out_a,
        (color[1] * src_a + dst_g * keep) / out_a,
        (color[2] * src_a + dst_b * keep) / out_a,
        out_a,
    )


def _to_rgba(color: tuple[float, float, float, float]) -> RGBA:
    """Round a float colour to 8-bit channels.

    Args:
        color: ``(r, g, b, alpha)`` with channels in ``[0, 255]`` and alpha in
            ``[0, 1]``.

    Returns:
        The colour as an 8-bit RGBA tuple.
    """
    r, g, b, a = color
    return (
        _channel(r),
        _channel(g),
        _channel(b),
        _channel(a * 255.0),
    )


def _channel(value: float) -> int:
    """Round and clamp one colour channel to ``[0, 255]``.

    Args:
        value: Channel value.

    Returns:
        The nearest integer in ``[0, 255]``.
    """
    return min(max(int(value + 0.5), 0), 255)


def _hashable(color: object) -> object:
    """Return a hashable key for a colour value.

    Args:
        color: Any colour value (``pygame.Color`` and lists are unhashable).

    Returns:
        ``color`` itself if hashable, else a tuple of its channels.
    """
    if isinstance(color, (str, tuple)) or not isinstance(color, Iterable):
        return color
    return tuple(color)
