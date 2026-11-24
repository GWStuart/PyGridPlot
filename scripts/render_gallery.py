"""Render a gallery of tricky grid and plot views to PNG files for visual review.

This is a development tool. It runs headless (no window), renders each view
in :data:`VIEWS` to ``<out>/<name>.png``, and also writes a contact sheet
``<out>/_sheet.png`` with every view at reduced size.

Compare galleries from before and after a change to catch regressions by
eye::

    python scripts/render_gallery.py --out renders/before
    # ... change something ...
    python scripts/render_gallery.py --out renders/after
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import pygridplot as pgp

Scene = Callable[[pgp.Grid, pygame.Surface], None]
"""Draws plots on top of a view's grid."""


def scene_test_set(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Plot the shared function test set from the project plan.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    for func in (lambda x: 1 / x, np.tan, np.sqrt, np.log, np.floor, lambda x: x**2):
        grid.plot(surface, func)


def scene_hard(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Plot functions that are hard to sample well.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.plot(surface, math.gamma)
    grid.plot(surface, lambda x: np.sin(x) / x)
    grid.plot(surface, lambda x: np.sin(1 / x))
    grid.plot(surface, lambda x: np.sqrt(np.sin(x)))


def scene_shapes(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw every marker shape, polylines and polygons (both fill rules).

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    for i, marker in enumerate(("circle", "square", "diamond", "cross", "plus")):
        x = -6 + 1.3 * i
        grid.points(surface, (x, 4.0), marker=marker, radius=6)  # type: ignore[arg-type]
        grid.points(
            surface,
            (x, 3.0),
            marker=marker,
            radius=6,
            edge_color="white",  # type: ignore[arg-type]
        )
    angles = np.pi / 2 + np.arange(5) * 4 * np.pi / 5
    star = np.column_stack((2 * np.cos(angles), 2 * np.sin(angles)))
    grid.polygon(surface, star + np.array([3.5, 2.5]), fill_rule="nonzero")
    grid.polygon(surface, star + np.array([7.0, 2.5]), fill_rule="evenodd")
    grid.polygon(surface, [(-6, -1), (-3, -1), (-4.5, 1.5)])
    theta = np.linspace(0, 6 * np.pi, 300)
    spiral = np.column_stack(
        (0.12 * theta * np.cos(theta), 0.12 * theta * np.sin(theta))
    )
    grid.polyline(surface, spiral + np.array([2.0, -2.5]), width=2)
    xs = np.linspace(-7, 7, 50)
    grid.points(surface, np.column_stack((xs, -4.5 + 0.4 * np.sin(xs))), radius=3)


def scene_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Plot power laws and exponentials, which are straight on log axes.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.plot(surface, lambda x: x)
    grid.plot(surface, lambda x: x**2)
    grid.plot(surface, np.sqrt)
    grid.plot(surface, lambda x: 1 / x)
    grid.plot(surface, np.exp)
    xs = np.geomspace(0.02, 50, 12)
    grid.points(surface, np.column_stack((xs, 3 * xs**0.5)), edge_color="white")


def scene_curves(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw parametric and polar curves: loops, cusps, poles and domain edges.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.polar(surface, lambda a: 2 * np.cos(4 * a))
    grid.polar(surface, lambda a: 3 * np.sqrt(np.cos(2 * a)))
    grid.polar(surface, lambda a: 0.2 * a, (0, 8 * math.pi), width=1.2)
    grid.parametric(
        surface,
        lambda t: 5 + 1.5 * np.cos(3 * t),
        lambda t: 2 + 1.5 * np.sin(2 * t),
        (0, 2 * math.pi),
    )
    grid.parametric(
        surface, lambda t: -5 + t - np.sin(t), lambda t: -3 + (1 - np.cos(t)), (-6, 6)
    )
    grid.parametric(surface, np.tan, lambda t: -4 + 0.3 * t, (-4, 4))


def scene_implicit(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw implicit curves: loops, crossings, poles, jumps and domain edges.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.implicit(surface, lambda x, y: x**2 + y**2, 9)
    grid.implicit(surface, lambda x, y: (x**2 + y**2) ** 2 - 8 * (x**2 - y**2))
    grid.implicit(
        surface,
        lambda x, y: (
            ((x - 5) ** 2 + (y - 2) ** 2 - 2) ** 3 - 3 * (x - 5) ** 2 * (y - 2) ** 3
        ),
    )
    grid.implicit(surface, lambda x, y: x**3 + y**3 - 6 * x * y)
    grid.implicit(surface, lambda x, y: y + 4 - np.tan(x), width=1.5)
    grid.implicit(surface, lambda x, y: y - 3 - np.floor(x) / 2, width=1.5)
    grid.implicit(surface, lambda x, y: y**2 - x**3 + 2 * x + 6, width=1.5)
    grid.implicit(
        surface, lambda x, y: np.sin(3 * x) * np.sin(3 * y), 0.4, width=1.2, opacity=0.6
    )


def scene_regions(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw inequalities and areas between graphs.

    A disc with its boundary, an annulus, a compound region, a region that
    ends at the asymptotes of ``tan``, the area between two crossing graphs
    and the area under ``1/x`` down to the bottom edge.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.inequality(surface, lambda x, y: (x + 5) ** 2 + (y - 2) ** 2 < 3)
    grid.implicit(surface, lambda x, y: (x + 5) ** 2 + (y - 2) ** 2, 3, width=1.5)
    grid.inequality(
        surface,
        lambda x, y: (
            ((x + 5) ** 2 + (y + 2.5) ** 2 > 0.6)
            & ((x + 5) ** 2 + (y + 2.5) ** 2 < 2.5)
        ),
    )
    grid.inequality(
        surface, lambda x, y: (y > (x - 1) ** 2 - 4) & (y < 1 - 0.5 * x) & (x < 2.5)
    )
    grid.inequality(surface, lambda x, y: (y < np.tan(x)) & (y > 2.5), opacity=0.35)
    grid.fill_between(surface, np.sin, lambda x: 0.4 * x - 3, domain=(4, 9))
    grid.plot(surface, np.sin, domain=(4, 9), width=1.5)
    grid.plot(surface, lambda x: 0.4 * x - 3, domain=(4, 9), width=1.5)
    grid.fill_between(surface, lambda x: 1 / x, -math.inf, domain=(-2.5, -0.4))


def scene_fields(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a vector field, a slope field with a solution, and arrows.

    The left half shows a rotation field by magnitude, the right half the
    slope field of ``dy/dx = x - y`` with one solution, and arrows of
    several widths and opacities lie on top, including two translucent ones
    drawn in one call that cross without darkening.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.vector_field(
        surface, lambda x, y: (np.where(x < 0, -(y + 1), np.nan), x), scale=6.0
    )
    grid.slope_field(surface, lambda x, y: np.where(x > 0.5, x - y, np.nan))
    grid.plot(surface, lambda x: x - 1 + 2 * np.exp(-x), domain=(0.5, 9))
    grid.arrow(surface, (-6.0, -4.0), [(-2.0, -4.0), (-3.0, -1.5)], width=3.0)
    grid.arrow(
        surface,
        [(-1.0, 3.0), (1.0, 4.5)],
        [(3.0, 4.5), (2.5, 2.5)],
        width=6.0,
        opacity=0.5,
    )
    grid.arrow(surface, (2.0, -4.5), (2.2, -4.4), head_size=12.0)  # all head


def scene_draggable(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw draggable points of every kind, one hovered and one being dragged.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    free = grid.draggable((1.2345, 2.5))
    slider = grid.draggable((-6.0, 1.5), lock="x", y_range=(0, 3))
    grid.draggable((2.0, -3.5), lock="y", coordinates="always")
    ring = grid.draggable(curve=(np.cos, np.sin, (0, 2 * math.pi)), t=2.2)
    free.set_interaction(hovered=True, dragging=False)
    ring.set_interaction(hovered=True, dragging=True)
    grid.polyline(surface, [(-6, 0), (-6, 3)], width=4, opacity=0.4)
    grid.plot(surface, lambda x: slider.y * np.sin(x - free.x) + free.y - slider.y)
    grid.parametric(surface, np.cos, np.sin, (0, 2 * math.pi), width=1.5)
    grid.draw_controls(surface)


def scene_derivatives(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a function, its first three derivatives and a tangent line.

    The function has a jump and a kink, where the derivatives must jump
    cleanly, without the spikes of finite differences.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(x) + 0.3 * np.abs(x - 2) + np.where(x < -3, 1.0, 0.0)

    for order, width in ((3, 1.2), (2, 1.5), (1, 2.0), (0, 2.5)):
        grid.plot(surface, pgp.derivative(f, order), width=width)
    a = 0.7
    value, slope = pgp.derivatives(f, a, 1)
    grid.plot(surface, lambda x: value + slope * (x - a), width=1.5, opacity=0.8)
    grid.points(surface, (a, value), edge_color="white")


def scene_integrals(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw areas and antiderivatives of functions with jumps, kinks and poles.

    The area under a function with a jump and a kink is shaded between two
    bounds, with its antiderivative from the first bound passing through
    the area's value at the second; the antiderivative of ``1/x`` from 1
    (``log x``) ends at the pole; that of ``floor`` is piecewise linear; and
    midpoint rectangles approximate an area.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(x) + 0.3 * np.abs(x - 2) + np.where(x < -3, 1.0, 0.0)

    a, b = -3.5, 2.0
    grid.fill_between(surface, f, domain=(a, b))
    grid.plot(surface, f)
    grid.plot(surface, pgp.antiderivative(f, a), width=2.0)
    grid.points(surface, (b, pgp.integral(f, a, b)), edge_color="white")
    grid.plot(surface, pgp.antiderivative(lambda x: 1 / x, 1.0), width=2.0)
    grid.plot(surface, pgp.antiderivative(np.floor, 4.0), width=1.5, domain=(4, 9))
    x = np.linspace(-8.0, -5.0, 7)
    h = np.cos((x[:-1] + x[1:]) / 2) - 3
    gap = np.full(h.size, np.nan)
    boxes = np.column_stack(
        [x[:-1], 0 * h - 3, x[:-1], h, x[1:], h, x[1:], 0 * h - 3, gap, gap]
    )
    grid.polyline(surface, boxes.reshape(-1, 2), width=1.2)
    grid.plot(surface, lambda t: np.cos(t) - 3, domain=(-9, -4))


def scene_series(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw Taylor polynomials of ``sin`` and Fourier sums of a square and a sawtooth.

    The Taylor polynomials of degrees 1, 3, 5 and 7 about 0 peel away from
    ``sin`` one after the other; the square wave's partial sum overshoots
    at its jumps (the Gibbs phenomenon).

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.plot(surface, np.sin, width=3.0, color=(40, 40, 40), opacity=0.5)
    for degree in (1, 3, 5, 7):
        grid.plot(surface, pgp.taylor(np.sin, 0.0, degree), width=1.8)
    square = pgp.fourier(lambda x: np.sign(x) + 3, 15)
    grid.plot(surface, square)
    saw = pgp.fourier(lambda x: x / 2 - 3.5, 7)
    grid.plot(surface, saw)


def scene_sequences(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw sequences: a spectrum, an alternating sequence and scalar-only terms.

    A square wave's spectrum (an array) has stems only at the odd
    harmonics; ``(-1)ⁿ / n`` closes in on -3 with stems from that limit;
    a row of Pascal's triangle comes from ``math.comb``, one term at a time;
    and an exponential runs over negative ``n`` in diamonds, ending at 0.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.sequence(surface, 3 * pgp.fourier(np.sign, 9).amplitudes, width=3)
    grid.sequence(
        surface, lambda n: -3 + 1.5 * (-1) ** n / n, start=1, baseline=-3, width=1.5
    )
    grid.sequence(
        surface,
        lambda n: math.comb(6, n + 7) / 6 - 1 if n < 0 else math.nan,
        start=-7,
        width=1.5,
    )
    grid.sequence(  # NaN for n >= 0: no terms there
        surface,
        lambda n: np.where(n < 0, -3 * np.exp(n / 3), np.nan),
        start=-8,
        marker="diamond",
        width=0,
    )


def scene_sequences_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw sequences on log axes: power laws are straight rows of markers.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.sequence(surface, lambda n: 30 / n**2, start=1, width=0, radius=2.5)
    grid.sequence(surface, 10 * pgp.fourier(np.abs, 60).amplitudes, radius=3)


def scene_sliders(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw sliders of every kind in three corners, one hovered and one dragged.

    A continuous, an integer (hovered) and a log slider stack in the
    bottom-left; a stepped one is being dragged in the top-right; a
    disabled one is faded in the bottom-right.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    a = grid.slider("a", (0.0, 3.0), value=1.5)
    n = grid.slider("n", (1, 20), value=7, step=1)
    grid.slider("r", (0.01, 100.0), value=3.0, log=True)
    phase = grid.slider(
        "phase", (-math.pi, math.pi), value=0.5, step=math.pi / 12, corner="top-right"
    )
    grid.slider("off", (0, 1), value=0.3, enabled=False, corner="bottom-right")
    n.set_interaction(hovered=True, dragging=False)
    phase.set_interaction(hovered=True, dragging=True)
    grid.plot(surface, lambda x: a.value * np.sin(n.value * x / 4 + phase.value))
    grid.draw_controls(surface)


def scene_annotations(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw labelled plots with a legend, and text marking points.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.fill_between(
        surface,
        np.sin,
        np.cos,
        domain=(math.pi / 4, 5 * math.pi / 4),
        label="sin x > cos x",
    )
    grid.plot(surface, np.sin, label="sin x")
    grid.plot(surface, np.cos, label="cos x")
    peak = (math.pi / 2, 1.0)
    grid.points(surface, [peak, (0.0, 1.0)], color=(40, 40, 40), radius=3.5)
    grid.text(surface, "max of sin", peak, anchor="bottom", gap=8)
    grid.text(surface, "max of cos", (0.0, 1.0), anchor="bottom-right", gap=6)
    grid.legend(surface)


def scene_annotations_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw filter gains on log-log axes, with a legend and text on the slopes.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    for order in (1, 2, 3):
        color = grid.theme.palette[order - 1]
        grid.plot(
            surface,
            lambda w, n=order: 1 / np.sqrt(1 + w ** (2 * n)),
            color=color,
            label=f"order {order}",
        )
        grid.text(
            surface,
            f"−{20 * order} dB/decade",
            (10.0, 10.0**-order),
            anchor="bottom-left",
            gap=6,
            color=color,
        )
    grid.legend(surface, corner="bottom-left")


def scene_legend(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw every kind of plot, each labelled, and the legend of all of them.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.vector_field(surface, lambda x, y: (-y, x), label="vector field")
    grid.slope_field(surface, lambda x, y: x * y / 4, spacing=80, label="slope field")
    grid.inequality(
        surface, lambda x, y: (x + 3) ** 2 + (y + 2) ** 2 < 1, label="region"
    )
    grid.fill_between(surface, np.sin, -1, domain=(1, 3), label="area")
    grid.plot(surface, np.sin, label="graph")
    grid.parametric(
        surface,
        np.cos,
        lambda t: np.sin(2 * t),
        (0, 2 * math.pi),
        label="curve",
        width=1.5,
    )
    grid.polygon(surface, [(2, -3), (4, -3), (3, -1.5)], label="polygon")
    grid.polyline(surface, [(-4, 2), (-3, 3), (-2, 2.5)], label="polyline")
    grid.points(surface, [(-1, -2), (0, -2.5)], marker="diamond", label="points")
    grid.sequence(surface, lambda n: 3 / n, start=1, label="sequence")
    grid.arrow(surface, (-4, -1), (-2, 0), label="arrow")
    grid.legend(surface, corner="top-left")


def scene_legend_sliders(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a legend in the corner of two sliders: it goes below their panel.

    The legend is drawn first, in the plane's first frame, so it goes by
    the sliders it expects.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    a = grid.slider("a", (0.0, 2.0), value=1.0, corner="top-right")
    k = grid.slider("k", (1, 5), value=2, step=1, corner="top-right")
    for m in (1, 2):
        grid.plot(
            surface,
            lambda x, m=m: a.value * np.sin(m * k.value * x / 2) / m,
            label=f"a sin({m}kx/2)/{m}",
        )
    grid.legend(surface)
    grid.draw_controls(surface)


def scene_text(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw text in each of the nine anchors, and lines of text.

    The points are in a 3 x 3 grid, each labelled with its anchor (the
    point of the text's box placed at the point, 6 px away), so the
    ``"top-left"`` text hangs below and to the right of its point.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    anchors = (
        ("top-left", "top", "top-right"),
        ("left", "center", "right"),
        ("bottom-left", "bottom", "bottom-right"),
    )
    for row, names in enumerate(anchors):
        for column, anchor in enumerate(names):
            point = (-6.0 + 2.5 * column, 3.5 - 1.5 * row)
            grid.points(surface, point, color=(40, 40, 40), radius=3)
            grid.text(surface, anchor, point, anchor=anchor, gap=6)  # type: ignore[arg-type]
    grid.text(surface, "centred\non (3, 2)", (3, 2), size=16, color=(41, 111, 189))
    grid.text(
        surface,
        "left-aligned lines\nanchored top-left",
        (2, -1),
        anchor="top-left",
        style=pgp.TextStyle(color=(204, 59, 52), opacity=0.8),
    )
    grid.text(surface, "right-aligned\nlines", (-1, -2), anchor="right", gap=4)
    grid.text(surface, "no halo, over the axis", (3, -0.05), halo=False)


def scene_data(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw measurements with error bars on both axes, and a model through them.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(1)
    xs = np.arange(-6.0, 7.0)
    ys = 0.4 * xs + 1 + rng.normal(0, 0.3, xs.size)
    grid.plot(surface, lambda x: 0.4 * x + 1, width=1.5, label="model")
    grid.points(
        surface,
        np.column_stack((xs, ys)),
        y_error=0.25 + 0.03 * np.abs(xs),
        edge_color="white",
        label="measured",
    )
    grid.points(
        surface,
        [(-4.0, 3.5), (-2.0, 3.0)],
        x_error=[0.4, 0.8],
        y_error=[[0.2, 0.5], [0.6, 0.3]],
        marker="square",
        label="x and y errors",
    )
    grid.legend(surface, corner="bottom-right")


def scene_scatter(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a scatter plot coloured and sized by a value, and named colours.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(400, 2)) * 1.5
    far = np.clip(np.hypot(pts[:, 0], pts[:, 1]) / 4, 0, 1)
    colors = (1 - far)[:, None] * (40, 100, 200) + far[:, None] * np.array(
        (220, 60, 40)
    )
    grid.points(
        surface, pts, color=colors, radius=2 + 5 * far, edge_color="white", edge_width=1
    )
    named = ["crimson", "seagreen", "gold", (120, 60, 180), (0, 0, 0, 120)]
    grid.points(
        surface,
        [(x, 4.0) for x in range(-6, 4, 2)],
        color=named,
        radius=[3, 5, 7, 9, 11],
    )


def scene_steps(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw an empirical distribution function and steps of each kind.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    samples = np.sort(np.random.default_rng(3).normal(size=40))
    ecdf = np.column_stack((samples, np.arange(1, 41) / 40))
    grid.polyline(surface, ecdf, step="post", label="post: empirical distribution")
    data = np.column_stack((np.arange(-6, 7, 2.0), [1.5, 2.5, 2.0, 3.5, 3.0, 4.2, 3.8]))
    for step, lift in (("pre", 0.0), ("mid", -4.0)):
        shifted = data - np.array([0.0, -lift])
        grid.polyline(surface, shifted, step=step, label=step)  # type: ignore[arg-type]
        grid.points(surface, shifted, radius=3, color=(40, 40, 40))
    grid.legend(surface, corner="bottom-right")


def scene_bars(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw bar charts: heights, side by side, below the axis, with error bars.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    heights = [3.0, 1.5, 4.0, 2.5, 3.5]
    grid.bars(surface, np.column_stack((np.arange(-6, -1) - 0.2, heights)), width=0.4,
               label="north")  # fmt: skip
    grid.bars(surface, np.column_stack((np.arange(-6, -1) + 0.2, heights[::-1])),
               width=0.4, label="south")  # fmt: skip
    tops = np.column_stack((np.arange(-6, -1) - 0.2, heights))
    grid.points(surface, tops, y_error=0.4, radius=0, color=(40, 40, 40), error_cap=6)
    grid.bars(surface, [2.0, -1.5, 1.0, -2.5], start=1, label="up and down")
    for x, h in zip(range(1, 5), [2.0, -1.5, 1.0, -2.5], strict=True):
        grid.text(surface, f"{h:g}", (x, h), anchor="bottom" if h > 0 else "top", gap=4)
    grid.legend(surface, corner="top-right")


def scene_histogram(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw two histograms as densities, and the density of one.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(4)
    grid.histogram(surface, rng.normal(size=3000), density=True, label="N(0, 1)")
    grid.histogram(surface, rng.normal(1.5, 0.5, 800), density=True, width=0.1,
                    label="N(1.5, 0.5²)")  # fmt: skip
    grid.plot(
        surface, lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), label="density"
    )
    grid.legend(surface)


def scene_histogram_far(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a histogram of narrow bins zoomed out: the lines between them gone.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    samples = np.random.default_rng(5).standard_cauchy(5000)
    grid.histogram(surface, samples, width=0.25)


def scene_data_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a log-normal histogram on log axes, with points and error bars.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.histogram(
        surface, np.random.default_rng(6).lognormal(2, 1, 5000), label="counts"
    )
    grid.points(
        surface,
        [(1, 10), (10, 100), (100, 30)],
        y_error=[[9, 50, 40], [20, 50, 10]],
        x_error=0.5,
        label="points",
    )
    grid.legend(surface)


def scene_color_map(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw points coloured by a value, with the legend and a colour bar.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(500, 2)) * 1.6
    value = np.hypot(pts[:, 0], pts[:, 1])
    grid.points(
        surface,
        pts,
        color=pgp.color_map(value, "viridis", (0, 4)),
        radius=4,
        label="points",
    )
    grid.legend(surface)
    grid.color_bar(surface, "viridis", (0, 4), title="distance")
    sizes = 10 ** rng.uniform(-1, 3, 60)
    places = np.column_stack((rng.uniform(-6, 6, 60), rng.uniform(-4, 4, 60)))
    colors = pgp.color_map(sizes, "coolwarm", (0.1, 1000), log=True)
    grid.points(surface, places, color=colors, radius=3, marker="square")
    grid.color_bar(surface, "coolwarm", (0.1, 1000), log=True, title="size",
                   corner="bottom-left")  # fmt: skip


def scene_bar_kinds(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw stacked, floating and horizontal bars.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    a, b, c = (
        np.array([2, 3, 1, 2.5]),
        np.array([1, 2, 2, 1]),
        np.array([1, 0.5, 2, 1.5]),
    )
    grid.bars(surface, a, start=-6, label="a")
    grid.bars(surface, a + b, start=-6, baseline=a, label="b")
    grid.bars(surface, a + b + c, start=-6, baseline=a + b, label="c")
    grid.bars(surface, [(0, 3.5), (1, 2), (2, 4)], baseline=[1, -1, 2.5], width=0.5,
               label="floating")  # fmt: skip
    grid.bars(surface, [5, 2.5, 6.5], start=-3, baseline=3, horizontal=True, width=0.6,
               label="horizontal")  # fmt: skip
    grid.legend(surface, corner="top-left")


def scene_cumulative(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a cumulative histogram and the empirical distribution function.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    samples = np.random.default_rng(8).normal(size=300)
    grid.histogram(surface, samples, width=0.25, cumulative=True, density=True,
                    label="cumulative histogram")  # fmt: skip
    ecdf = np.column_stack((np.sort(samples), np.arange(1, 301) / 300))
    grid.polyline(surface, ecdf, step="post", width=1.5, label="empirical distribution")
    grid.legend(surface, corner="top-left")


def scene_boxes(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw violins with narrow box plots inside, the groups named.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(9)
    groups = [
        rng.normal(5, 1, 300),
        rng.gamma(2, 1.2, 300) + 2,
        np.r_[rng.normal(7, 0.5, 150), rng.normal(9.5, 0.6, 100)],
        np.clip(rng.standard_t(2, 300) + 5, 0.5, 12),
    ]
    grid.violin_plot(surface, groups, label="violins")
    grid.box_plot(surface, groups, width=0.12, opacity=0.9, label="boxes")
    grid.legend(surface)


def scene_boxes_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw horizontal box plots and a violin on a log x axis.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    rng = np.random.default_rng(10)
    groups = [rng.lognormal(m, s, 200) for m, s in ((1, 1), (3, 0.6), (2, 1.5))]
    grid.box_plot(surface, groups, horizontal=True)
    grid.violin_plot(surface, [rng.lognormal(1, 1, 400)], start=3, horizontal=True)


def scene_categories(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw bars of many named categories (thinned when crowded).

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.bars(surface, np.abs(np.sin(np.arange(60) / 4)) * 4 + 0.5)


CATEGORY_NAMES = [f"item {k}" for k in range(60)]
"""Names for the categories scene."""


def scene_dashes(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw every kind of line dashed or dotted, with a legend.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    grid.plot(surface, np.sin, dash="dashed", label="dashed")
    grid.plot(surface, lambda x: 1 / (x - 1), dash="dotted", label="dotted, a pole")
    grid.plot(surface, np.sqrt, dash="dashed", width=5, domain=(0, 4), label="thick")
    grid.parametric(
        surface,
        lambda t: 2 * np.cos(t) - 3.5,
        lambda t: np.sin(3 * t) + 2,
        (0, 2 * np.pi),
        dash=(14, 5, 0, 5),
        label="dash-dot",
    )
    grid.implicit(
        surface, lambda x, y: x**2 + y**2, [4.0, 9.0], dash="dashed", width=1.5
    )
    grid.polygon(
        surface, [(3, 2), (6, 2.5), (4.5, 4.2)], dash="dashed", label="outline"
    )
    grid.polyline(
        surface, [(-6, -3), (-5, -2), (-4, -3), (-3, -2)], step="post", dash="dotted"
    )
    grid.polyline(surface, [(1, -10), (1, 10)], dash=(4, 4), width=1.0, color="gray")
    grid.legend(surface)


def scene_dashes_log(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw dashed lines on log-log axes: power laws are straight.

    Args:
        grid: The plane.
        surface: The surface to draw on.
    """
    for k in (-1, 0.5, 1, 2):
        grid.plot(surface, lambda x, k=k: x**k, dash="dashed", label=f"x^{k}")
    grid.polyline(surface, [(1, 0.01), (1, 1000)], dash="dotted", color="gray")
    grid.legend(surface, corner="bottom-right")


SCENES: dict[str, Scene] = {
    "test_set": scene_test_set,
    "hard": scene_hard,
    "shapes": scene_shapes,
    "log": scene_log,
    "curves": scene_curves,
    "implicit": scene_implicit,
    "regions": scene_regions,
    "fields": scene_fields,
    "draggable": scene_draggable,
    "derivatives": scene_derivatives,
    "integrals": scene_integrals,
    "series": scene_series,
    "sequences": scene_sequences,
    "sequences_log": scene_sequences_log,
    "sliders": scene_sliders,
    "annotations": scene_annotations,
    "annotations_log": scene_annotations_log,
    "legend": scene_legend,
    "legend_sliders": scene_legend_sliders,
    "text": scene_text,
    "data": scene_data,
    "scatter": scene_scatter,
    "steps": scene_steps,
    "bars": scene_bars,
    "histogram": scene_histogram,
    "histogram_far": scene_histogram_far,
    "data_log": scene_data_log,
    "color_map": scene_color_map,
    "bar_kinds": scene_bar_kinds,
    "cumulative": scene_cumulative,
    "boxes": scene_boxes,
    "boxes_log": scene_boxes_log,
    "categories": scene_categories,
    "dashes": scene_dashes,
    "dashes_log": scene_dashes_log,
}


@dataclass(frozen=True)
class GalleryView:
    """One view to render.

    Attributes:
        name: File name (without extension).
        center: World point at the middle of the plane (default: the
            plane's default).
        scale: Pixels per world unit or per decade (one number, or ``(x, y)``).
        theme: Theme name: ``"light"`` or ``"dark"``.
        overrides: Theme fields to override.
        embedded: Draw the plane in a sub-rectangle of the surface.
        scene: Name of the plots to draw on top (see :data:`SCENES`), if any.
        log_x: Make the x axis logarithmic.
        log_y: Make the y axis logarithmic.
        x_range: Initial x range (overrides centre and scale for x).
        y_range: Initial y range (overrides centre and scale for y).
        titles: The x and y axis titles.
        names: Category names for the x and y axes.
    """

    name: str
    center: tuple[float, float] | None = None
    scale: float | tuple[float, float] | None = None
    theme: str = "light"
    overrides: dict[str, object] = field(default_factory=dict)
    embedded: bool = False
    scene: str | None = None
    log_x: bool = False
    log_y: bool = False
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    titles: tuple[str | None, str | None] = (None, None)
    names: tuple[object, object] = (None, None)


ARROWS: dict[str, object] = {"show_arrows": True}
"""Theme overrides for arrowheads on the axes."""

VIEWS: tuple[GalleryView, ...] = (
    GalleryView("01_default"),
    GalleryView("02_zoom_in", center=(0.3, 0.2), scale=5e4),
    GalleryView("03_zoom_in_sci", center=(0.0, 0.0), scale=2e7),
    GalleryView("04_zoom_out", scale=0.5),
    GalleryView("05_zoom_out_sci", scale=2e-5),
    GalleryView("06_axis_near_bottom", center=(0.0, 5.6)),
    GalleryView("07_axis_below_screen", center=(3.0, 40.0)),
    GalleryView("08_axis_near_left", center=(8.7, 0.0)),
    GalleryView("09_axes_off_corner", center=(-40.0, -30.0)),
    GalleryView("10_far_offset_deep", center=(12345.678, -98.7654), scale=1e7),
    GalleryView("11_fade_1_to_2", scale=100 / math.sqrt(2)),
    GalleryView("12_fade_2_to_5", scale=100 / math.sqrt(10)),
    GalleryView("13_dark", theme="dark"),
    GalleryView("14_embedded", embedded=True, center=(2.0, 1.0)),
    GalleryView("15_unequal_aspect", scale=(50.0, 5000.0), center=(0.0, 0.03)),
    GalleryView("16_snap_mode", overrides={"transition": "snap"}),
    GalleryView("17_big_negative", center=(-2.5e6, -1.5e6), scale=2e-4),
    GalleryView("18_axis_near_top_right", center=(-8.8, -5.9)),
    GalleryView("19_functions", scale=60.0, scene="test_set"),
    GalleryView("20_functions_dark", scale=60.0, theme="dark", scene="test_set"),
    GalleryView("21_hard_functions", scale=60.0, scene="hard"),
    GalleryView("22_zoomed_function", center=(1.0, 0.8415), scale=4e5, scene="hard"),
    GalleryView("23_shapes", scale=55.0, scene="shapes"),
    GalleryView("24_shapes_embedded", scale=40.0, embedded=True, scene="shapes"),
    GalleryView("25_loglog_default", log_x=True, log_y=True, scene="log"),
    GalleryView("26_semilog_y", log_y=True, scale=(60.0, 120.0), scene="log"),
    GalleryView("27_semilog_x", log_x=True, center=(1.0, 2.0), scene="log"),
    GalleryView("28_log_coarse", log_x=True, log_y=True, scale=20.0),
    GalleryView("29_log_fade_decades", log_x=True, log_y=True, scale=100 / 2**0.5),
    GalleryView("30_log_paper", log_x=True, log_y=True, scale=110.0),
    GalleryView("31_log_125", log_x=True, log_y=True, scale=400.0),
    GalleryView("32_log_fine", log_x=True, log_y=True, scale=1500.0, center=(3, 3)),
    GalleryView("33_log_deep", log_x=True, log_y=True, scale=1e5, center=(3.7, 3.7)),
    GalleryView("34_log_sci", log_x=True, log_y=True, center=(1e8, 1e-7)),
    GalleryView("35_log_dark", log_x=True, log_y=True, theme="dark", scene="log"),
    GalleryView(
        "36_ranges", x_range=(-2.0, 8.0), y_range=(-1.0, 3.0), scene="test_set"
    ),
    GalleryView("37_log_ranges", log_y=True, x_range=(0, 10), y_range=(0.1, 1e4)),
    GalleryView("38_curves", scale=60.0, scene="curves"),
    GalleryView("39_curves_dark", scale=60.0, theme="dark", scene="curves"),
    GalleryView("40_curve_cusp_zoom", center=(-5.0, -3.0), scale=3e4, scene="curves"),
    GalleryView("41_curves_loglog", log_x=True, log_y=True, scene="curves"),
    GalleryView("42_implicit", scale=60.0, scene="implicit"),
    GalleryView("43_implicit_dark", scale=60.0, theme="dark", scene="implicit"),
    GalleryView("44_implicit_zoom_out", scale=12.0, scene="implicit"),
    GalleryView("45_implicit_double_point", scale=2e3, scene="implicit"),
    GalleryView("46_implicit_loglog", log_x=True, log_y=True, scene="implicit"),
    GalleryView("47_draggable", scale=60.0, scene="draggable"),
    GalleryView("48_draggable_dark", scale=60.0, theme="dark", scene="draggable"),
    GalleryView(
        "49_draggable_zoomed", center=(1.234, 2.5), scale=6e4, scene="draggable"
    ),
    GalleryView("50_derivatives", scale=60.0, scene="derivatives"),
    GalleryView(
        "51_derivatives_kink",
        center=(2.0, -0.4),
        scale=(2e3, 400.0),
        scene="derivatives",
    ),
    GalleryView("52_regions", scale=60.0, scene="regions"),
    GalleryView("53_regions_dark", scale=60.0, theme="dark", scene="regions"),
    GalleryView("54_regions_loglog", log_x=True, log_y=True, scene="regions"),
    GalleryView(
        "55_regions_zoomed", center=(-5.0, 2.0 + 3**0.5), scale=2e4, scene="regions"
    ),
    GalleryView("56_fields", scale=60.0, scene="fields"),
    GalleryView("57_fields_dark", scale=60.0, theme="dark", scene="fields"),
    GalleryView("58_fields_fading", scale=60.0 * 1.38, scene="fields"),
    GalleryView("59_fields_loglog", log_x=True, log_y=True, scene="fields"),
    GalleryView("60_integrals", scale=60.0, scene="integrals"),
    GalleryView("61_integrals_semilog_x", log_x=True, scene="integrals"),
    GalleryView(  # the corner of the antiderivative where f jumps
        "62_integrals_zoomed_kink", center=(-3.0, 1.34104), scale=3e4, scene="integrals"
    ),
    GalleryView("63_series", scale=60.0, scene="series"),
    GalleryView("64_series_dark", scale=60.0, theme="dark", scene="series"),
    GalleryView("65_sequences", scale=60.0, scene="sequences"),
    GalleryView("66_sequences_dark", scale=60.0, theme="dark", scene="sequences"),
    GalleryView(  # far out along x: terms thinned, stems merged per column
        "67_sequences_zoomed_out", scale=(0.05, 60.0), scene="sequences"
    ),
    GalleryView("68_sequences_loglog", log_x=True, log_y=True, scene="sequences_log"),
    GalleryView("69_sliders", scale=60.0, scene="sliders"),
    GalleryView("70_sliders_dark", scale=60.0, theme="dark", scene="sliders"),
    GalleryView("71_sliders_embedded", scale=40.0, embedded=True, scene="sliders"),
    GalleryView(  # the panels stay put however the view moves
        "72_sliders_zoomed", center=(1e4, -3e3), scale=3e-3, scene="sliders"
    ),
    GalleryView(
        "73_titles_arrows",
        scale=60.0,
        overrides=ARROWS,
        scene="annotations",
        titles=("x", "y"),
    ),
    GalleryView(
        "74_titles_arrows_dark",
        scale=60.0,
        theme="dark",
        overrides=ARROWS,
        scene="annotations",
        titles=("time t (s)", "signal"),
    ),
    GalleryView(  # both axes off screen: the titles pinned, x riding on the labels
        "75_titles_pinned",
        center=(9.0, 6.0),
        scale=60.0,
        overrides=ARROWS,
        scene="annotations",
        titles=("time t (s)", "signal"),
    ),
    GalleryView(
        "76_titles_loglog",
        log_x=True,
        log_y=True,
        x_range=(0.01, 300.0),
        y_range=(1e-4, 4.0),
        overrides=ARROWS,
        scene="annotations_log",
        titles=("frequency ω (rad/s)", "gain"),
    ),
    GalleryView("77_legend_every_kind", scale=60.0, scene="legend"),
    GalleryView("78_legend_every_kind_dark", scale=60.0, theme="dark", scene="legend"),
    GalleryView("79_legend_with_sliders", scale=60.0, scene="legend_sliders"),
    GalleryView("80_text", scale=60.0, scene="text"),
    GalleryView("81_data_errors", scale=60.0, scene="data", titles=("x", "y")),
    GalleryView("82_data_scatter", scale=60.0, scene="scatter"),
    GalleryView("83_data_scatter_dark", scale=60.0, theme="dark", scene="scatter"),
    GalleryView("84_data_steps", scale=60.0, scene="steps"),
    GalleryView("85_data_bars", scale=60.0, scene="bars"),
    GalleryView(
        "86_data_histograms",
        x_range=(-4.0, 4.0),
        y_range=(-0.1, 0.9),
        scene="histogram",
    ),
    GalleryView(  # bins of 0.25 a pixel wide: the lines between them faded out
        "87_data_histogram_far",
        x_range=(-40.0, 40.0),
        y_range=(-10.0, 450.0),
        scene="histogram_far",
    ),
    GalleryView(
        "88_data_loglog",
        log_x=True,
        log_y=True,
        x_range=(0.1, 1000.0),
        y_range=(0.5, 3000.0),
        scene="data_log",
    ),
    GalleryView("89_color_map", scale=60.0, scene="color_map"),
    GalleryView("90_color_map_dark", scale=60.0, theme="dark", scene="color_map"),
    GalleryView(
        "91_bar_kinds", x_range=(-7.0, 7.5), y_range=(-4.0, 7.5), scene="bar_kinds"
    ),
    GalleryView(
        "92_cumulative", x_range=(-3.5, 3.5), y_range=(-0.1, 1.15), scene="cumulative"
    ),
    GalleryView(  # the axis below the data: the names pinned to the bottom
        "93_boxes_violins",
        x_range=(-0.8, 3.8),
        y_range=(0.5, 12.5),
        scene="boxes",
        names=(["normal", "gamma", "two peaks", "heavy tails"], None),
    ),
    GalleryView(
        "94_boxes_log",
        log_x=True,
        x_range=(0.1, 1000.0),
        y_range=(-1.0, 3.8),
        scene="boxes_log",
    ),
    GalleryView(
        "95_categories",
        x_range=(-1.0, 15.0),
        y_range=(-1.0, 6.0),
        scene="categories",
        names=(CATEGORY_NAMES, None),
    ),
    GalleryView(  # zoomed out: every 4th name or so, the others fading
        "96_categories_crowded",
        x_range=(-2.0, 62.0),
        y_range=(-1.0, 6.0),
        scene="categories",
        names=(CATEGORY_NAMES, None),
    ),
    GalleryView("97_dashes", center=(0.0, 0.5), scale=60.0, scene="dashes"),
    GalleryView(
        "98_dashes_dark", center=(0.0, 0.5), scale=60.0, theme="dark", scene="dashes"
    ),
    GalleryView(  # zoomed in on sin x far from the origin: anchors placed precisely
        "99_dashes_far",
        center=(1000.2, math.sin(1000.2)),
        scale=4000.0,
        scene="dashes",
    ),
    GalleryView(
        "100_dashes_log",
        log_x=True,
        log_y=True,
        x_range=(0.01, 100.0),
        y_range=(0.01, 1000.0),
        scene="dashes_log",
    ),
)


def theme_for(view: GalleryView) -> pgp.GridTheme:
    """Return the theme for a gallery view, with its overrides applied.

    Args:
        view: The gallery view.

    Returns:
        The theme to render with.
    """
    base = pgp.DARK if view.theme == "dark" else pgp.LIGHT
    return replace(base, **view.overrides)  # type: ignore[arg-type]


def render_view(view: GalleryView, size: tuple[int, int]) -> pygame.Surface:
    """Render one gallery view to a new surface.

    Args:
        view: The view to render.
        size: Surface size in pixels.

    Returns:
        The rendered surface.
    """
    surface = pygame.Surface(size)
    surface.fill((90, 96, 110))
    rect = None
    if view.embedded:
        rect = pygame.Rect(220, 40, size[0] - 260, size[1] - 80)
    grid = pgp.Grid(
        rect,
        x_range=view.x_range,
        y_range=view.y_range,
        log_x=view.log_x,
        log_y=view.log_y,
        center=view.center,
        scale=view.scale,
        equal_aspect=False if isinstance(view.scale, tuple) else None,
        theme=theme_for(view),
    )
    grid.draw(
        surface,
        x_title=view.titles[0],
        y_title=view.titles[1],
        x_labels=view.names[0],  # type: ignore[arg-type]
        y_labels=view.names[1],  # type: ignore[arg-type]
    )
    if view.scene is not None:
        SCENES[view.scene](grid, surface)
    return surface


def contact_sheet(
    images: list[tuple[str, pygame.Surface]], columns: int
) -> pygame.Surface:
    """Arrange half-size thumbnails of the images in a grid.

    Args:
        images: ``(name, surface)`` pairs.
        columns: Thumbnails per row.

    Returns:
        The contact sheet.
    """
    thumb_w, thumb_h = (s // 2 for s in images[0][1].get_size())
    rows = math.ceil(len(images) / columns)
    sheet = pygame.Surface((columns * (thumb_w + 8) + 8, rows * (thumb_h + 24) + 8))
    sheet.fill((60, 60, 60))
    font = pygame.font.Font(None, 20)
    for index, (name, image) in enumerate(images):
        col, row = index % columns, index // columns
        x = 8 + col * (thumb_w + 8)
        y = 8 + row * (thumb_h + 24)
        sheet.blit(font.render(name, True, (230, 230, 230)), (x, y))
        sheet.blit(pygame.transform.smoothscale(image, (thumb_w, thumb_h)), (x, y + 16))
    return sheet


def main() -> None:
    """Render the gallery as configured on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("renders/gallery"))
    parser.add_argument("--size", default="900x600", help="surface size, WxH")
    parser.add_argument("--only", nargs="*", help="render only these view names")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    args.out.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.display.set_mode((1, 1))
    rendered = []
    for view in VIEWS:
        if args.only and view.name not in args.only:
            continue
        image = render_view(view, size)  # type: ignore[arg-type]
        pygame.image.save(image, args.out / f"{view.name}.png")
        rendered.append((view.name, image))
    if rendered:
        pygame.image.save(contact_sheet(rendered, 3), args.out / "_sheet.png")
    print(f"rendered {len(rendered)} views to {args.out}")
    pygame.quit()


if __name__ == "__main__":
    main()
