"""Tests for finding regions (:mod:`pygridplot.core.plot.regions`).

Like the implicit sampler's tests, these call the sampler directly with
conditions of ``(u, w)``, which are plain ``(x, y)`` here, over an
800 × 600 px plane at 50 px per unit, sampled every 2 px.

The main check is independent of how the region is built: points are
classified by the even-odd rule against the returned rings, and wherever
that disagrees with the condition itself, the point must lie within a pixel
of the region's outline.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.regions import RegionRings, boundary_cells, sample_region

Condition = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

SCALE = 50.0
"""Pixels per unit."""
STEP = (2 / SCALE, 2 / SCALE)
"""The fine grid step: 2 px."""
U_RANGE = (-8.0, 8.0)
W_RANGE = (-6.0, 6.0)


def sample(
    condition: Condition,
    config: ImplicitSampling | None = None,
    u_range: tuple[float, float] = U_RANGE,
    w_range: tuple[float, float] = W_RANGE,
) -> RegionRings | None:
    """Sample a region over the standard area."""
    return sample_region(
        condition, u_range, w_range, STEP, config or ImplicitSampling()
    )


def world_rings(region: RegionRings | None) -> list[NDArray[np.float64]]:
    """Return a result's rings in (transformed) world coordinates."""
    assert region is not None
    origin, step = np.array(region.origin), np.array(region.step)
    return [(ring + origin) * step for ring in region.rings]


def signed_area(ring: NDArray[np.float64]) -> float:
    """Return a ring's signed area (positive anticlockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def area(region: RegionRings | None) -> float:
    """Return the area of a region (its holes wind the other way)."""
    return sum(signed_area(ring) for ring in world_rings(region))


def segments(rings: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Return every edge of the rings, as rows ``(x0, y0, x1, y1)``."""
    return np.concatenate(
        [np.column_stack((ring, np.roll(ring, -1, axis=0))) for ring in rings]
    )


def even_odd(points: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray:
    """Classify points by the even-odd rule (a ray towards +x)."""
    px, py = points[:, 0:1], points[:, 1:2]
    x0, y0, x1, y1 = (edges[:, k][None, :] for k in range(4))
    straddles = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    crossings = straddles & (crossing_x > px)
    return np.asarray(crossings.sum(axis=1) % 2 == 1)


def distance_to_edges(
    points: NDArray[np.float64], edges: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return each point's distance to the nearest edge."""
    p = points[:, None, :]
    a, b = edges[None, :, :2], edges[None, :, 2:]
    ab = b - a
    length = np.maximum(np.sum(ab * ab, axis=2), 1e-300)
    t = np.clip(np.sum((p - a) * ab, axis=2) / length, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return np.asarray(np.sqrt(np.sum((p - nearest) ** 2, axis=2)).min(axis=1))


def assert_classifies(condition: Condition, region: RegionRings | None) -> None:
    """Check the region against the condition at many points of the plane."""
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (rng.uniform(-7.5, 7.5, 3000), rng.uniform(-5.5, 5.5, 3000))
    )
    with np.errstate(all="ignore"):
        truth = np.asarray(condition(points[:, 0], points[:, 1]), dtype=np.float64)
    truth = np.isfinite(truth) & (truth != 0)
    rings = world_rings(region) if region is not None else []
    if not rings:
        assert not truth.any()
        return
    edges = segments(rings)
    found = even_odd(points, edges)
    wrong = found != truth
    if wrong.any():
        # Only points next to the outline may be classified differently.
        assert distance_to_edges(points[wrong], edges).max() * SCALE < 1.0


CONDITIONS: dict[str, Condition] = {
    "disc": lambda x, y: x**2 + y**2 < 4,
    "outside a disc": lambda x, y: x**2 + y**2 > 4,
    "annulus": lambda x, y: (x**2 + y**2 > 1) & (x**2 + y**2 < 9),
    "below a curve": lambda x, y: y < np.sin(x),
    "right of a curve": lambda x, y: x > np.sin(y),
    "below tan": lambda x, y: (y < np.tan(x)) & (y > -2),
    "floor steps": lambda x, y: y < np.floor(x),
    "checkerboard": lambda x, y: np.sin(2 * x) * np.sin(2 * y) > 0.1,
    "or": lambda x, y: (np.abs(x - 3) < 1) | (np.abs(y + 2) < 0.5),
    "not": lambda x, y: ~((np.abs(x) < 2) & (np.abs(y) < 1)),
    "undefined half": lambda x, y: np.sqrt(x) + 0 * y > -1,
    "zero or one": lambda x, y: np.where(x * y > 1, 1.0, 0.0),
}


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_points_are_classified_correctly(name: str) -> None:
    condition = CONDITIONS[name]
    assert_classifies(condition, sample(condition))


def test_disc_area_is_accurate() -> None:
    region = sample(lambda x, y: x**2 + y**2 < 4)
    assert len(world_rings(region)) == 1
    # Vertices lie on the circle, so the polygon is inscribed: a little small.
    assert area(region) == pytest.approx(4 * math.pi, rel=5e-4)


def test_annulus_has_a_hole() -> None:
    region = sample(lambda x, y: (x**2 + y**2 > 1) & (x**2 + y**2 < 9))
    rings = world_rings(region)
    assert len(rings) == 2
    assert area(region) == pytest.approx(8 * math.pi, rel=5e-4)
    assert not even_odd(np.array([[0.0, 0.0]]), segments(rings))[0]


def test_boundaries_are_refined_to_a_fraction_of_a_pixel() -> None:
    (ring,) = world_rings(sample(lambda x, y: x**2 + y**2 < 4))
    error_px = np.abs(np.hypot(ring[:, 0], ring[:, 1]) - 2) * SCALE
    assert error_px.max() < 0.15  # 3 halvings of a 2 px edge: within 1/8 px


def test_without_refinement_the_boundary_is_within_a_pixel() -> None:
    region = sample(lambda x, y: x**2 + y**2 < 4, ImplicitSampling(refine_steps=0))
    (ring,) = world_rings(region)
    error_px = np.abs(np.hypot(ring[:, 0], ring[:, 1]) - 2) * SCALE
    assert 0.3 < error_px.max() < 1.05  # midpoints of 2 px edges


def test_regions_end_at_asymptotes() -> None:
    condition = CONDITIONS["below tan"]
    rings = world_rings(sample(condition))
    edges = segments(rings)
    near = np.array([[math.pi / 2 - 0.05, 5.0], [math.pi / 2 + 0.05, 5.0]])
    assert even_odd(near, edges).tolist() == [True, False]


def test_undefined_points_are_outside() -> None:
    def broken(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x > 0, np.inf, np.nan) + 0 * y  # never a usable value

    assert sample(broken) is None
    region = sample(lambda x, y: np.log(x) + 0 * y > -100)  # x > 0
    left_edge = min(ring[:, 0].min() for ring in world_rings(region))
    assert abs(left_edge) * SCALE < 0.15  # refined onto x = 0


def test_true_everywhere_is_one_rectangle() -> None:
    region = sample(lambda x, y: 0 * x + 0 * y + 1)
    (ring,) = world_rings(region)
    assert len(ring) == 4
    assert ring[:, 0].min() <= U_RANGE[0]
    assert ring[:, 0].max() >= U_RANGE[1]
    assert ring[:, 1].min() <= W_RANGE[0]
    assert ring[:, 1].max() >= W_RANGE[1]


def test_false_everywhere_is_nothing() -> None:
    assert sample(lambda x, y: x**2 + y**2 < -1) is None


def test_regions_much_larger_than_their_boundary_are_cheap() -> None:
    """Only the boundary's neighbourhood is evaluated on the fine grid."""
    small = sample(lambda x, y: x**2 + y**2 < 0.25)
    large = sample(lambda x, y: x**2 + y**2 > 0.25)
    full_grid = (16 / STEP[0]) * (12 / STEP[1])
    for region in (small, large):
        assert region is not None
        assert region.evaluations < full_grid / 8


@settings(max_examples=40, deadline=None)
@given(di=st.integers(-80, 80), dj=st.integers(-80, 80))
def test_panning_moves_regions_rigidly(di: int, dj: int) -> None:
    """Shifting the region by whole fine steps shifts its outline exactly."""
    du, dw = di * STEP[0], dj * STEP[1]

    def folium(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(x**3 + y**3 - 6 * x * y + 0.5 < 0, dtype=np.float64)

    before = np.concatenate(world_rings(sample(folium)))
    after = np.concatenate(world_rings(sample(lambda x, y: folium(x - du, y - dw))))
    inside = (np.abs(before[:, 0] + du) < 7) & (np.abs(before[:, 1] + dw) < 5)
    inside &= (np.abs(before[:, 0]) < 7) & (np.abs(before[:, 1]) < 5)

    def keys(points: NDArray[np.float64]) -> set[tuple[float, float]]:
        rounded = np.round(points / STEP[0], 5) + 0.0  # + 0.0 turns -0 into 0
        return {(float(a), float(b)) for a, b in rounded}

    assert keys(before[inside] + (du, dw)) <= keys(after)


@pytest.mark.parametrize("budget", [1000, 5000, 20000, 100_000])
def test_budget_is_respected(budget: int) -> None:
    config = ImplicitSampling(max_samples=budget)
    for condition in (
        CONDITIONS["disc"],
        CONDITIONS["checkerboard"],
        CONDITIONS["below tan"],
    ):
        calls = 0

        def counted(
            x: NDArray[np.float64], y: NDArray[np.float64], f: Condition = condition
        ) -> NDArray[np.float64]:
            nonlocal calls
            calls += x.size
            return f(x, y)

        region = sample(counted, config)
        assert region is not None
        assert calls == region.evaluations <= budget


def test_small_budgets_coarsen_the_grid_but_keep_the_region() -> None:
    region = sample(lambda x, y: x**2 + y**2 < 9, ImplicitSampling(max_samples=2000))
    assert region is not None
    assert region.step[0] > STEP[0]  # coarser than asked
    assert area(region) == pytest.approx(9 * math.pi, rel=0.02)


def test_huge_areas_coarsen_the_coarse_grid() -> None:
    """A 40,000 × 30,000 px area still fits the budget."""
    region = sample(
        lambda x, y: x**2 + y**2 < 90_000, u_range=(-400, 400), w_range=(-300, 300)
    )
    assert region is not None
    assert region.evaluations <= ImplicitSampling().max_samples
    assert area(region) == pytest.approx(90_000 * math.pi, rel=1e-3)


def test_unusable_areas_give_nothing() -> None:
    disc = CONDITIONS["disc"]
    assert sample(disc, u_range=(-math.inf, 1.0)) is None
    assert sample(disc, w_range=(math.nan, 1.0)) is None
    assert sample_region(disc, U_RANGE, W_RANGE, (0.0, 0.04)) is None
    assert sample(disc, u_range=(-1e12, 1e12)) is None  # too many steps


def test_boundary_cells() -> None:
    inside = np.zeros((6, 7))
    inside[2:4, 3] = 1.0
    cells = boundary_cells(inside)
    assert cells.shape == (5, 6)
    # The cells around the two inside nodes disagree, widened by one.
    expected = np.zeros((5, 6), dtype=bool)
    expected[0:5, 1:5] = True
    assert np.array_equal(cells, expected)
    assert not boundary_cells(np.ones((4, 4))).any()
    assert not boundary_cells(np.zeros((4, 4))).any()
