"""Tests for drawing sequences (:mod:`pygridplot.core.plot.sequences`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot import sequences
from pygridplot.core.plot.builders import sequence_primitives
from pygridplot.core.plot.sequences import (
    BASE,
    LOG_FADE_LEVELS,
    MIN_SPACING_PX,
    SequenceLattice,
    as_terms,
    check_start,
    integer_arguments,
    sequence_lattice,
    visible_indices,
)
from pygridplot.core.plot.style import SequenceStyle
from pygridplot.core.primitives import Markers, Stroke
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 800, 600)
STYLE = SequenceStyle()
BLACK = (0, 0, 0)


def lattice(view: View, start: int = -(2**52)) -> SequenceLattice:
    """Return a view's lattice, which must exist."""
    terms = sequence_lattice(view, start, 0.0)
    assert terms is not None
    return terms


def valuations(n: NDArray[np.int64]) -> NDArray[np.float64]:
    """How many times 3 divides each n (infinite for 0)."""
    count = np.where(n == 0, np.inf, 0.0)
    rest = n.copy()
    for _ in range(40):
        divides = (rest % BASE == 0) & (rest != 0)
        count += divides
        rest = np.where(divides, rest // BASE, rest)
    return count


def brute_force(view: View, start: int) -> tuple[set[int], set[int]]:
    """The rule applied to every visible index: the terms drawn, and the fading ones."""
    visible = visible_indices(view, start, 0.0)
    assert visible is not None
    n = np.arange(visible[0], visible[1] + 1, dtype=np.int64)
    spacing = view.scale_x * view.transform_x.derivative(n.astype(float))
    e = np.ceil(np.log(MIN_SPACING_PX / (BASE * spacing)) / math.log(BASE))
    v = valuations(n)
    drawn = v >= np.maximum(e, 0)
    fading = drawn & (e >= 0) & (v == e)
    return set(n[drawn].tolist()), set(n[fading].tolist())


def test_every_term_is_drawn_when_they_are_far_enough_apart() -> None:
    view = View(RECT, (0.3, 0.0), 50.0)  # 50 px per term
    terms = lattice(view)
    assert np.array_equal(terms.n, np.arange(-7, 9))
    assert np.all(terms.fade == 1.0)
    assert lattice(view, start=2).n[0] == 2


def test_far_out_terms_are_thinned_to_powers_of_three() -> None:
    view = View(RECT, (1234.5, 0.0), 0.001)  # a thousand terms per pixel
    terms = lattice(view)
    steps = np.diff(terms.n)
    step = int(steps.min())
    assert step == BASE ** round(math.log(step, BASE))  # a power of three
    assert np.all(terms.n % step == 0)
    assert terms.n.size <= RECT.width / (MIN_SPACING_PX / BASE) + 2
    full = terms.n[terms.fade == 1.0]
    assert np.all(np.diff(full) * view.scale_x >= MIN_SPACING_PX - 1e-9)


def test_thinning_keeps_both_parities() -> None:
    """Powers of two would keep only even terms, hiding (-1)ⁿ's negative half."""
    view = View(RECT, (0.0, 0.0), 0.05)
    terms = lattice(view)
    assert np.any(terms.n % 2 == 0)
    assert np.any(terms.n % 2 == 1)


@pytest.mark.parametrize(
    ("center", "scale", "log"),
    [
        ((10.0, 0.0), 3.0, False),
        ((0.0, 0.0), 0.15, False),
        ((0.0, 0.0), 0.011, False),
        ((100.0, 1.0), 300.0, True),  # n from 5 to 2000 on a log axis
        ((300.0, 1.0), 150.0, True),  # n from 1 to 10⁵
    ],
)
def test_the_lattice_follows_the_rule_term_by_term(
    center: tuple[float, float], scale: float, log: bool
) -> None:
    """The levels found by bisection give exactly the terms the rule draws.

    Only terms just starting to fade in (fade 0) may be missing.
    """
    view = View(RECT, center, scale, log_x=log)
    terms = set(lattice(view, start=1).n.tolist())
    drawn, fading = brute_force(view, start=1)
    assert terms <= drawn
    assert drawn - terms <= fading


def test_log_axes_crowd_terms_without_drawing_them_all() -> None:
    view = View(RECT, (1000.0, 1.0), 150.0, log_x=True)  # n from 2 to 5·10⁵
    terms = lattice(view, start=1)
    assert terms.n[0] >= 1
    assert terms.n.size < 4 * RECT.width / MIN_SPACING_PX
    near = terms.n[terms.n < 10]
    assert np.array_equal(near, np.arange(near[0], 10))  # every term while sparse
    levels = terms.fade * LOG_FADE_LEVELS
    assert np.allclose(levels, np.round(levels))  # fades in eighths


def test_nothing_overflows_however_far_out() -> None:
    view = View(RECT, (0.0, 0.0), 1e-300)
    terms = lattice(view)
    assert terms.n.size >= 1
    assert view.limits is not None  # the view clamps zooms; the lattice copes anyway
    assert sequence_lattice(View(ScreenRect(0, 0, 0, 0)), 0, 0.0) is None
    assert sequence_lattice(View(RECT, (0.0, 0.0), 50.0), 100, 0.0) is None  # past view


def visible_fades(view: View) -> dict[int, float]:
    terms = lattice(view)
    return dict(zip(terms.n.tolist(), terms.fade.tolist(), strict=True))


@settings(max_examples=150, deadline=None)
@given(
    exponent=st.floats(-4.0, 1.0),
    center=st.floats(-1e4, 1e4),
    log=st.booleans(),
)
def test_a_small_zoom_changes_every_fade_only_a_little(
    exponent: float, center: float, log: bool
) -> None:
    """No pops: terms appear and vanish only at fade zero, and change gently."""
    scale = 10.0**exponent * (150.0 if log else 1.0)
    where = 10.0 ** (abs(center) % 3.0) if log else center
    view = View(RECT, (where, 1.0), scale, log_x=log)
    before = visible_fades(view)
    view.zoom(1.002)
    after = visible_fades(view)
    largest = 1.0 / LOG_FADE_LEVELS if log else 0.03  # log fades step by eighths
    for n in set(before) | set(after):
        x = float(view.to_screen_x(float(n)))
        if RECT.left + 20 < x < RECT.right - 20:  # not entering or leaving
            assert abs(before.get(n, 0.0) - after.get(n, 0.0)) <= largest + 1e-9


@pytest.mark.parametrize(("dx", "log"), [(7, False), (-12, False), (9, True)])
def test_panning_draws_the_same_terms(dx: int, log: bool) -> None:
    view = View(RECT, (500.0 if log else 37.2, 1.0), 120.0 if log else 0.3, log_x=log)
    before = visible_fades(view)
    view.pan_pixels(dx, 0)
    after = visible_fades(view)
    shared = set(before) & set(after)
    assert len(shared) > 100
    assert all(before[n] == after[n] for n in shared)


def test_the_first_index_is_validated() -> None:
    assert check_start(3) == 3
    assert check_start(np.int64(-2)) == -2
    for bad in (1.5, True, "1", 2**60):
        with pytest.raises(ValueError, match="start"):
            check_start(bad)


def test_finite_terms_are_validated() -> None:
    np.testing.assert_array_equal(as_terms([1, 2.5, np.nan]), [1.0, 2.5, np.nan])
    assert as_terms(np.array([True, False])).dtype == np.float64
    original = np.array([1.0, 2.0])
    assert as_terms(original) is not original  # a copy
    with pytest.raises(ValueError, match="real"):
        as_terms(np.array([1 + 2j]))
    for bad in ([[1, 2], [3, 4]], "abc", 3.0, [1, "a"]):
        with pytest.raises(ValueError, match="1-D"):
            as_terms(bad)  # type: ignore[arg-type]


def test_single_indices_reach_functions_as_ints() -> None:
    term = integer_arguments(math.factorial)
    assert term(5.0) == 120
    array = np.array([1.0, 2.0])
    np.testing.assert_array_equal(integer_arguments(lambda n: n * 2)(array), [2, 4])
    assert integer_arguments(lambda n: n)(2.5) == 2.5  # not whole: passed on


# --- The primitives ----------------------------------------------------------


def split(
    primitives: tuple[Stroke | Markers, ...],
) -> tuple[list[Stroke], list[Markers]]:
    strokes = [p for p in primitives if isinstance(p, Stroke)]
    markers = [p for p in primitives if isinstance(p, Markers)]
    return strokes, markers


def test_finite_sequences_draw_a_stem_and_a_marker_per_term() -> None:
    view = View(RECT, (2.0, 1.0), 50.0)
    terms = np.array([1.0, -2.0, np.nan, 0.5])
    strokes, markers = split(sequence_primitives(view, terms, 1, STYLE, BLACK))
    assert len(strokes) == len(markers) == 1
    n = np.array([1, 2, 4])  # the NaN term is left out
    px, py = view.to_screen(n.astype(float), np.array([1.0, -2.0, 0.5]))
    np.testing.assert_allclose(markers[0].positions, np.column_stack((px, py)))
    _, base = view.to_screen(0.0, 0.0)
    for path, x, y in zip(strokes[0].paths, px, py, strict=True):
        np.testing.assert_allclose(path[:, 0], x)
        assert sorted(path[:, 1]) == pytest.approx(sorted((base, y)))
    assert markers[0].radius == STYLE.radius
    assert strokes[0].width == STYLE.width


def test_only_the_visible_part_of_a_long_sequence_is_built() -> None:
    view = View(RECT, (1000.0, 0.0), 50.0)
    terms = np.sin(np.arange(100_000.0))
    _, markers = split(sequence_primitives(view, terms, 0, STYLE, BLACK))
    assert 14 <= markers[0].positions.shape[0] <= 18
    assert (
        sequence_primitives(View(RECT, (-50.0, 0.0), 50.0), terms, 0, STYLE, BLACK)
        == ()
    )


def test_stems_start_at_the_baseline() -> None:
    view = View(RECT, (2.0, 1.0), 50.0)
    strokes, _ = split(
        sequence_primitives(view, np.array([1.5, 0.5]), 0, STYLE, BLACK, baseline=1.0)
    )
    _, foot = view.to_screen(0.0, 1.0)
    for path in strokes[0].paths:
        assert foot in path[:, 1]
    log = View(RECT, (2.0, 1.0), 50.0, log_y=True)
    strokes, _ = split(
        sequence_primitives(log, np.array([1.0, 10.0]), 1, STYLE, BLACK, baseline=0.1)
    )
    _, foot = log.to_screen(0.0, 0.1)
    assert all(foot in path[:, 1] for path in strokes[0].paths)  # finite on a log axis


def test_baselines_are_validated() -> None:
    assert sequences.check_baseline(2) == 2.0
    for bad in (math.nan, math.inf, "0", None, True):
        with pytest.raises(ValueError, match="baseline"):
            sequences.check_baseline(bad)


def test_stems_alone_and_markers_alone() -> None:
    view = View(RECT, (2.0, 0.0), 50.0)
    terms = np.ones(5)
    stems = SequenceStyle(radius=0)
    dots = SequenceStyle(width=0)
    assert all(
        isinstance(p, Stroke) for p in sequence_primitives(view, terms, 0, stems, BLACK)
    )
    assert all(
        isinstance(p, Markers) for p in sequence_primitives(view, terms, 0, dots, BLACK)
    )


def test_dense_terms_share_one_stem_per_pixel_column() -> None:
    """The union of a column's stems is one stem: the cost stays one per pixel."""
    view = View(RECT, (5000.0, 0.0), 0.08, equal_aspect=False)
    view.set_scale(0.08, 50.0)
    terms = np.sin(np.arange(10_000.0))
    strokes, markers = split(sequence_primitives(view, terms, 0, STYLE, BLACK))
    assert len(strokes[0].paths) <= RECT.width + 2
    assert markers[0].positions.shape[0] > 5 * len(strokes[0].paths)  # every term
    top = min(float(path[:, 1].min()) for path in strokes[0].paths)
    bottom = max(float(path[:, 1].max()) for path in strokes[0].paths)
    assert top == pytest.approx(float(markers[0].positions[:, 1].min()))
    assert bottom == pytest.approx(float(markers[0].positions[:, 1].max()))


def test_functions_and_their_failures() -> None:
    view = View(RECT, (3.0, 0.0), 50.0)
    _, markers = split(
        sequence_primitives(view, lambda n: 1 / n, 0, STYLE, BLACK)  # undefined at 0
    )
    xs, _ = view.to_screen(np.arange(1.0, 12.0), np.zeros(11))
    np.testing.assert_allclose(markers[0].positions[:, 0], xs)

    def broken(n: object) -> float:
        raise ValueError

    assert sequence_primitives(view, broken, 0, STYLE, BLACK) == ()
    factorial = integer_arguments(lambda n: math.factorial(n) / 100)
    _, markers = split(sequence_primitives(view, factorial, 0, STYLE, BLACK))
    _, py = view.to_screen(np.array([4.0]), np.array([0.24]))
    assert markers[0].positions[4, 1] == pytest.approx(py[0])


def test_log_y_stems_reach_the_bottom_and_skip_what_cannot_show() -> None:
    view = View(RECT, (3.0, 1.0), 50.0, log_y=True)
    terms = np.array([1.0, 10.0, -1.0, 0.0, 100.0])
    strokes, markers = split(sequence_primitives(view, terms, 1, STYLE, BLACK))
    assert markers[0].positions.shape[0] == 3  # -1 and 0 cannot be shown
    for path in strokes[0].paths:
        assert float(path[:, 1].max()) >= RECT.bottom  # to the bottom edge


def test_fading_terms_are_drawn_as_their_own_primitives() -> None:
    view = View(RECT, (0.0, 0.0), 0.2)  # thinned, part way through a fade
    terms = lattice(view)
    assert 0 < terms.fade.min() < 1
    primitives = sequence_primitives(view, lambda n: np.cos(n), 0, STYLE, BLACK)
    opacities = sorted({p.opacity for p in primitives})
    assert len(opacities) == 2
    assert opacities[-1] == 1.0
    assert opacities[0] == pytest.approx(terms.fade.min())


def evenly(values: NDArray[np.float64]) -> bool:
    return bool(np.allclose(np.diff(values), np.diff(values)[0]))


def test_terms_are_placed_exactly_far_from_the_origin() -> None:
    view = View(RECT, (1e9 + 0.5, 0.0), 40.0)
    _, markers = split(sequence_primitives(view, lambda n: 0 * n, 0, STYLE, BLACK))
    assert evenly(markers[0].positions[:, 0])
    assert sequences.MAX_INDEX == 2**52
