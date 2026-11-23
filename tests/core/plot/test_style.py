"""Tests for plot styles and the palette colour cycle (:mod:`core.style`)."""

from __future__ import annotations

import pytest

from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    ColorCycle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
    with_overrides,
)


def test_default_styles_are_valid() -> None:
    assert LineStyle().width == 2.5
    assert PointStyle().marker == "circle"
    assert PolygonStyle().fill is True
    assert FillStyle().opacity == 0.25
    assert FillStyle().color is None
    assert ArrowStyle().head_size is None  # automatic
    assert ArrowStyle(head_size=0).head_size == 0  # no head
    assert (SequenceStyle().width, SequenceStyle().radius) == (2.0, 3.5)
    assert SequenceStyle(width=0).radius > 0  # markers alone
    assert SequenceStyle(radius=0).width > 0  # stems alone
    assert PointStyle(radius=0).radius == 0  # error bars alone
    assert (PointStyle().error_width, PointStyle().error_cap) == (1.5, 8.0)
    assert (BarStyle().opacity, BarStyle().edge_width) == (0.5, 1.5)
    assert BarStyle(edge_width=0).edge_width == 0  # no outline


@pytest.mark.parametrize(
    "make",
    [
        lambda: LineStyle(width=0),
        lambda: LineStyle(width=float("nan")),
        lambda: LineStyle(opacity=1.5),
        lambda: PointStyle(radius=-1),
        lambda: PointStyle(marker="star"),  # type: ignore[arg-type]
        lambda: PointStyle(edge_width=-0.5),
        lambda: PolygonStyle(width=-1),
        lambda: PolygonStyle(fill_opacity=2),
        lambda: PolygonStyle(fill_rule="winding"),  # type: ignore[arg-type]
        lambda: FillStyle(opacity=-0.1),
        lambda: FillStyle(opacity=float("nan")),
        lambda: ArrowStyle(width=0),
        lambda: ArrowStyle(head_size=-1),
        lambda: ArrowStyle(head_size=float("inf")),
        lambda: ArrowStyle(opacity=2),
        lambda: SequenceStyle(width=0, radius=0),  # nothing to draw
        lambda: SequenceStyle(width=-1),
        lambda: SequenceStyle(radius=float("inf")),
        lambda: SequenceStyle(marker="star"),  # type: ignore[arg-type]
        lambda: SequenceStyle(opacity=-1),
        lambda: PointStyle(error_width=0),
        lambda: PointStyle(error_cap=-1),
        lambda: BarStyle(opacity=1.5),
        lambda: BarStyle(edge_width=-1),
        lambda: BarStyle(edge_width=float("nan")),
    ],
)
def test_invalid_styles_are_rejected(make: object) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        make()  # type: ignore[operator]


def test_polygon_outline_can_be_switched_off() -> None:
    assert PolygonStyle(width=0).width == 0


def test_with_overrides_ignores_none_and_validates() -> None:
    base = LineStyle(width=3, color="red")
    assert with_overrides(base, color=None, width=None) is base
    changed = with_overrides(base, width=5, opacity=0.5)
    assert (changed.width, changed.opacity, changed.color) == (5, 0.5, "red")
    with pytest.raises(ValueError, match="width"):
        with_overrides(base, width=-2)
    with pytest.raises(TypeError, match="radius"):
        with_overrides(base, radius=3)


def test_color_cycle_order_wrap_and_reset() -> None:
    cycle = ColorCycle(("a", "b", "c"))
    assert [cycle.next() for _ in range(4)] == ["a", "b", "c", "a"]
    cycle.reset()
    assert cycle.next() == "a"


def test_explicit_colours_do_not_use_up_the_palette() -> None:
    cycle = ColorCycle(("a", "b"))
    assert cycle.resolve("red") == "red"
    assert cycle.resolve(None) == "a"
    assert cycle.resolve(None) == "b"


def test_palette_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="palette"):
        ColorCycle(())
    cycle = ColorCycle(("a",))
    with pytest.raises(ValueError, match="palette"):
        cycle.palette = ()


def test_fill_styles_take_overrides() -> None:
    base = FillStyle(color="red")
    changed = with_overrides(base, opacity=0.5)
    assert (changed.color, changed.opacity) == ("red", 0.5)
    with pytest.raises(TypeError, match="width"):
        with_overrides(base, width=3)  # areas have no outline
