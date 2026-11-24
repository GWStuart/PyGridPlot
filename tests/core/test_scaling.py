"""Tests for scaling primitives onto bigger or smaller pictures (export)."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.primitives import (
    Fill,
    FontSpec,
    Image,
    LabelText,
    LineBatch,
    Markers,
    Orientation,
    PaintLayer,
    Stroke,
    TextItem,
    TextRun,
)
from pygridplot.core.scaling import (
    MAX_SIDE_PX,
    Scaling,
    check_scale,
    scale_fill,
    scale_image,
    scale_line_batch,
    scale_markers,
    scale_stroke,
    scale_text,
)
from tests.conftest import FakeMeasurer


def test_pixel_edges_scale_and_the_plane_moves_to_the_corner() -> None:
    double = Scaling(2.0, left=30, top=20)
    assert double.x(29.5) == -0.5  # the plane's left edge: the picture's
    assert double.y(19.5) == -0.5
    assert double.x(30.0) == 0.5  # pixel 30's centre: between pixels 0 and 1
    assert double.x(31.5) == 3.5  # 1.5 px in: 3 px in
    same = Scaling(1.0, left=30, top=20)
    assert same.x(123.25) == 93.25  # at scale 1, only moved
    points = np.array([[[30.0, 20.0], [31.0, 21.0]]])
    np.testing.assert_allclose(double.points(points), [[[0.5, 0.5], [2.5, 2.5]]])


def test_picture_sizes_are_whole_and_limited() -> None:
    assert Scaling(2.0).size(400, 300) == (800, 600)
    assert Scaling(0.5).size(401, 301) == (200, 150)
    assert Scaling(1e-6).size(400, 300) == (1, 1)
    with pytest.raises(ValueError, match="too large"):
        Scaling(100.0).size(400, 300)
    assert max(Scaling(MAX_SIDE_PX / 400).size(400, 300)) == MAX_SIDE_PX


def test_scales_are_validated() -> None:
    assert check_scale(2) == 2.0
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale"):
            check_scale(bad)
    for bad in ("2", None, True):
        with pytest.raises(TypeError, match="scale"):
            check_scale(bad)


def test_shapes_scale_their_positions_and_sizes() -> None:
    k = Scaling(3.0)
    stroke = scale_stroke(Stroke((np.array([[0.0, 0.0], [10.0, 0.0]]),), "red", 2.0), k)
    assert stroke.width == 6.0
    np.testing.assert_allclose(stroke.paths[0], [[1.0, 1.0], [31.0, 1.0]])
    batched = scale_stroke(Stroke(np.zeros((4, 2, 2)), "red", 1.5), k)
    assert isinstance(batched.paths, np.ndarray)
    assert batched.paths.shape == (4, 2, 2)
    fill = scale_fill(Fill((np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),), "b"), k)
    np.testing.assert_allclose(fill.rings[0][1], [7.0, 1.0])
    markers = scale_markers(
        Markers(
            np.array([[1.0, 1.0]]),
            "circle",
            4.0,
            "g",
            edge_width=1.5,
            radii=np.array([2.0]),
        ),
        k,
    )
    assert (markers.radius, markers.edge_width) == (12.0, 4.5)
    assert markers.radii is not None
    assert markers.radii[0] == 6.0
    np.testing.assert_allclose(markers.positions, [[4.0, 4.0]])


def test_grid_lines_stay_whole_pixels_wide_and_crisp() -> None:
    batch = LineBatch(
        Orientation.VERTICAL,
        np.array([10.0, 20.0]),
        (-0.5, 99.5),
        (PaintLayer("gray"),),
        width=1,
    )
    double = scale_line_batch(batch, Scaling(2.0))
    assert double.width == 2
    np.testing.assert_allclose(double.positions, [20.5, 40.5])  # pixels 20-21, 40-41
    assert double.span == (-0.5, 199.5)
    half = scale_line_batch(batch, Scaling(0.5))
    assert half.width == 1  # never thinner than a pixel
    flat = LineBatch(
        Orientation.HORIZONTAL, np.array([5.0]), (0.0, 10.0), (PaintLayer("k"),), 2
    )
    assert scale_line_batch(flat, Scaling(1.5, left=0, top=5)).positions[0] == 0.25


def test_images_are_resampled_to_their_new_size() -> None:
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    image = scale_image(Image(pixels, left=10, top=4), Scaling(2.0, left=10, top=0))
    assert image.pixels.shape == (4, 6, 4)
    assert (image.left, image.top) == (0, 8)
    np.testing.assert_array_equal(image.pixels[::2, ::2], pixels)


def test_text_keeps_its_centre_in_a_font_scaled_with_it() -> None:
    measurer = FakeMeasurer()
    label = TextItem(
        LabelText((TextRun("12"),)),
        left=100,
        top=50,
        font=FontSpec(None, 12),
        color="black",
        halo_color="white",
        halo_width=2,
    )
    same = scale_text(label, Scaling(1.0), measurer, measurer)
    assert (same.left, same.top, same.font) == (100, 50, label.font)
    double = scale_text(label, Scaling(2.0), measurer, measurer)
    assert double.font == FontSpec(None, 24)
    assert double.halo_width == 4
    old = measurer.measure_text(label.text, label.font)
    new = measurer.measure_text(label.text, double.font)
    centre = 2 * (100 - 0.5 + old.width / 2) + 0.5
    assert abs(double.left - 0.5 + new.width / 2 - centre) <= 0.5
    thin = scale_text(label, Scaling(0.1), measurer, measurer)
    assert (thin.font.size, thin.halo_width) == (1, 1)  # never 0
