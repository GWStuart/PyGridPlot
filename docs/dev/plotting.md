# Plotting

Everything in `core/plot/` turns a plotted object (a function, a curve,
points, a polyline, a polygon) into clipped, screen-space primitives
(`Stroke`, `Fill`, `Markers`). The entry point is
`core/plot/builders.py`, one function per kind of plot; the other modules
are its building blocks.

| Module | Role |
|---|---|
| `builders.py` | `function_stroke`, `parametric_stroke`, `polar_stroke`, `polyline_stroke`, `polygon_shapes`, `point_markers`: world data → clipped primitives. |
| `evaluation.py` | Calling user functions so that they can never raise. |
| `sampling.py` | Adaptive sampling of `y = f(x)`. |
| `curves.py` | Adaptive sampling of parametric curves (and so polar ones) on screen. |
| `geometry.py` | Normalising point input; clipping polylines (Liang–Barsky) and polygons (Sutherland–Hodgman). |
| `style.py` | `LineStyle`, `PointStyle`, `PolygonStyle`, `with_overrides`, and `ColorCycle` (the palette). |

## Robust evaluation: `evaluation.py`

User functions can do anything: work only on single numbers (`math`),
raise for part of their domain, return NaN, infinities, complex numbers,
masked arrays or values of the wrong shape. `evaluate(func, xs)` turns all
of that into a float array with NaN wherever there is no real value:

1. Call the function once on the whole array (fast for NumPy functions).
2. If that raises or returns something unusable (a scalar, the wrong
   shape, a non-numeric type), call it point by point, catching errors per
   point.
3. Complex results keep their real part where the imaginary part is
   negligible (relative `1e-12`) and become NaN elsewhere; masked values
   become NaN.

`evaluate_quietly` does the same without silencing NumPy floating-point
errors and warnings; the samplers silence them once around all their
evaluations, which is much cheaper than once per batch.

## Sampling functions: `sampling.py`

`sample_function(func, x_range, scale, y_band, config)` returns world-space
polylines that look right on screen. All tolerances are in screen pixels,
which is why it needs the axis scales.

1. **World-anchored samples.** One sample per pixel column, at `x = k ×
   step` for whole numbers `k`. A pan by whole pixels therefore reuses
   exactly the same world points, and curves move rigidly instead of
   shimmering.
2. **Adaptive refinement** (vectorised rounds, up to `max_rounds`). Only
   segments that matter are halved: those whose midpoint lies more than
   `tolerance_px` above or below the chord (*vertical* deviation, which
   catches near-vertical jumps that a perpendicular distance would miss),
   and those with an undefined end. Segments entirely above or below the
   visible band are left alone. When the evaluation budget
   (`max_samples`) runs short, the segments that change fastest go first.
3. **Telling jumps from steep curves.** A segment still rising more than
   `jump_px` after the last round gets one more midpoint test
   (`_classify_jumps`): if the rise sits entirely in one half (a step of
   `floor`) or the midpoint overshoots both ends (a pole, as in `1/x`), it
   is a discontinuity. If both halves share the rise, it is a steep but
   continuous branch and stays joined, so the sides of a pole run all the
   way off the plane.
4. **Chasing poles and edges** (`_chase`). Some features are narrower than
   the refinement can resolve: the poles of `gamma` at large negative x, or
   the edge of a domain. Such segments are bisected like a root search,
   keeping the feature inside (by sign for poles, by definedness for edges),
   until the curve has left the screen or the interval is a millionth of a
   pixel wide (a billionth for poles). So branches reach the edge of the plane, and `sqrt` starts
   exactly at the origin.
5. **Polylines.** The samples are split wherever the function is undefined
   or jumps (`_breaks`), and pieces entirely outside the band are dropped.

## Sampling curves: `curves.py`

`sample_curve(curve, t_range, box, config)` samples a function from `t` to
**screen** points, so it knows nothing about the view, the axis kinds or
how the curve was defined. Everything is measured along the curve on
screen, since a parametric curve can go in any direction, double back,
loop or spiral.

1. **Initial samples**: `initial_samples` evenly spaced values of `t`. They
   depend only on the range, so panning reuses them and curves move
   rigidly.
2. **The depth limit adapts to the zoom** (`_depth_limit`). Because the
   samples are even in `t`, not in pixels, the deeper the zoom the longer
   the first segments are on screen and the more halvings a smooth curve
   needs. The limit is the number of halvings that bring the *median*
   initial segment near the screen down to `max_segment_px`, plus
   `max_rounds`. (A fixed limit made a unit circle vanish at a scale of
   10⁹.)
3. **Refinement** (vectorised rounds). Segments far from the screen (further
   than their own length) are closed without evaluating anything. The rest
   are halved while their midpoint is more than `tolerance_px` from the
   chord (distance to the *segment*, so doubling back counts), while either
   half is longer than `max_segment_px / 2`, or while one end is undefined.
   Segments that floating point cannot split any further are treated as
   fully refined.
4. **Domain edges** (`_find_edges`). Each segment with one defined end on
   screen is bisected until the defined end moves less than 0.05 px per
   step, or misses ten times in a row, and the final defined point is
   inserted. Curves such as `r = sqrt(cos 2θ)` therefore end exactly at the
   origin.
5. **Jumps.** A segment still longer than `max_segment_px` at the depth
   limit cannot belong to a continuous curve: it would have to cross that
   many pixels in a tiny fraction of the step. It is a jump or a pole, so
   the curve is broken there. The sides of a pole have been refined up to
   it and run off the plane.

The samples are kept in a small `_Samples` structure (parameters, points,
and per segment an `open` flag and a depth); `split` inserts midpoints with
`_insert`, a lean replacement for `np.insert`, which matters in the hot
loop.

## Building primitives: `builders.py`

Each builder follows the same pattern: convert to screen space through the
view, clip a little outside the plane (`CLIP_MARGIN_PX` plus half the line
width, so round ends and antialiasing are never visibly cut; the renderer
clips exactly), and return a primitive, or `None` when nothing is visible.

- **Functions** are sampled in *transformed* coordinates: on log axes the
  sampler is given `w = T_y(f(T_x⁻¹(u)))` (`transformed_function`), sampled
  over the transformed visible range, so sampling is even on screen and the
  sampler needs no knowledge of log axes. A `domain` is transformed too (on
  a log x axis, the part at or below zero is dropped).
- **Parametric and polar curves** wrap each component robustly
  (`_component`, which also accepts plain numbers), map them through
  `View.to_screen` and hand the result to `sample_curve`. Polar curves are
  parametric curves `(r cos θ, r sin θ)`.
- **Polylines, polygons and points** transform vertex by vertex. On a log
  axis a negative coordinate becomes NaN (a break) and zero becomes
  infinite; `geometry.clamp_coordinates` clamps such values to
  `±COORDINATE_LIMIT` (10⁹ px) before clipping, so edges towards zero run off
  the plane and nothing overflows.

## Styles and the palette: `style.py`

Styles are frozen dataclasses validated in `__post_init__`, so a bad value
raises when the style is created. A drawing call combines a style with its
keyword arguments through `with_overrides(style, **kwargs)`, where `None`
means "keep the style's value", then asks the plane's `ColorCycle` for the
colour: an explicit colour is returned unchanged, `None` takes the next
palette colour. `draw_grid` (or `begin_frame`) resets the cycle every frame,
so colours follow drawing order. The marker and fill-rule vocabulary
(`MarkerShape`, `FillRule`) lives in `core/primitives.py`, because the
primitives and the rasteriser use it too.
