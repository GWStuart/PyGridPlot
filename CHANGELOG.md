# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/) once 1.0 is released.

## [Unreleased]

### Added

- **Derivatives**: `pgp.derivative(f, n)` returns the `n`-th derivative of a
  function of one variable (up to order 40), to plot or evaluate;
  `pgp.derivatives(f, x, n)` returns every order at once. Exact for NumPy
  functions, by Taylor-mode automatic differentiation built in (no new
  dependency), including next to jumps, kinks and poles and for functions
  with Python `if` statements; numerical, by Richardson-extrapolated finite
  differences, for `math` functions. `examples/derivatives.py`: tangent
  lines, Taylor polynomials and osculating circles at a draggable point.
- **Draggable points**: `Plane.draggable(...)` creates a `DraggablePoint`
  the user drags with the mouse; read its `x`, `y` (and `t`) every frame
  and draw the points with `Plane.draw_draggables`. Points can be free,
  locked to one axis (`lock="x"` / `"y"`, for sliders), or kept on a
  parametric curve (`curve=(x, y, t_range)`), which they follow through
  crossings without jumping between branches. Bounds, a hover ring and hand
  cursor, coordinates rounded to a pixel, a `moved` flag, and new
  `InteractionConfig` fields (`drag_button`, `grab_margin_px`,
  `hover_cursor`); `examples/draggable.py`.
- **Implicit curves**: `Plane.implicit(surface, F, level=0)` draws any
  curve `F(x, y) = level`, with loops, crossings and separate pieces, found
  by marching squares (contourpy) on a grid that is fine only near the
  curve. Crossings are checked by bisection, so poles and jumps are never
  drawn as lines, and refined onto the curve. A sequence of levels draws
  contour lines with a single evaluation of `F`. Tuned with
  `ImplicitSampling`; works on log axes; `examples/implicit.py`.
- **Parametric and polar curves**: `Plane.parametric(surface, x, y,
  t_range)` and `Plane.polar(surface, r, theta_range)`, sampled adaptively
  on screen, with exact domain edges and broken at jumps and poles; tuned
  with `CurveSampling`.
- **Logarithmic axes**: `Plane(log_x=True, log_y=True)` and the assignable
  `plane.log_x` / `plane.log_y`. The grid shows thinned decades, log paper,
  then finer subdivisions as you zoom, all cross-faded; labels choose their
  notation per value.
- **View from ranges**: `Plane(x_range=..., y_range=...)`, fitted once the
  plane knows its size; equal aspect is decided automatically.
- `show()` accepts the same view options as `Plane`.
- Documentation split into a user guide and a developer guide with topic
  pages, recipes, and an API reference generated from the docstrings
  (`scripts/build_api_docs.py`), with tests that run every documented
  example and check every link.
- `scripts/benchmark.py` for per-frame timings; `examples/curves.py` and
  `examples/log_axes.py`; a headless smoke test for every example.
- `py.typed`, so type checkers use the package's annotations.

### Changed

- `Plane(interactive=False)` now switches off only panning and zooming;
  draggable points still work.
- PyGridPlot now depends on contourpy (small, compiled, NumPy only).

- Strokes are rasterised 1.5 to 3 times faster (two-level candidate
  generation, duplicate merging with `np.maximum.at`), and blended up to
  six times faster on 32-bit surfaces; large fills are about 1.4 times
  faster.
- `core` is organised into foundation modules plus the `core.grid` and
  `core.plot` subpackages, with the layering enforced by tests.

### Fixed

- Labels fading near each other (on log axes, and pinned labels in the
  corners) no longer change opacity in steps of 1/8 as they cross pixel
  boundaries while zooming: the fades measure exact distances.

## [0.1.0.dev0] - 2026-11-20

### Added

- The interactive plane: adaptive 1-2-5 grid with cross-fades, exact
  labels, push-and-pin label layout, anchored dragging and eased zooming.
- Plotting of functions, points, polylines and polygons, with robust
  evaluation, adaptive sampling and antialiased rendering.
- Light and dark themes, styles and the colour palette, `show()`.
