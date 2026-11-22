# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/) once 1.0 is released.

## [Unreleased]

### Added

- **Vector fields**: `Plane.vector_field(surface, field)` draws a field
  `(u, v) = field(x, y)` as arrows on a lattice that is fixed to the plane
  (panning is rigid) and keeps its density on screen: the spacing is a
  power of two per axis, and while zooming the new arrows grow and fade in
  between the others, so nothing pops. Arrows have one length (the field's
  direction) or, with `scale=`, a length in proportion to the magnitude,
  capped so that they never meet; they are tangent to the flow lines on log
  axes too. `spacing=` sets their density.
- **Slope fields**: `Plane.slope_field(surface, slope)` shows
  `dy/dx = slope(x, y)` as segments on the same lattice, tangent to the
  solution curves (vertical for an infinite slope).
- **Arrows**: `Plane.arrow(surface, start, end)` draws one arrow or many
  (one start for many ends, or the other way round), each filled as one
  shape, so translucent arrows have no darker joins. `ArrowStyle(color,
  width, opacity, head_size)`, with heads in proportion to the width by
  default.
- `AxisTransform.derivative`, and `evaluate_vectors` for robustly
  evaluating functions that return a pair. `examples/fields.py`: fields,
  a solution through a draggable point, and vectors at a point.
- **Inequalities and regions**: `Plane.inequality(surface, condition)`
  shades the region where a condition such as `x**2 + y**2 < 4` holds,
  written in Python and combined with NumPy's `&`, `|` and `~`. Regions
  can have any shape (several pieces, holes, edges at asymptotes and
  jumps); they are found on the implicit curves' anchored grids, sampled
  finely only near the boundary, with edges bisected onto the true
  boundary to within 1/8 px, so they pan rigidly and cost about as much as
  their outline. Undefined points are outside.
- **Areas between graphs**: `Plane.fill_between(surface, f, g=0)` shades
  between two graphs (or a graph and a number, `±inf` for the edges of the
  plane), sampled exactly as `plot` samples them. It covers only where both
  are defined, stops with a clean edge at jumps and asymptotes, and fills
  both sides of a crossing; `domain=` restricts it, and on a log y axis `0`
  reaches the bottom edge.
- `FillStyle(color, opacity)`: the reusable style of both.
  `examples/regions.py`: inequalities, compound conditions and areas
  between draggable graphs.
- `docs/philosophy.md`: the principles behind PyGridPlot (one way to do
  things, done well; Python first; a library, not a framework; immediate
  mode; ...), how features are judged, and the features that were tried
  and retired.
- **Reusing unchanged plots**: plotting calls whose view, style, inputs and
  function values are unchanged since the previous frame reuse their
  primitives and rasterised coverage, and the grid is reused while the view
  and theme stay the same. Exact (functions are re-checked at every point
  where they were sampled), automatic, and several times faster for idle
  frames and for frames where only some plots change. `Plane(cache=...)`,
  `plane.cache`, `plane.cache_stats`, `plane.clear_cache()`, `CacheStats`.
- `ROADMAP.md`: ideas for future features and limitations to lift.
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

- Fills of many polygons are rasterised up to twice as fast (the rings are
  collected in one pass, crossings are ordered by one sort key instead of
  `np.lexsort`, and span weights are scattered with `np.bincount`): about
  9 → 4.5 ms for a vector field's arrows, 7.6 → 4.5 ms for a region of 100
  pieces, unchanged for a single polygon. Coverage changes by at most
  `1e-6`.
- Log-axis grids are drawn about a third faster: the subdivisions of every
  whole decade on screen are worked out once per frame.

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

- A draggable point no longer keeps its plane alive: a plane that is no
  longer used is freed even while its points are still referenced.
- The derivatives of an undefined constant (NaN or infinite) inside a
  differentiated function are now undefined too, rather than 0, so
  `derivative(lambda x: np.where(x > 0, x**2, np.nan))` is undefined where
  the function is.
- With several planes in one window, moving between their draggable points
  could leave the hand cursor stuck; the planes now share one cursor.

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
