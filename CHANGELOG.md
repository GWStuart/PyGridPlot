# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/) once 1.0 is released.

## [Unreleased]

### Added

- **Objects**: persistent functions, curves and points, made once and
  changed through their attributes: `pgp.Function`, `pgp.ParametricCurve`,
  `pgp.PolarCurve`, `pgp.ImplicitCurve`, `pgp.Point`, `pgp.Points`,
  `pgp.Polyline`, `pgp.Polygon`, each taking the arguments of the plane
  method it mirrors and keeping them as attributes read every frame.
  `plane.add(...)` keeps them and `plane.draw(screen)` draws the grid,
  the objects and the draggable points in one call (`plane.remove`,
  `plane.objects`, `plane.clear_objects`, `plane.draw_objects` to mix with
  immediate drawing; `obj.draw(plane, screen)` for one object). Functions
  are callable and differentiable (`f(2)`, `f.derivative()`,
  `pgp.derivative(f)`); a `Point` is fixed, derived from other objects (a
  function of nothing, per coordinate or whole) or draggable
  (`draggable=True`, with `lock`, bounds, or `curve=` a function or curve
  *object*), on any number of planes; objects take text in the expression
  language too. Objects without a colour keep a stable palette slot; one
  that cannot be drawn is skipped with its reason in `obj.error` and a
  `pgp.PlotWarning` pointing at the program's own line.
  `examples/objects.py`; gallery view 54; benchmark cases.
- **Graphs**: `pgp.Graph`, a named list of objects (plus parameters and
  function definitions, `graph.Parameter` and `graph.Definition`) built on
  the objects and drawn with `graph.draw(plane, surface)` or
  `plane.add(graph)`; `graph.add("y = x^2")` returns a `pgp.Function`, and
  `graph.append(obj)` takes objects made on their own.
  Items are added from lines of maths (`graph.add("f(x) = a sin(x)")`,
  `"x^2 + y^2 = 9"`, `"P = (1, 2)"`, `"a = 2"`) or with one method per
  kind taking the plane method's arguments, and use each other's names in
  any order; circular definitions and unknown names are reported per item
  (`item.error`, `graph.errors`, a `pgp.PlotWarning` once per error) while
  the rest still draws. Draggable points are kept by the graph, in step
  with their handles on every plane, can follow another item's curve by
  name, and are saved where they are. `graph.evaluate(...)`,
  `graph.function(name)`, registered Python functions
  (`Graph(functions=...)`, `graph.register`).
- **Saving graphs as JSON**: `graph.save(path)` / `Graph.load(path)`,
  `to_json` / `from_json`, `to_dict` / `from_dict`. A versioned, readable
  format listing only non-default settings; strict reading with errors
  that say where a hand-written file is wrong (`GraphError`); views saved
  with `graph.capture_view(plane)` and restored exactly with
  `graph.apply_view(plane)` (`pgp.GraphView`).
- **The expression language**: a small, safe parser and compiler (no
  `eval`) from text to NumPy functions: implicit multiplication (`2x`,
  `a sin(x)`), `^`, Unicode operators and superscripts, 38 built-in
  functions, constants, comparisons and NaN-aware conditions (`if`, `and`,
  `or`, `not`), exact derivatives with primes (`f'(x)`), points (`P.x`),
  and error messages with a caret and suggestions (`sinx`: did you mean
  `sin(x)`?). `pgp.expression(text, ...)` compiles one into a function;
  `pgp.ExpressionError`.
- `pgp.show` also draws lines of maths and graphs:
  `pgp.show("y = x^2", graph)`.
- `View.fitted_ranges()`: the ranges `View.fit` maps back to exactly the
  same view.
- `examples/graph_editor.py`: a small expression list to type into, with
  saving and loading; gallery views 52 and 53; graph benchmark cases.

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

- The graph's item classes became the objects (`PlotItem` is now
  `pgp.Function`, `PointItem` and `DraggableItem` are now `pgp.Point`, and
  so on), and their `expression` attribute is `func`; `GraphWarning` is
  now `pgp.PlotWarning`. Invalid names raise a `ValueError` (a
  `GraphError` is still raised for a name already taken). The JSON format
  is unchanged.
- The type alias `Function` of `pygridplot.core.plot.evaluation` is now
  `FunctionLike`, as `pgp.Function` is the object.

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
