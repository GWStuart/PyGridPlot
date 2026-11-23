# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/) once 1.0 is released.

## [Unreleased]

### Changed

- **`Plane` is now `Grid`**, after the package (`grid = pgp.Grid()`; the
  module `pygridplot.plane` is `pygridplot.grid`), and names were made
  consistent before the first release:
  - `grid.draw(screen)` draws the grid and starts the frame (was
    `draw_grid`), and `grid.last_scene` is what it drew (was `last_grid`);
  - `grid.draw_controls(screen)` draws draggable points and sliders (was
    `draw_draggables`, which drew sliders too);
  - `pgp.color_map` and `grid.color_bar` (were `colormap` and `colorbar`),
    whose range of values is `value_range`, as for sliders; the slider's
    own is `Slider.value_range` (was `range`);
  - `GridTheme.show_arrows`, like the other `show_*` switches (was
    `axis_arrows`);
  - `FunctionSampling`, beside `CurveSampling` and `ImplicitSampling` (was
    `SamplingConfig`).
- Importing PyGridPlot with the original `pygame` installed instead of
  pygame-ce now raises an `ImportError` that says how to fix it, rather
  than failing later with an obscure error.
- **Points on graphs follow the cursor's x.** A draggable point on a curve
  that runs steadily left to right on screen, such as `(t, f(t))` or a
  cycloid, now goes to the point of the curve at the cursor's x, instead
  of the nearest point: it stays level with the cursor at any height,
  passes through jumps and asymptotes, and waits at domain edges. On a
  wavy graph the nearest point lagged up to 190 px behind the cursor and
  then jumped. A curve running steadily up or down follows the cursor's y;
  other curves (circles, spirals, figure-eights) still follow the nearest
  point, continuously. Placing a point from code is unchanged.
- A new `Image` primitive (pixels drawn as they are), and `Stroke.paths`
  may be one `(M, N, 2)` array of paths of one length (faster for error
  bars and stems: 500 points with both kinds of error bar, 6.9 to 6.0 ms).
- `PointStyle(radius=0)` is allowed (no marker, for error bars alone), and
  `PointStyle` has `error_width` and `error_cap`. `Markers` can carry a
  colour and a size per marker (`colors`, `radii`).
- `TextSize` has a `descent` (the font's, measured by the renderer), for
  text with letters to keep clear of; `panel.background_primitives` takes
  a rectangle.

### Fixed

- The lowest supported pygame-ce is now 2.5.7 (it was given as 2.5):
  label halos use `Font.outline`, which older versions lack.
- A slider whose step is as wide as its range, such as `(1.0, 1.001)` with
  `step=0.001`, is no longer rejected because the difference of the ends
  rounds to a hair less than the step.

### Added

- **A licence**: PyGridPlot is free software under the GNU LGPL, version 2.1
  or later (`LICENSE`), like pygame-ce.
- **Packaging for PyPI**: `pip install pygridplot` installs it. The package
  has its licence, authors, classifiers and project links; the version is
  read from `pygridplot.__version__`; the README's links become absolute
  on PyPI; and the source archive holds the tests, docs and examples.
- **Continuous integration** (`.github/workflows/ci.yml`): the tests on
  Python 3.10 to 3.15 on Linux, and on Windows and macOS, with the oldest
  supported dependencies, lint and type checks, and a build whose wheel
  and source archive are installed and tested. A release workflow
  publishes to PyPI from a GitHub release, and
  [docs/dev/releasing.md](docs/dev/releasing.md) lists the steps.
- **Colour by value**: `pgp.color_map(values, colors="viridis",
  value_range=None, log=False)` returns a colour for each value, for `points(color=...)`,
  along a built-in map (`viridis`, `plasma`, `inferno`, `magma`, the
  diverging `coolwarm`, `gray`) or any sequence of colours; NaN is
  transparent. `grid.color_bar(screen, colors, value_range, title=, log=)` draws
  its key: the map as a strip with ticks at round values, in a corner
  panel that stacks with the legend and the sliders.
- **Box plots and violin plots**: `grid.box_plot(screen, groups)` (quartiles,
  median, Tukey's whiskers and outliers) and `grid.violin_plot(screen,
  groups)` (a kernel density, Silverman's bandwidth), for one group of
  samples or several, at `start` or `positions`, upright or `horizontal`.
- **Categories on an axis**: `grid.draw(screen, x_labels=["a", "b",
  ...])` (or a mapping from positions to names, or `y_labels=`) names
  positions in place of numbers, pinned and faded like tick labels and
  thinned to every 2nd, 4th, ... name when crowded.
- **Stacked, floating and horizontal bars**: `grid.bars(...)` takes a
  `baseline=` per bar and `horizontal=True`.
- **Cumulative histograms**: `grid.histogram(..., cumulative=True)`, with
  `density=True` rising to 1.
- **Ranges that show data**: `pgp.data_range(values, margin=0.05,
  include=None, log=False)` for `Grid(x_range=...)` and `set_view`, or
  `None` with no data (which leaves an axis as it is).
- **Data plots.** `grid.points` takes a colour and a size **per point**
  (`color=` a sequence of colours or an `(N, 3)` array, `radius=` a
  sequence of sizes), for scatter plots that show a value, drawn about as
  fast as one colour; and **error bars** (`x_error=`, `y_error=`: one
  length, one per point, or below and above), in the points' colour under
  the markers, with caps (`error_width=`, `error_cap=`; `radius=0` for the
  bars alone). `grid.polyline(..., step="post" | "pre" | "mid")` draws
  **step plots**. New `grid.bars(screen, heights)` draws a **bar chart**
  from heights or `(x, top)` pairs, and `grid.histogram(screen, samples)`
  a **histogram** of raw samples: bins anchored at multiples of the width
  (so they never shift as samples arrive), an automatic 1-2-5 width,
  explicit `edges=`, `density=True` for comparing with a density, lines
  between bins that fade out when zoomed out, and bins equal on screen on
  a log x axis. Both take a new `pgp.BarStyle`, and the legend has samples
  for them and for points with error bars. `examples/data.py`.
- **Axis titles**: `grid.draw(screen, x_title="t (s)", y_title="v")`
  titles the axes at their positive ends, as in textbook diagrams: the x
  axis' title right-aligned at the right edge just above the axis, the y
  axis' at the top just right of it. They stay on screen when their axes
  leave it, pinned to the edges like the tick labels (with the x axis below
  the plane, its title rides above the pinned labels), tick labels fade
  where they meet one, and they keep clear of arrowheads. Their size is the
  theme's new `title_size` (14). `pgp.show(..., x_title=, y_title=)` takes
  them too.
- **Arrowheads on the axes**: a theme with `show_arrows=True` ends each axis
  in an arrowhead at its positive end, sized from `axis_width`; the axis
  line stops under it. Off by default.
- **Legends**: every plotting method takes a `label=`, and
  `grid.legend(screen, corner="top-right")` draws a legend of the frame's
  labelled plots, each beside a sample of the plot in its style and colour
  (a line, a filled box, a polygon, a marker, a stem or an arrow). It is
  drawn in a corner panel like the sliders' (presses on it never pan the
  plane), goes beyond the sliders' panel in a shared corner, lists an exact
  repeat once, and is drawn from the same primitives while unchanged.
  `examples/functions.py` uses it instead of its hand-made legend, and the
  legend recipe now shows `label=`.
- **Text in world space**: `grid.text(screen, "max", (x, y),
  anchor="bottom", gap=8)` writes text at a point, moving with it while
  panning and zooming; nine anchors, a pixel gap, several lines, the
  theme's label font, colour (text never takes palette colours) and halo,
  and a reusable `pgp.TextStyle(color=, size=, opacity=, halo=)`.
  `examples/annotations.py` shows titles, arrowheads, legends and text
  together.
- **Sliders**: `Grid.slider(label, (low, high), value=, step=, log=)`
  creates a slider, drawn by `draw_controls` in a panel in a corner of
  the plane (`corner=`) with its label and value (`a = 1.5`), which stays
  put while the plane is panned and zoomed. Press the track to move the
  handle there, drag the handle, or turn the mouse wheel over it to move it
  a step; presses and the wheel on a panel never pan or zoom the plane, and
  sliders are above draggable points. Values are exact: steps are counted
  in decimal from `low` (a step of 0.1 gives 0.3), `step=1` makes an
  integer slider whose `value` is an `int` (typed `Slider[int]`), a dragged
  value without a step is rounded to a pixel of the track, and `log=True`
  makes the track geometric. `value` can be set from code, `moved` tells
  whether it changed since the previous frame, and a slider not drawn in a
  frame cannot be dragged. Unchanged panels are drawn from the same
  primitives, so their coverage is reused. `examples/sliders.py`; the
  examples that made sliders from draggable points (`derivatives.py`,
  `integrals.py`, `series.py`, `sequences.py`) now use `grid.slider`.
- **Sequences**: `Grid.sequence(surface, terms, start=0)` draws the terms
  `aₙ` of a sequence as stems from the x axis with a marker each (or
  markers alone with `width=0`, stems alone with `radius=0`). `terms` is a
  list or array (a finite sequence from index `start`, drawn in full) or a
  function of `n` (an infinite one, drawn at every visible integer, with
  Python ints for scalar-only functions such as `math.factorial`). Zoomed
  far out, terms sharing a pixel column share one stem, and an infinite
  sequence's terms closer than a quarter pixel are thinned along powers of
  three, fading smoothly, so nothing pops, panning stays rigid, alternating
  signs stay visible and the work stays bounded. `baseline=` starts the
  stems at a limit instead of the x axis. `SequenceStyle(color,
  width, radius, marker, opacity)`. `FourierSeries.amplitudes` gives a
  series' spectrum, for `grid.sequence`. `examples/sequences.py`, and
  `examples/series.py` now draws its spectrum with `sequence`.
- **Integrals**: `pgp.integral(f, a, b)` returns a definite integral,
  computed by adaptive Gauss-Kronrod quadrature (21 points, QUADPACK's
  error estimate) vectorised over all intervals at once: about 10
  significant digits for smooth functions, jumps, kinks and integrable
  singularities (`1/sqrt(x)`, `log x` at 0), with infinite limits mapped
  onto finite ones. NaN rather than a wrong number when the integral
  diverges or the function is undefined on part of the interval; a single
  undefined point (`sin(x) / x` at 0) makes no difference. Bounded work, so
  no integral can freeze a frame.
- **Antiderivatives**: `pgp.antiderivative(f, a=0)` returns
  `F(x) = ∫ₐˣ f(t) dt` as a function to plot or evaluate, computed by
  cumulative quadrature over the points it is evaluated at (so plotting
  one costs a few integrals, not one per point). `a` may be `-np.inf` (a
  cumulative distribution), and `F` is undefined beyond a divergence (the
  antiderivative of `1/x` from 1 is `log x` for `x > 0` only).
  `pgp.derivative(F)` is `f` exactly. The plot cache checks an
  antiderivative through its integrand, as its last digits depend on the
  other points it is evaluated with, so unchanged antiderivatives are
  reused like other plots.
- **Taylor polynomials**: `pgp.taylor(f, a, n)` returns the Taylor
  polynomial of degree `n` (up to 40) about `a`, from the exact
  derivatives, with its `coefficients`.
- **Fourier series**: `pgp.fourier(f, n, interval=(-π, π))` returns the
  partial sum with `n` harmonics (up to 1000) of the Fourier series of a
  function on an interval, repeated with the interval's length as period,
  with its `cosines` and `sines`. The coefficients come from one adaptive
  quadrature run, accurate at the jumps and kinks of square and sawtooth
  waves.
- Taylor polynomials and Fourier sums accept jets, so they differentiate
  exactly. `examples/integrals.py` (the area between two draggable points
  with its antiderivative, Riemann sums with a slider, improper integrals)
  and `examples/series.py` (Fourier series with a slider, and their
  spectrum in a second plane); `examples/derivatives.py` uses
  `pgp.taylor`, and `examples/regions.py` measures its area with
  `pgp.integral`. The API reference's "Derivatives" section is now
  "Calculus".
- **Vector fields**: `Grid.vector_field(surface, field)` draws a field
  `(u, v) = field(x, y)` as arrows on a lattice that is fixed to the plane
  (panning is rigid) and keeps its density on screen: the spacing is a
  power of two per axis, and while zooming the new arrows grow and fade in
  between the others, so nothing pops. Arrows have one length (the field's
  direction) or, with `scale=`, a length in proportion to the magnitude,
  capped so that they never meet; they are tangent to the flow lines on log
  axes too. `spacing=` sets their density.
- **Slope fields**: `Grid.slope_field(surface, slope)` shows
  `dy/dx = slope(x, y)` as segments on the same lattice, tangent to the
  solution curves (vertical for an infinite slope).
- **Arrows**: `Grid.arrow(surface, start, end)` draws one arrow or many
  (one start for many ends, or the other way round), each filled as one
  shape, so translucent arrows have no darker joins. `ArrowStyle(color,
  width, opacity, head_size)`, with heads in proportion to the width by
  default.
- `AxisTransform.derivative`, and `evaluate_vectors` for robustly
  evaluating functions that return a pair. `examples/fields.py`: fields,
  a solution through a draggable point, and vectors at a point.
- **Inequalities and regions**: `Grid.inequality(surface, condition)`
  shades the region where a condition such as `x**2 + y**2 < 4` holds,
  written in Python and combined with NumPy's `&`, `|` and `~`. Regions
  can have any shape (several pieces, holes, edges at asymptotes and
  jumps); they are found on the implicit curves' anchored grids, sampled
  finely only near the boundary, with edges bisected onto the true
  boundary to within 1/8 px, so they pan rigidly and cost about as much as
  their outline. Undefined points are outside.
- **Areas between graphs**: `Grid.fill_between(surface, f, g=0)` shades
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
  frames and for frames where only some plots change. `Grid(cache=...)`,
  `grid.cache`, `grid.cache_stats`, `grid.clear_cache()`, `CacheStats`.
- `ROADMAP.md`: ideas for future features and limitations to lift.
- **Derivatives**: `pgp.derivative(f, n)` returns the `n`-th derivative of a
  function of one variable (up to order 40), to plot or evaluate;
  `pgp.derivatives(f, x, n)` returns every order at once. Exact for NumPy
  functions, by Taylor-mode automatic differentiation built in (no new
  dependency), including next to jumps, kinks and poles and for functions
  with Python `if` statements; numerical, by Richardson-extrapolated finite
  differences, for `math` functions. `examples/derivatives.py`: tangent
  lines, Taylor polynomials and osculating circles at a draggable point.
- **Draggable points**: `Grid.draggable(...)` creates a `DraggablePoint`
  the user drags with the mouse; read its `x`, `y` (and `t`) every frame
  and draw the points with `Grid.draw_controls`. Points can be free,
  locked to one axis (`lock="x"` / `"y"`, for sliders), or kept on a
  parametric curve (`curve=(x, y, t_range)`), which they follow through
  crossings without jumping between branches. Bounds, a hover ring and hand
  cursor, coordinates rounded to a pixel, a `moved` flag, and new
  `InteractionConfig` fields (`drag_button`, `grab_margin_px`,
  `hover_cursor`); `examples/draggable.py`.
- **Implicit curves**: `Grid.implicit(surface, F, level=0)` draws any
  curve `F(x, y) = level`, with loops, crossings and separate pieces, found
  by marching squares (contourpy) on a grid that is fine only near the
  curve. Crossings are checked by bisection, so poles and jumps are never
  drawn as lines, and refined onto the curve. A sequence of levels draws
  contour lines with a single evaluation of `F`. Tuned with
  `ImplicitSampling`; works on log axes; `examples/implicit.py`.
- **Parametric and polar curves**: `Grid.parametric(surface, x, y,
  t_range)` and `Grid.polar(surface, r, theta_range)`, sampled adaptively
  on screen, with exact domain edges and broken at jumps and poles; tuned
  with `CurveSampling`.
- **Logarithmic axes**: `Grid(log_x=True, log_y=True)` and the assignable
  `grid.log_x` / `grid.log_y`. The grid shows thinned decades, log paper,
  then finer subdivisions as you zoom, all cross-faded; labels choose their
  notation per value.
- **View from ranges**: `Grid(x_range=..., y_range=...)`, fitted once the
  plane knows its size; equal aspect is decided automatically.
- `show()` accepts the same view options as `Grid`.
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

- `Grid(interactive=False)` now switches off only panning and zooming;
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
