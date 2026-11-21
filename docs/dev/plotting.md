# Plotting

Everything in `core/plot/` turns a plotted object (a function, a
parametric, polar or implicit curve, points, a polyline, a polygon) into
clipped, screen-space primitives
(`Stroke`, `Fill`, `Markers`). The entry point is
`core/plot/builders.py`, one function per kind of plot; the other modules
are its building blocks.

| Module | Role |
|---|---|
| `builders.py` | `function_stroke`, `parametric_stroke`, `polar_stroke`, `implicit_stroke`, `polyline_stroke`, `polygon_shapes`, `point_markers`: world data → clipped primitives. |
| `evaluation.py` | Calling user functions so that they can never raise. |
| `sampling.py` | Adaptive sampling of `y = f(x)`. |
| `curves.py` | Adaptive sampling of parametric curves (and so polar ones) on screen. |
| `jets.py` | Taylor arithmetic: the `Jet` number type that carries derivatives through NumPy code. |
| `derivatives.py` | `derivative` and `derivatives`: exact via jets, point by point, or numerical. |
| `implicit.py` | Finding implicit curves `F(x, y) = c` on a grid: marching squares (contourpy), checked and refined. |
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
evaluations, which is much cheaper than once per batch. `evaluate_xy` and
`evaluate_xy_quietly` are the same for functions of two variables (implicit
curves); both share `_evaluate_arrays`, which tries the vectorised call and
falls back to calling the function point by point.

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

## Finding implicit curves: `implicit.py`

An implicit curve `F(x, y) = c` has no parameter to walk along, so it has to
be found from samples of `F`. The method is **marching squares** on a grid
that is fine only near the curve, with every crossing it finds checked and
refined. `sample_implicit(func, u_range, w_range, step, config, levels)`
works on `F` as a function of *transformed* coordinates, so like
`sample_curve` it knows nothing of the view or the kind of axes.

### Why marching squares, and why contourpy

The alternatives considered were tracing the curve from seed points (fast
and smooth, but finding every component, including small loops, needs a
search of the plane anyway, and tracing through crossings and singular
points is fragile); an adaptive quadtree refined where `F` changes sign
(the same search as here, but subdividing irregularly, so panning does not
reuse samples, and marching on an irregular mesh needs care at cell size
changes); and interval arithmetic (guaranteed, but it needs functions
written for it, not arbitrary Python). Marching squares on an anchored
regular grid handles any topology for free, is fully vectorised, and keeps
panning rigid. [contourpy](https://contourpy.readthedocs.io/) (the
contouring library behind matplotlib, small and compiled, with only NumPy
as a dependency) does the marching and links the crossings into lines,
correctly and fast; this module decides where to sample and checks what it
finds. It is the only dependency of `core` besides NumPy (see
[Architecture](architecture.md#layers)).

### The steps

1. **The lattice.** Fine nodes are multiples of `step` (normally
   `resolution_px` pixels) in transformed coordinates; coarse nodes are
   multiples of `coarse_factor` fine steps. Both are anchored to the world,
   not the screen, so panning by whole pixels evaluates exactly the same
   points and the curve moves rigidly (a property test checks this).
2. **The coarse grid** (8 px by default) covers the region plus one cell
   all round. If it alone would take more than a quarter of the budget (a
   huge window, or a very fine setting), its cells are doubled until it
   fits; powers of two keep the lattices nested.
3. **Candidate cells** (`candidate_cells`): a coarse cell can contain the
   curve if `F - c` changes sign at its corners (or is zero at one), if
   some corners are defined and some are not (a domain edge), or if the
   smallest `|F - c|` at its corners is no more than the largest variation
   across a *non-crossing* cell in its 3 × 3 neighbourhood. The last test
   catches loops smaller than a cell and curves that dip in and out between
   corners; crossing cells are left out of the variation because next to a
   pole they would make every neighbour a candidate. Candidates are widened
   by one cell. With several levels, the candidates are the union over all
   of them.
4. **Cropping**: everything after works on the candidates' bounding box,
   because contourpy's cost grows with the size of the array it is given.
5. **The fine grid** (`_Grid.fine_values`) is evaluated only on the nodes of
   candidate cells. The others take the value of the nearest coarse node:
   a non-candidate cell's corners all have the same sign, so these stand-in
   values cannot create a contour, and unlike a mask they leave contourpy a
   plain array, which it contours several times faster. Only genuinely
   undefined nodes are masked.
6. **Contouring** (`_contour`): one `contour_generator` with
   `corner_mask=False` (so every vertex lies on a grid edge, never across a
   diagonal) and `LineType.ChunkCombinedOffset` (all lines as one array plus
   offsets, so nothing is split in Python), asked for each level in turn.
7. **Checking crossings** (`_check_crossings`, `_Brackets`). A sign change
   is not always a zero: `F` changes sign across a pole (`y - tan x`) and
   across a jump (`y - floor x`), and marching squares would draw a false
   line there. Each vertex lies on a grid edge whose ends have opposite
   signs, so that edge is bisected `refine_steps` times, for all vertices
   at once. Near a root `max(|F - c|)` at the bracket's ends shrinks (by
   about 2 per step for a smooth root); at a jump it stays the same and at a
   pole it grows. A vertex is kept if it shrank below `_ROOT_SHRINK` (0.6)
   of the original, and moved onto the root interpolated in the final
   bracket; otherwise it becomes NaN, which breaks its line.
   - Vertices still undecided get up to `_EXTRA_STEPS` more steps. On an
     edge nearly tangent to the curve, the root lies close to one end and
     `F` first moves *away* from zero, so that end, and so the maximum,
     stays unchanged until the bracket reaches it; with only the regular
     steps, between one small closed loop in seventy and one in twenty
     (depending on its size) was drawn broken.
   - A vertex on a node where `F` equals the level exactly is kept only if
     `F - c` takes both signs among the node's eight neighbours
     (`_changes_sign_around`). Otherwise `F` only touches the level there:
     without this test `(x - y)²` would draw a dotted line along `x = y`
     whenever the nodes happened to lie on it, and nothing otherwise.
8. **Lines** (`_runs`): the vertices are cut at line starts and at NaN
   vertices, on the whole array at once, and runs of at least two vertices
   are returned, in grid coordinates with the grid's origin and step
   (`ImplicitContours`).

### The budget

`max_samples` bounds the evaluations of `F` per call, which is what keeps
slow (scalar-only) functions usable. The coarse grid takes at most a
quarter. `_subdivision` then picks the largest power of two, up to
`coarse_factor`, whose cost fits what is left: the new fine nodes, counted
exactly from the candidate cells' edges and interiors, plus a reserve for
checking crossings (`_VERTICES_PER_CELL` vertices per cell per fine step,
measured at 0.06 to 0.43). The dense array is also limited to
`_DENSE_FACTOR × max_samples` nodes. If the vertices turn out to need more
than is left, fewer bisection steps are taken, and the extra steps only
spend what remains. A test counts every evaluation.

### Mapping to the screen

`implicit_stroke` wraps the user's function as `F(T_x⁻¹(u), T_y⁻¹(w))`
(robustly, with `evaluate_xy_quietly`), samples the visible transformed
range plus the clip margin with a step of `resolution_px / scale`, and maps
grid coordinates to pixels from the grid origin's pixel, computed precisely
with `View.multiples_to_screen` (relative to the centre, so a deep zoom far
from the origin loses no precision), plus whole steps of `step × scale`
pixels each. The paths are clipped like every other stroke.

### Limits

`F` must change sign across the curve: a double root (`F = g²`) is not
found, and is documented for users. Features smaller than the fine grid can
be missed (loops are found reliably from about 1.5 px in radius). A curve
stops within a fine cell of where `F` becomes undefined; bisecting the
domain edge, as the curve sampler does, would be a possible improvement.

## Derivatives: `jets.py` and `derivatives.py`

User functions are arbitrary Python, so symbolic differentiation is out,
and the autodiff libraries either need functions written with their own
NumPy (autograd) or are far too large (JAX). Finite differences work on
anything but fail exactly where this library cares most: next to a jump
they invent a spike (`floor` gets a "derivative" of thousands), at a kink
they smear, at a pole they can get the sign wrong, and their error grows
quickly with the order (about `1e-10` for a first derivative, `1e-6` by the
fourth, even with Richardson extrapolation and the best step).

So derivatives use **Taylor-mode forward automatic differentiation**,
built in (`jets.py`, NumPy only):

- A `Jet` holds, for every point at once, the Taylor coefficients
  `c[k] = g⁽ᵏ⁾(x) / k!` of whatever has been computed so far, shape
  `(n + 1, *shape)`. `Jet.variable(x, n)` is `x` itself (`c = [x, 1, 0, ...]`).
- NumPy hands every ufunc call on a jet to `Jet.__array_ufunc__` (and the
  operators go through the ufuncs), so a user's function runs unchanged.
  Each ufunc has a rule on coefficient arrays (`_RAW_RULES`), costing
  `O(n²)` per point: Cauchy products and their inverse, the recurrences
  that solve each function's differential equation (`y' = y u'` for `exp`,
  `u y' = p y u'` for powers, coupled `sin`/`cos`, `y' = (1 ± y²) u'` for
  `tan` and `tanh`), and, for inverse functions, integrating their
  derivative computed with jets one order lower (`_from_derivative`).
- Whole powers use repeated squaring, exact even where the base is 0 (the
  power recurrence divides by the base). Piecewise functions (`abs`,
  `floor`, `maximum`, `np.where`, `np.clip`, `np.select`, `np.piecewise`
  via `__array_function__`) take the derivatives of the piece that applies
  at each point, which is why derivatives jump cleanly.
- Comparisons and tests act on the values; a jet of one point converts to
  `bool`, so Python `if` statements work point by point. A jet refuses to
  become a `float`, so `math` functions raise `TypeError` cleanly, and so
  does NumPy for any ufunc or function without a rule.
- Where a value is undefined (NaN or infinite), all its derivatives are set
  to NaN (`_wrap`): a recurrence would otherwise happily produce the
  derivative of `log` at -1.

`derivatives.taylor_derivatives` picks the method, falling through in turn:

1. the vectorised call `func(Jet.variable(xs, n))`;
2. if that fails (a Python `if` on an array, say), single-point jets: up to
   three probe points first, so a function that can never take a jet costs
   only three failed calls, then every point, with NaN where the function
   itself fails;
3. otherwise numerically: central differences of order `k` with step
   `eps^(1/(k+4)) · max(1, |x|)`, Richardson-extrapolated over two steps,
   with all shifted points evaluated robustly in one call.

A result that is a plain number or numeric array (a constant function) is
a jet with zero derivatives; anything else (an object array, a jet of the
wrong order) means the method failed. `Derivative` objects are ordinary
callables, so the samplers, builders and the robust evaluation need no
changes; `derivative(derivative(f, a), b)` collapses to
`derivative(f, a + b)`. A derivative called with a jet (a derivative inside
a function being differentiated) raises `TypeError`, which sends that case
to the numerical path.

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
- **Implicit curves** are found on a grid in transformed coordinates and
  mapped to the screen precisely; see
  [Mapping to the screen](#mapping-to-the-screen).
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
