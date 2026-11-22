# Plotting

Everything in `core/plot/` turns a plotted object (a function, a
parametric, polar or implicit curve, an area between graphs, a region
where an inequality holds, a vector or slope field, points, a polyline, a
polygon, arrows) into clipped, screen-space primitives
(`Stroke`, `Fill`, `Markers`). The entry point is
`core/plot/builders.py`, one function per kind of plot; the other modules
are its building blocks.

| Module | Role |
|---|---|
| `builders.py` | `function_stroke`, `parametric_stroke`, `polar_stroke`, `implicit_stroke`, `area_fill`, `region_fill`, `vector_field_fills`, `slope_field_strokes`, `arrow_fill`, `polyline_stroke`, `polygon_shapes`, `point_markers`: world data → clipped primitives. |
| `evaluation.py` | Calling user functions so that they can never raise. |
| `sampling.py` | Adaptive sampling of `y = f(x)`. |
| `curves.py` | Adaptive sampling of parametric curves (and so polar ones) on screen. |
| `jets.py` | Taylor arithmetic: the `Jet` number type that carries derivatives through NumPy code. |
| `derivatives.py` | `derivative` and `derivatives`: exact via jets, point by point, or numerical. |
| `quadrature.py` | Adaptive Gauss-Kronrod quadrature of many integrals (pieces) at once, vectorised per round. |
| `integrals.py` | `integral` and `antiderivative`: definite integrals (infinite limits mapped to finite ones) and cumulative quadrature. |
| `series.py` | `taylor` and `fourier`: Taylor polynomials from exact derivatives, Fourier partial sums from one quadrature run. |
| `memo.py` | `PlotCache`: reusing a plotting call's result between frames, exactly. |
| `lattice.py` | The anchored coarse and fine grids shared by implicit curves and regions, and their evaluation budget. |
| `implicit.py` | Finding implicit curves `F(x, y) = c` on a grid: marching squares (contourpy), checked and refined. |
| `regions.py` | Finding regions where a condition holds, on the same grids: filled contours (contourpy), refined by bisection. |
| `areas.py` | The area between two sampled graphs, as rings. |
| `fields.py` | The anchored, fading lattice of vector and slope fields, and their directions on screen. |
| `arrows.py` | Arrows as polygons (shaft and head as one ring), for many arrows at once. |
| `geometry.py` | Normalising point input; clipping polylines (Liang–Barsky) and polygons (Sutherland–Hodgman). |
| `style.py` | `LineStyle`, `PointStyle`, `PolygonStyle`, `FillStyle`, `ArrowStyle`, `with_overrides`, and `ColorCycle` (the palette). |

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

`evaluate_vectors` (and `evaluate_vectors_quietly`) does the same for vector
fields, which return a pair `(u, v)`: a tuple or list of two components,
or an array of shape `(2, N)`. A component that is a plain number (as in
`(1, y)`) is spread over the points; if both are, or the result is not a
pair at all, the field is called point by point, where each result is
unpacked as `u, v`. The result has shape `(N, 2)`, with NaN in the
components that are undefined (and in both where the field raised).

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
`sample_curve` it knows nothing of the view or the kind of axes. The grids
themselves (steps 1, 2, 4 and 5 below, and the budget) are in
`lattice.py`, shared with [regions](#finding-regions-regionspy).

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
2. **The coarse grid** (8 px by default, `lattice.coarse_lattice`) covers
   the region plus one cell all round. If it alone would take more than a quarter of the budget (a
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
4. **Cropping** (`lattice.crop`): everything after works on the
   candidates' bounding box, because contourpy's cost grows with the size
   of the array it is given.
5. **The fine grid** (`lattice.Grid.fine_values`) is evaluated only on the nodes of
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
quarter. `lattice.subdivision` then picks the largest power of two, up to
`coarse_factor`, whose cost fits what is left: the new fine nodes, counted
exactly from the candidate cells' edges and interiors, plus a reserve for
checking crossings (`VERTICES_PER_CELL` vertices per cell per fine step,
measured at 0.06 to 0.43). The dense array is also limited to
`DENSE_FACTOR × max_samples` nodes. If the vertices turn out to need more
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

## Finding regions: `regions.py`

A region where a condition holds (`x² + y² < 4`, `y < tan x`, several
combined with `&`, `|` and `~`) is found on the same anchored grids as an
implicit curve, by `sample_region(func, u_range, w_range, step, config)`,
which again works in transformed coordinates.

The obvious approach would be to sample `F` and fill where `F < 0`, reusing
the implicit sampler. Taking the condition as a *predicate* instead is
both more general and simpler: users write any Python condition, combined
freely, and the sampler sees only 1 (inside) or 0 (outside, including
undefined). There are no signs to check, so poles and jumps need no
special care: where `y < tan x` changes from everything to nothing at an
asymptote, the region really does end there.

1. **The coarse grid** (`lattice.coarse_lattice`) is evaluated as for
   implicit curves. Nothing inside: no region. Inside everywhere: one
   rectangle covering the grid.
2. **Candidates** (`boundary_cells`) are the coarse cells whose corners
   disagree, widened by one cell. There is no near-miss test, as a
   predicate has no magnitude: islands and holes smaller than a coarse
   cell can be missed (documented for users).
3. **Cropping and the fine grid**, exactly as for implicit curves: the
   candidates' bounding box, fine values only on candidate cells, stand-in
   coarse values elsewhere (right for a predicate too: a non-candidate cell
   is inside or outside as a whole).
4. **Filled contours** (`_filled`): contourpy's `filled(0.5, 1.5)` on the
   grid of 0s and 1s, with `FillType.OuterOffset`, gives the box's region as
   outer rings and holes. Every vertex lies halfway along a grid edge
   between an inside and an outside node, or on a node of the box's border.
5. **Refining** (`_refine`): each edge vertex is moved onto the boundary
   by bisecting the predicate along its edge, `refine_steps` times, for all
   vertices at once (the bracket is 1/8 of a fine step after the default 3
   steps, so the edge is within 1/16 of a step, or 1/8 px). The steps are
   cut if they would overrun the budget. Without refinement, the edges
   would be a staircase of half-cells, visibly jagged once antialiased.
6. **The frame** (`_frame`): around the box, no cell is a candidate, so
   each of the four strips (below, above, left and right of the box, the
   first two full width) is inside or outside as a whole, decided by one
   coarse node. Inside strips are added as rectangles. They meet the box's
   rings along shared edges, which cancel exactly under the even-odd rule,
   as the rasteriser accumulates crossings of all rings together (a test
   checks that a large region has no seams).

The result (`RegionRings`) is rings in fine grid coordinates with the
grid's origin and step, mapped to the screen by `region_fill` exactly as
implicit curves are ([Mapping to the screen](#mapping-to-the-screen)), and
filled with the even-odd rule. The rings reach a little beyond the plane
(the clip margin, and the grid's last cell) but never far, so they need no
clipping: the rasteriser clips to its box.

The cost is that of the boundary, not of the area: about 1 to 2 ms for a
disc or a half-plane on an 800 × 600 plane, 13 ms for a checkerboard of 150
pieces. Most of it is contourpy and building the fine grid.

## Areas between graphs: `areas.py`

`area_fill` samples both sides with the function sampler, in transformed
coordinates exactly as `function_stroke` does, so the area's edges are
exactly the curves `plot` draws. It passes `keep_offscreen=True`, because
a piece of a graph entirely above or below the plane draws nothing as a
curve but still bounds an area that may fill the screen. A constant side
is one horizontal piece, its height kept within a screen height of the
band (so infinities and, on a log y axis, values `<= 0` lie just beyond the
edge).

`rings_between(first, second)` then sweeps the two sorted lists of pieces
together. Wherever a piece of each covers the same x interval, both are
cut to it (`section`, interpolating the ends exactly at the interval's
bounds) and joined into a ring: the first forwards, the second backwards.
So the area exists only where both graphs are defined, with a vertical
edge where either one stops (a jump, a pole, a domain edge). Where the
graphs cross, the ring crosses itself; its two lobes wind in opposite
directions, and the nonzero rule fills both. The rings are mapped to the
screen and clipped with `clip_polygon` (a side can be far off screen near
a pole), which keeps winding numbers inside the clip box unchanged.

## Vector and slope fields: `fields.py`

A field is drawn as a mark (an arrow, or a segment) at each node of a
lattice. The lattice has to keep panning rigid, keep a steady density on
screen at every zoom, and change continuously while zooming.
`field_lattice(view, spacing, margin)` does all three:

- **Anchored.** The nodes are multiples of a step in transformed
  coordinates, so they stay on the same world points while panning (a
  property test pans by random whole pixels), and on log axes they are even
  on screen.
- **Powers of two.** Per axis, `lattice_step(scale, spacing)` takes the
  power of two with `spacing / 2 <= step × scale < spacing`. Powers of two
  nest: when zooming in halves the step, every old node is still a node.
- **Fading.** The nodes at odd multiples of the step are the ones that
  appear at the halving. Their fade is 0 for the first half of the octave
  (`FADE_START`: measured as `log2(2 × step × scale / spacing)`), then rises
  by a smoothstep to 1 at its end. So the fully drawn marks are always
  between `spacing` and twice that apart, more than half of all zoom levels
  show no fading marks at all, and just before and just after a halving
  the visible nodes and their fades are the same (a test zooms through a
  halving by 2 × 10⁻⁹). A node that is odd along both axes takes the
  smaller of the two fades rather than their product, so there are at most
  three fade levels (two on equal-aspect planes), each one primitive.

Builders shorten marks that are fading in as well as making them fainter,
so that they grow into place: with marks at most `ARROW_FRACTION` (0.8) of
the spacing long, a growing mark never reaches its neighbours.

`screen_directions(view, x, y, u, v)` maps world vectors at world points to
unit vectors on screen through the axis transforms' derivatives
(`AxisTransform.derivative`: 1 on a linear axis, `1 / (x ln 10)` on a log
axis), so arrows are tangent to the field's flow lines and slope segments
to the solutions on any axes. Each vector is first divided by its larger
component, so that huge values next to a pole still have a direction;
vectors that are zero, infinite or undefined have none and get no mark.

`vector_field_fills` evaluates the field once on all the nodes
(`evaluate_vectors_quietly`), sets each arrow's length (the cap, or
`scale × |F|` up to it, times its fade), centres it on its node, and builds
one `Fill` of arrows per fade level. `slope_field_strokes` does the same
with the direction `(1, slope)` (`(0, ±1)` for an infinite slope) and one
`Stroke` of two-point segments per level.

On an 1100 × 720 plane a vector field takes about 7.5 ms, nearly all of it
rasterising the arrows, and about 13 ms while new arrows fade in (two
fills, the fading one with three times as many arrows), which only happens
while zooming; an unchanged field is reused in about 1.2 ms. A slope field
takes about 2.5 ms, strokes of short segments being cheaper to rasterise
than fills spread over the whole plane.

## Arrows: `arrows.py`

An arrow is one polygon of seven vertices (tail corners, shaft corners at
the head, barbs, tip) rather than a stroke with a head on top: a separate
shaft and head would overlap where they meet, and a translucent arrow would
be darker there. All the arrows of a call are rings of one `Fill` with the
nonzero rule; they all wind the same way (a property test checks the
signed area of random arrows), so overlapping arrows merge. `arrow_rings`
builds them for all arrows at once, leaving out arrows without a direction
(ends in the same place, or not finite). The head's size is in pixels
(`head_shape`: by default `3 × width + 4` long and 0.8 of that wide); an
arrow shorter than its head becomes a smaller head alone, with the same
proportions.

`arrow_fill` transforms the ends one by one (like a polyline's vertices,
clamping coordinates so that an end at zero on a log axis runs off the
plane), pairs a single start or end with all the others, and drops arrows
whose box lies wholly outside the plane. Fills of many small rings rely on
the rasteriser handling all rings together; see
[Rendering](rendering.md#fills).

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

## Integrals: `quadrature.py` and `integrals.py`

Integrals are the other half of calculus, and the library computes them
numerically, like the derivatives of `math` functions but with an
accuracy that does not degrade: **adaptive Gauss-Kronrod quadrature**
(QUADPACK's 21-point rule, `qk21`, and its error estimate), written for
NumPy rather than ported loop by loop.

`quadrature.integrate(func, lo, hi, piece, pieces)` integrates one robust
integrand over several *pieces* at once (one for `integral`, one per gap
between points for an antiderivative), each starting from one or more
intervals. Every round evaluates the integrand once, at the 21 nodes of
every interval being refined, so a NumPy function is called a handful of
times in all, and the work per round is a few matrix products. The rest
follows from the cases it must get right:

- **Global error control per piece.** A piece is done when the sum of its
  intervals' errors is at most `TOLERANCE` (1e-10) times its integral of
  `|f|`. Relative to `∫|f|`, not `|∫f|`, so `sin` over a period converges,
  and the result is independent of units. Per piece rather than per
  interval, because at an integrable singularity (`1/sqrt(x)` at 0) the
  interval touching it always has a large error *relative to its own
  size*; only its share of the whole shrinks.
- **Which intervals to cut.** Those whose error exceeds an equal share of
  the piece's tolerance. Rounds cost NumPy overhead (100 to 150 µs each)
  while evaluations are cheap, so how intervals are cut depends on how
  many need it. When one or two do, they are usually troubled at an end (a
  singularity, the lower limit of an antiderivative, a domain edge), so
  they are cut *graded*: into eighths, with further cuts at `2⁻⁴`, `2⁻⁸`,
  ..., `2⁻²⁴` of the width from each end (20 parts, closing in on an end by
  24 halvings per round). Up to 16 are cut into eighths (jumps, kinks);
  more are halved (oscillation, where halving wastes least). Graded cuts
  that would leave parts too narrow for distinct nodes (near an end away
  from 0) are left out.
- **When the two rules agree by chance.** QUADPACK's estimate rests on one
  difference, Kronrod minus Gauss. For an integrand that is not smooth
  (kinks, jumps, cusps), both rules have errors of similar size, and their
  difference crosses zero at particular positions of the feature: about 1%
  of kink positions get an estimate a thousand times too small (a
  property test found one). So the error is at least `TAIL_FACTOR` (8)
  times the half width times `|(c₁₉, c₂₀)|`, the interpolant's two highest
  Legendre coefficients (a fixed `(21, 2)` matrix of the node values). They
  do not decay at such features, and an odd and an even one do not vanish
  together, so the bound caught every one of 160,000 positions of kinks,
  jumps and cusps tested; for smooth integrands they are at rounding level
  by the time the rules agree. The bound is not applied to intervals too
  narrow to cut: there, next to a singularity away from 0, rounded node
  positions look like roughness that cannot be refined away, and the bound
  would turn arcsine-type integrals into NaN.
- **What neither rule sees.** The outermost nodes are 0.43% of the half
  width from the ends, so a jump or kink in those slivers is invisible to
  both rules, which then agree and report a tiny error for a wrong value
  (QUADPACK has the same blind spot). Each interval is therefore probed
  near both ends, and `|probe - P(probe)|` times the sliver's width (with
  `P` the nodes' interpolating polynomial, whose extrapolation weights
  are well conditioned) is added to its error. The probes sit 4e-6 of the
  half width inside the ends, not at them: a jump exactly at an end is a
  single point (`floor` integrated up to 1), and from the ends alone it
  would look like one just inside, needing refinement forever. A probe
  that rounds onto its end (a very narrow interval away from 0) tells
  nothing. 23 evaluations per interval in all.
- **Undefined points.** An interval with one undefined node has an
  infinite error and is cut, which turns the point into an end of the
  parts (never evaluated): `sin(x) / x` at the middle node of `[-1, 1]`
  vanishes after one round. Two or more undefined nodes mean the function
  is undefined on a stretch, and the piece is NaN at once (one evaluation
  for `sqrt` from -1, not a budget's worth).
- **Divergence.** A piece whose error has not halved in `STALL_ROUNDS` (8)
  rounds is given up: at `1/x` the error of the interval touching 0 never
  shrinks. Rounds (`MAX_ROUNDS`) and evaluations (`MAX_NEW_INTERVALS`) are
  bounded too, so no integral can freeze a frame.
- **Floating-point limits.** Next to a singularity away from 0
  (`1 / sqrt(1 - x²)` at 1) intervals cannot be cut below about `1e-13`,
  and the error stays above the target. A piece that missed the target
  keeps its value if its error is within `ACCEPTABLE` (1e-6) of `∫|f|`
  (there it is good to about 8 digits) and is NaN otherwise.
- **Speed for the common case.** If every piece converges on the first
  evaluation (smooth integrands), the result is returned without the
  refinement bookkeeping: about 0.1 ms for one integral, 0.7 ms for the
  1,100 gaps of an antiderivative's first sampler call.

The process is deterministic, as the plot cache requires, but pieces are
not independent bit for bit: how intervals are cut depends on how many
are cut in all.

`integrals.integral` validates the limits (NaN limits give NaN, equal ones
0, reversed ones the negative), wraps the user's function in robust
evaluation, and maps infinite limits onto finite intervals:
`x = c + 1/t - 1` for `t` in `(0, 1]` covers `[c, ∞)`, `x = c + 1/t + 1` for
`t` in `[-1, 0)` covers `(-∞, c]`, and both together the whole line, with
the integrand `f(x(t)) / t²`. The tail lands next to `t = 0`, where floats
are densest, so `1/x^1.5` converges and `1/x` is found to diverge. It starts
from 8 equal parts, so a narrow feature is less likely to fall between the
first nodes.

`Antiderivative` evaluates `F(x) = ∫ₐˣ f` by **cumulative quadrature**: the
query points and `a` are sorted, the gaps between neighbours are the pieces
of one run (each starting from parts in proportion to its share of the
range, so a long first gap from a distant `a` is not undersampled), and `F`
is their running sum outwards from `a` (`np.cumsum`, negated to the left).
A NaN gap makes `F` NaN beyond it, which is right: beyond a divergence the
integral from `a` does not exist. Infinite limits and points add tails
computed like `integral`'s. `derivative(F, n)` collapses to
`derivative(f, n - 1)` and `taylor_derivatives` handles antiderivatives the
same way, so `F'` is `f` exactly; a jet passed to `F` (inside a function
being differentiated) raises `TypeError`, falling back to numerical
differences, as for `Derivative`.

**Antiderivatives and the plot cache.** `F(x)` is a sum over the gaps
between the points it is evaluated with, so its last digits depend on
them: it is not bit-for-bit pointwise, and the cache's check (all recorded
points in one call) would never match. `Recorder` therefore sees through an
`Antiderivative`: it evaluates `Antiderivative(Recorder(f), a)`, recording
the *integrand's* evaluations, and a later frame's antiderivative is
unchanged if its lower limit is the same bit for bit and its integrand
gives the same values at every recorded node. Since the quadrature is
deterministic, the plot would then be computed exactly as before. An
antiderivative wrapped in a lambda (`lambda x: F(x) + 1`) is checked like
any function, and is drawn again every frame.

## Series: `series.py`

`taylor(f, a, n)` takes `derivatives(f, a, n)`, divides by the factorials
and returns a `TaylorPolynomial`, evaluated by Horner's rule in `x - a`.
`fourier(f, n, interval)` returns a `FourierSeries`. Both are computed when
created, are cheap callables afterwards, keep their coefficients as
read-only arrays, and accept jets, so they differentiate exactly.

The Fourier coefficients `aₖ, bₖ = (2/T) ∫ f(x) (cos, sin)(kωx) dx` all come
from **one** quadrature run over `f` alone, with `samples=True`: the run
starts from `max(8, 2n)` equal parts, so every part spans at most half a
period of the highest harmonic (on which the 21-point rule integrates
`cos` to rounding), and refines wherever `f` needs it (the jumps of a square
wave). Then `Σ w f(x) e^{-ikωx}` over all the final nodes gives every
coefficient, with the powers of `e^{-iωx}` built by repeated
multiplication (rounding grows like `kε`), not `2n` trigonometric calls per
node. Evaluating reduces `x` into the interval (the sum has period `T`) and
sums `Re Σ cₖ zᵏ` with a `cumprod` of `z = e^{iωx}`, in blocks of at most
`2¹⁷` entries; jets take a term-by-term loop instead, as they are not
complex.

## Reusing plots between frames: `memo.py`

Immediate mode redraws everything every frame, but most frames change
nothing, or only the plots that depend on a dragged point. `PlotCache`
(one per plane) reuses the result of each plotting call that has not
changed, and the renderer then reuses its rasterised coverage (see
[Rendering](rendering.md#reusing-coverage-coverage_cachepy)).

**Telling that a plot has not changed** is the whole problem. Function
identity says nothing: lambdas are created anew every frame, and the same
function can return new values because it reads a changed variable
(`lambda x: a.y * np.sin(x)`). Code or closure inspection is fragile. So
the cache relies on a property the samplers already have: they are
**deterministic**. Given the same view and settings, a sampler queries the
function at the same points in the same order *as long as it receives the
same values*, and so produces the same primitives. Hence:

1. While a call is built, its functions are wrapped in `Recorder`s, which
   evaluate robustly (like `evaluate_quietly`, so the builders' own
   evaluation passes the results through untouched) and record every input
   and result. A recorder knows its function's number of arguments
   (`arity`) and of results (`outputs`: 2 for vector fields, whose values
   it returns as a `(2, N)` array, like a field).
2. In the next frame, the call in the same position (calls are matched by
   order, like palette colours) is reused only if its kind is the same, its
   key is equal (`same_inputs`: the view's `fingerprint`, the style, the
   resolved colour, ranges, levels, sampling settings, copies of data
   arrays, constants standing in for curve components) and every recorder
   is `unchanged()`: the *current* function, called once on all the
   recorded inputs, returns exactly the recorded values, bit for bit (NaN
   equal to NaN, the sign of zero compared too).
3. If so, re-running the sampler would retrace the same steps, so the
   previous primitives *are* the result: the very same objects are returned.

`unchanged()` first checks `PROBE_POINTS` (32) recorded points spread over
the record, so a function that changes everywhere (an animation) is
rejected for almost nothing; only a function that passes the probe is
checked in full. A full check is one vectorised call over the recorded
points (a few thousand for a function, up to the evaluation budget for an
implicit curve), far cheaper than sampling and rasterising.

The one assumption is that functions are **pointwise**: their value at `x`
does not depend on the other points in the same call. NumPy itself is
reproducible here: the same input gives bit-identical results whatever the
size and alignment of the array it sits in (checked for the common ufuncs).
Were it not on some platform, the check would fail and the plot would be
drawn again: a lost speed-up, never a stale plot. A property test drives a
cached and an uncached plane through random sequences of changes (new
lambdas, changed closures, a narrow spike moving by less than a pixel, data
changed in place, plots appearing and disappearing, pans and zooms) and
compares every frame pixel for pixel.

`MAX_ENTRIES` (512) bounds the calls remembered per frame, which also bounds
memory for a program that never starts a frame. The grid scene is reused
the same way in `Plane._grid_scene`, keyed by the view's fingerprint and the
theme, except in the `"snap"` transition, whose hysteresis depends on
earlier frames.

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
- **Implicit curves** and **regions** are found on a grid in transformed
  coordinates and mapped to the screen precisely (`_grid_to_screen`); see
  [Mapping to the screen](#mapping-to-the-screen).
- **Areas between graphs** sample both sides as functions; see
  [Areas between graphs](#areas-between-graphs-areaspy).
- **Parametric and polar curves** wrap each component robustly
  (`_component`, which also accepts plain numbers), map them through
  `View.to_screen` and hand the result to `sample_curve`. Polar curves are
  parametric curves `(r cos θ, r sin θ)`.
- **Vector and slope fields** are drawn on a fading lattice; see
  [Vector and slope fields](#vector-and-slope-fields-fieldspy).
- **Polylines, polygons, points and arrows** transform vertex by vertex. On a log
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
