# Extending PyGridPlot

Recipes for the extensions the design anticipates, then the conventions
every change follows. Each recipe names the files to touch, in order.
Before starting on a new feature, check it against the
[philosophy](../philosophy.md#deciding-what-belongs): it should be a better
or a new plane method taking Python values, not a second way to draw.

- [A new kind of plot](#a-new-kind-of-plot)
- [A new kind of axis](#a-new-kind-of-axis)
- [A new kind of draggable point](#a-new-kind-of-draggable-point)
- [A new derivative rule](#a-new-derivative-rule)
- [A new marker shape](#a-new-marker-shape)
- [A new theme option](#a-new-theme-option)
- [A new primitive](#a-new-primitive)
- [A new backend](#a-new-backend)
- [A new public name](#a-new-public-name)
- [Examples and gallery views](#examples-and-gallery-views)
- [Coding conventions](#coding-conventions)

## A new kind of plot

Parametric, polar and implicit curves, areas between graphs,
inequalities, vector and slope fields and arrows were all added this way,
so they are worked examples to copy.
Take **inequalities** (`Plane.inequality`), which shade the region where a
condition such as `x**2 + y**2 < 4` holds:

1. **Sample it in `core/plot/`**, as a pure function that knows nothing
   about the view. `regions.sample_region(func, u_range, w_range, step,
   config)` takes the condition in transformed coordinates and returns
   rings in grid coordinates plus the grid's origin and step
   (`RegionRings`). It reuses the anchored grids of implicit curves
   (`lattice.py`: the coarse grid, cropping, the fine grid on candidate
   cells only, the budget), so panning stays rigid and the work follows
   the boundary. Look for such shared pieces first, and move them into a
   module of their own when a second sampler needs them, as `lattice.py`
   was split out of `implicit.py`. Evaluate robustly (inside one
   `np.errstate` block) and reuse an existing settings class if it fits
   (regions take `ImplicitSampling`), or put a new one, frozen and
   validated, next to the sampler.
2. **Build primitives in `core/plot/builders.py`.** `region_fill(view,
   func, style, color, *, sampling)` wraps the user's function robustly
   (`evaluation.evaluate_xy_quietly`) in transformed coordinates, so log
   axes work, maps to the screen precisely (`_grid_to_screen`, built on
   `View.multiples_to_screen`; use `View.to_screen` for other points),
   clips if needed (`geometry.clip_paths` or `clip_polygon`), and returns a
   primitive (`Fill`, `Stroke`), or `None` when nothing is visible. If no
   existing primitive fits, see [A new primitive](#a-new-primitive).
3. **Expose it in `plane.py`.** `Plane.inequality(surface, condition, *,
   color, opacity, style, sampling)` calls `self._sync_rect(surface)`,
   resolves the style with `with_overrides` and the colour with
   `self._colors.resolve`, builds the primitive through `self._cached`
   (with everything the result depends on in the key, and `arity=2` for a
   function of two variables, so unchanged frames are reused exactly;
   `vector_field` adds `outputs=2`, as its function returns a pair), and
   finishes with `self._draw(surface, primitive)` (several primitives may be
   passed, as `vector_field` does, one per fade level). Write the docstring for
   users: it becomes the [API reference](../user/reference.md). Export any
   new public class, such as `FillStyle`, from `pygridplot/__init__.py`
   (see [A new public name](#a-new-public-name)).
4. **Test it**: the sampler in `tests/core/plot/` (`test_regions.py`:
   accuracy against an independent check, edge cases such as asymptotes
   and undefined points, the budget, rigid panning as a property test),
   the builder in `tests/core/plot/test_builders.py` (log axes, clipping,
   broken functions), the whole path in `tests/test_plane_plotting.py`
   (the right pixels, palette colours, rigidity when panning), and the
   cache in `tests/test_plane_cache.py` (add the call to its scene).
5. **Document it**: run `python scripts/build_api_docs.py`, add a section to
   [Plotting](../user/plotting.md) with runnable examples and its limits,
   describe the method in [Plotting](plotting.md) for developers, add a
   benchmark case and a gallery scene, and consider an example program.

## A new kind of draggable point

Points are kept free, locked or on a curve by the constraints in
`core/handles/constraints.py`. A new one (a point on the graph of
`y = f(x)`, on an implicit curve, or snapped to a step) is a class with:

- `drag(view, target, current) -> Placement | None`: the position for a drag
  towards the screen point `target`, given the current `Placement`
  (`None` keeps the point where it is);
- a way to place the point from code (`AxisConstraint.place`,
  `CurveConstraint.at` and `nearest`), used by the `position` and `t`
  setters of `DraggablePoint`;
- if it caches anything that depends on time, an `invalidate()` that
  `DraggablePoint.begin_frame` calls.

Work on screen when the movement should follow what the user sees (as
`CurveConstraint` does), in transformed coordinates otherwise (as
`AxisConstraint` does, so log axes need no special cases). Add the option
to `DraggablePoint.__init__` and `Plane.draggable`, rejecting combinations
that make no sense, and test it in `tests/core/handles/` (projection,
bounds, log axes) and through events in `tests/test_dragging.py`. See
[Draggable points](dragging.md) for how the pieces fit together.

## A new derivative rule

A NumPy ufunc without a rule makes derivatives fall back to finite
differences. To differentiate it exactly, add a rule to `_RAW_RULES` in
`core/plot/jets.py`: a function from the inputs' coefficient arrays
(`c[k] = g⁽ᵏ⁾ / k!`, all of one order and shape) to the result's. The
helpers cover most cases:

- `_from_derivative(a, value, derivative)` when the derivative is easy to
  write with other rules (this is how `arctan` is done:
  `1 / (1 + u²)`);
- `_flat(op)` for piecewise-constant functions;
- `_select(condition, a, b)` for functions that pick one input per point.

Test the rule in `tests/core/plot/test_jets.py` against the derivatives
worked out by hand, at several orders. A NumPy *function* (not a ufunc)
goes in `_FUNCTIONS`, handled through `__array_function__`.

## A new kind of axis

For example a symmetric log axis (linear near zero, logarithmic beyond):

1. Add the kind to `AxisTransform` in `core/transforms.py` with its
   `forward`, `inverse` and `check` behaviour, and a `default_center`.
2. The `View`, interaction, framing and function and curve plotting then
   work unchanged, because they only use transformed coordinates. Check
   `View._scale_range` and `View._clamp_center` for the right zoom and pan
   limits, and `View.set_log`-style switching if users should be able to
   switch to it.
3. Give the grid a tick generator for it, like `core/grid/logticks.py`:
   return `AxisLines` (per-line opacities, continuous in the zoom) and,
   with a labels module like `loglabels.py`, `LabelCandidate`s; add a
   branch in `scene.axis_grid` and decide its overlap handling.
4. Add `Plane` and `View` constructor options, and repeat the grid property
   tests (`tests/core/grid/test_logticks.py` is the model) for the new axis.

## A new marker shape

1. Add the name to `MarkerShape` and `MARKER_SHAPES` in
   `core/primitives.py`.
2. Add a signed-distance branch to `render/raster._marker_sdf`, scaled so the
   shape has the circle's area. If it reaches beyond `1.45 × radius`, widen
   the stamp in `marker_coverage`.
3. Test its area and that it fits its stamp in `tests/render/test_raster.py`,
   and add it to the shapes gallery scene.

## A new theme option

1. Add a field (with validation in `__post_init__`) to `GridTheme` in
   `core/grid/theme.py`, documented in the class' `Attributes`.
2. Use it where the grid is built, instead of a constant. Every visual
   setting should live in the theme rather than being hard-coded.
3. Override it in `DARK` if it is a colour, add a playground toggle if it
   is worth trying live, and update [Styles and themes](../user/styles-and-themes.md).

## A new primitive

If no existing primitive fits (an image, say):

1. Define it in `core/primitives.py` as a frozen dataclass in screen pixels.
2. Add a method for it to the `Renderer` protocol in `render/base.py`, and
   implement it in `PygameRenderer`.
3. Dispatch it in `Plane._draw`.

Keep primitives few and general: every one must be implemented by every
backend. Most new kinds of plot need none: arrows, for example, are rings
of a `Fill`, and slope fields are `Stroke`s of two-point paths.

## A new backend

Implement the `Renderer` protocol (`render/base.py`): `measure_text`,
`draw_grid`, `draw_stroke`, `draw_fill` and `draw_markers`, working from the
primitives alone. Everything in `core` stays as it is. `render/raster.py`
is pure NumPy, so an image-based backend can reuse it.

`Plane` and `interaction.py` are the pygame-specific integration (`Plane`
uses the renderer's `clipped()` context manager and pygame surfaces and
events). A backend for another windowing library needs its own thin
façade, or a `Plane` generalised over the renderer and event types. An
export backend (SVG, say) needs no events at all: build the grid with
`build_grid` and the primitives with the builders, then write them out.

## A new public name

Export it from `pygridplot/__init__.py` (and `__all__`), then add it to a
section of `SECTIONS` in `scripts/build_api_docs.py`. The script refuses to
run while a public name is missing, so the reference can never silently
leave it out. Run the script and commit `docs/user/reference.md`.

## Examples and gallery views

- An **example** is a program in `examples/` with a module docstring that
  explains what it shows and which keys it uses, and a `main()` function.
  `tests/test_examples.py` runs every example automatically; list it in
  [Recipes](../user/recipes.md#examples) and the README.
- A **gallery view** is one `GalleryView(...)` line in
  `scripts/render_gallery.py`, optionally with a scene function that draws
  plots. Add one for every visual case a change could break.

## Coding conventions

**Structure.**

- Keep `core` pure (standard library, NumPy and contourpy only; never
  pygame), keep the foundation modules free of `core.grid` and
  `core.plot`, and keep those two apart.
  The architecture tests enforce this.
- One module per idea, with a module docstring that says what the module
  is for, how it fits with its neighbours and which function is its entry
  point. Split a module when it starts doing two things (as `logticks`
  and `loglabels` were split).
- Names starting with `_` are private to their module and never imported
  elsewhere; if another module needs something, make it public and
  document it.

**Documentation.** Every module, class and function has a Google-style
docstring (`Args:`, `Returns:`, `Raises:`, `Attributes:`), enforced by
ruff. Docstrings explain *why* as well as *what*: units (pixels or world
units), conventions (which way y points, which end is inclusive) and edge
cases. Constants are named, in `UPPER_CASE`, with a docstring below them
saying what they control. Public docstrings are written for users, because
they become the API reference.

**Types and values.** Full type hints; `src/` passes `mypy --strict`.
Settings are frozen dataclasses validated in `__post_init__`, so invalid
values raise `ValueError` when they are created, never in the middle of a
frame. Drawing never raises because of user data (functions, NaN points):
bad values become gaps.

**Numbers.** Spacings are `NiceStep` levels and label values `Decimal`s,
never floats, so they stay exact. Coordinates follow the conventions in
[Architecture](architecture.md#coordinates); snap with `core/pixels.py`,
never with `round`.

**Performance.** Per-frame code is vectorised with NumPy: loop in Python
over a handful of things (axes, tiers, decades), never over points or
pixels. Use `float32` for large per-pixel arrays. Measure with
`scripts/benchmark.py` before and after, and keep a change only if it does
not break a guarantee (rigid panning, no pops) for the sake of speed.

**Spelling.** API names use the US spellings pygame uses (`color`,
`center`); documentation and comments use British spelling (colour,
centre).

**Tests and docs** come with the change: see the
[checklist](testing.md#the-checklist).
