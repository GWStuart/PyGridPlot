# PyGridPlot developer guide

This guide explains how PyGridPlot is put together, so that you can find your
way around the code and extend it. For how to *use* the library, see the
[user guide](user-guide.md). Every module, class and function also has a
docstring explaining what it does and why, so the code is the detailed
reference.

- [Design principles](#design-principles)
- [Package map](#package-map)
- [Layers](#layers)
- [What happens in a frame](#what-happens-in-a-frame)
- [Key ideas](#key-ideas)
- [Extending PyGridPlot](#extending-pygridplot)
- [Testing and tooling](#testing-and-tooling)
- [Known limitations](#known-limitations)

## Design principles

1. **Library, not framework.** The user owns the main loop. The plane is an
   object they call (`handle_event`, `draw_grid`, `plot`, ...); it never takes
   control.
2. **Immediate mode.** Everything is recomputed from the view every frame.
   There is no retained scene to keep in sync, so animation and interaction
   come for free. The only state kept between frames is small: the view,
   eased-motion springs, snap hysteresis, and caches.
3. **Pure maths, thin backend.** All geometry, sampling, layout and
   formatting lives in `pygridplot.core`, which never imports pygame. Code
   there describes *what* to draw as backend-neutral **primitives** in screen
   space, and a renderer decides *how*.
4. **Exact where it matters.** Grid spacings are integers (a 1-2-5 ladder),
   grid lines come from integer indices, and labels are formatted from exact
   decimals. Float drift and `0.30000000000000004` cannot appear.
5. **No pops, no jumps.** Every visible change while panning or zooming is
   continuous. Spacings cross-fade, labels fade rather than appear or vanish,
   labels near edges are pushed rather than flipped, and samples are anchored
   to the world so curves move rigidly. Tests check these properties directly.

## Package map

```
src/pygridplot/
├── __init__.py          Public API re-exports (Plane, styles, themes, show, ...)
├── plane.py             Plane: the façade tying view, theme, controls and renderer together
├── interaction.py       pygame events -> pan/zoom (drag, wheel, keys, eased motion)
├── quick.py             show(): a ready-made main loop
├── core/                PURE: stdlib + NumPy only (enforced by tests/test_architecture.py)
│   ├── coords.py        ScreenRect, Bounds, Axis, ColorLike
│   ├── pixels.py        The pixel-snapping conventions (single source of truth)
│   ├── view.py          View: world/screen transforms, pan/zoom maths, ViewLimits
│   ├── easing.py        Critically damped Spring for eased motion
│   ├── nice.py          NiceStep: exact 1-2-5 spacings as integer ladder levels
│   ├── spacing.py       Choosing a spacing per axis: cross-fades (fade) or hysteresis (snap)
│   ├── ticks.py         Candidate grid lines from integer indices, classified major/minor
│   ├── gridlines.py     Classified lines -> painted LineBatch primitives, plus the axes
│   ├── labels.py        Exact number formatting -> LabelText (with superscript runs)
│   ├── ticklabels.py    Which labels show and how opaque: thinning, fades, notation
│   ├── layout.py        Label placement: rows/columns, push-and-pin, collision fades
│   ├── grid.py          build_grid(): orchestrates the above into a GridScene
│   ├── style.py         LineStyle / PointStyle / PolygonStyle, ColorCycle (palette)
│   ├── evaluation.py    Calling user functions robustly (errors/complex -> NaN)
│   ├── sampling.py      Adaptive sampling of y = f(x): refinement, jumps, poles, edges
│   ├── geometry.py      Point input normalisation; polyline and polygon clipping
│   ├── plotting.py      Plotted objects -> clipped screen primitives (Stroke/Fill/Markers)
│   ├── primitives.py    The primitives, plus the TextMeasurer protocol
│   └── theme.py         GridTheme, LIGHT, DARK
├── render/              The pygame backend
│   ├── base.py          Renderer protocol (what any backend must provide)
│   ├── pygame_renderer.py   PygameRenderer: draws every primitive type
│   ├── raster.py        Antialiased coverage: strokes, fills, markers (pure NumPy)
│   ├── composite.py     Blending coverage onto pygame surfaces
│   ├── markers.py       Cached, sub-pixel marker stamps
│   ├── text.py          Label typesetting (runs, superscripts, halos), cached
│   ├── fonts.py         Font loading; the bundled DejaVu Sans
│   └── colors.py        Colour resolution and paint-stack compositing
└── assets/fonts/        DejaVuSans.ttf and its licence

tests/                   Mirrors the package: tests/core, tests/render, top-level integration tests
examples/                Runnable programs (also smoke-tested)
scripts/render_gallery.py   Renders tricky views to PNGs for visual regression checks
```

## Layers

```
   user program
        │
   plane.py ─────────── interaction.py          (pygame-specific integration)
     │      │
     │      └── render/  (backend: turns primitives into pixels)
     │              │
     └──────────────┴── core/   (pure maths: never imports anything above it)
```

`core` may import only the standard library, NumPy and other `core` modules.
`tests/test_architecture.py` parses every core module's imports and fails if
this is broken. Keeping `core` pure means it is fast to test without a
display, and that a different backend (OpenGL, SVG export) can reuse all of
it.

The boundary between `core` and a renderer is small:

- **Primitives** (`core/primitives.py`), all in continuous screen pixels:
  `LineBatch` (axis-aligned grid lines with a paint stack), `TextItem`,
  `Stroke` (polylines with a round pen), `Fill` (polygons with a fill rule),
  and `Markers` (identical point markers).
- **`TextMeasurer`**: the one thing core needs *from* a renderer, because
  label layout must know label sizes.

## What happens in a frame

**`plane.draw_grid(surface)`**

1. `interaction.InteractionController.update()` advances eased motion (a
   `Spring` per pan/zoom axis) and applies it to the `View`.
2. `ColorCycle.reset()` restarts the palette for the new frame.
3. `core.grid.build_grid(view, theme, renderer, state)` builds a `GridScene`:
   1. `spacing.py` picks each axis' major spacing for the current scale. In
      `"fade"` mode this is a pure function returning a blend of two ladder
      levels.
   2. `ticks.py` generates candidate lines as integer multiples of a common
      base step and classifies them under each active level.
   3. `gridlines.py` groups lines with the same classification into
      `LineBatch`es. A paint stack lets a line be "60% major, 40% minor".
   4. `ticklabels.py` formats and measures the major-line labels, thins them
      if they would crowd, and computes each label's opacity (fades).
   5. `layout.py` places them. The x row sits below its axis, the y column to
      its left; both are pushed and pinned at the edges, and labels fade near
      higher-priority ones.
4. `PygameRenderer.draw_grid` fills the background, draws line batches as
   1-pixel rectangles (crisp), and blits cached label surfaces.

**`plane.plot(surface, f)`** (other plot calls are analogous)

1. Style resolution: `style.with_overrides()` merges keyword arguments into
   the style, and `ColorCycle.resolve()` picks a colour.
2. `core.plotting.function_stroke()` takes the visible x-range plus a
   margin, calls `sampling.sample_function()` to get world polylines,
   converts them to screen pixels (`View.to_screen`), and clips them
   (`geometry.clip_paths`). The result is one `Stroke`.
3. `Plane._draw()` clips to the plane's rectangle and dispatches each
   primitive to the renderer: `draw_stroke`, `draw_fill` or `draw_markers`.
4. `PygameRenderer.draw_stroke()` computes antialiased coverage with
   `raster.stroke_coverage()` and blends it with `composite.blend_sparse()`.

## Key ideas

**Coordinates.** World y points up; screen y points down. Integer screen
coordinates are pixel *centres*, and `core/pixels.py` defines snapping
(`floor(p + 0.5)`, never banker's rounding). The view centre is drawn at a
whole origin pixel (`rect.x + w // 2`), so the default view puts lines
exactly on pixels. **All** world/screen conversion goes through `View`.

**Exact spacings.** `NiceStep(level)` stands for `(1, 2, 5)[level % 3] ×
10^(level // 3)`. Minor steps are always `level - 2`, and divisibility checks
use integers. Grid line `n` of a base step is at `n × base`, computed relative
to the view centre (`View.multiples_to_screen`) so spacing stays even at deep
zoom.

**Cross-fades.** Around each switching point between two ladder levels
there is a narrow band of zoom (`fade_band`) where both grids are drawn,
blended by a smoothstep weight. Labels fade *sequentially*: the old labels are
gone before the new ones appear. Label text depends only on the value and the
level's notation, so shared labels never flicker.

**Labels.** Thinning keeps every 2nd/5th/10th label using a nested sequence
of nice steps, so it only ever removes labels, and the ones about to be
removed fade first. Collisions between the x row, the `0` and the y column
fade by distance, weighted by opacity.

**Sampling** (`sampling.py`). Initial samples sit at `x = k × step` (one per
pixel column), so a pan by whole pixels reuses exactly the same world points
and curves move rigidly. Vectorised refinement rounds then halve only the
segments that matter: their midpoint is more than `tolerance_px` off the
chord *vertically*, or they straddle an undefined value. Segments still
unresolved after the last round get one more midpoint test, which separates
real discontinuities (a step, or a pole that overshoots) from steep continuous
branches. Finally, sign-changing poles and domain edges are chased by
bisection, so branches reach the plane's edge and edges are exact. A budget
caps evaluations per frame.

**Rasterising** (`render/raster.py`, pure NumPy):

- **Strokes** are unions of capsules (segment distance fields, combined
  with `max`), so joins are round and a curve split into many segments looks
  exactly like a smooth one. Segments are cut into pieces with tight bounding
  boxes (longer pieces for axis-aligned segments) before every candidate pixel
  is tested.
- **Fills** use five sub-scanlines per pixel row, with exact horizontal span
  coverage accumulated by a cumulative-sum trick. They support nonzero and
  even-odd rules.
- **Markers** are distance-field stamps, cached per style and quarter-pixel
  phase, and blitted with `fblits`.

`render/composite.py` blends sparse coverage per pixel through
`surfarray` (or as a patch when it is huge), and handles per-pixel-alpha
targets with the full "over" formula.

**Interaction.** Dragging pins the grabbed world point to the cursor
(`View.pin`), so it cannot drift. Wheel and key input push targets into
critically damped `Spring`s (`core/easing.py`), which are advanced in
`update()` with an exact, frame-rate-independent solution.

## Extending PyGridPlot

**A new kind of plot** (for example parametric curves `(x(t), y(t))`):

1. Sample it in `core`. Write `sample_parametric(fx, fy, t_range, scale,
   band, config)` next to `sample_function`, reusing
   `evaluation.evaluate` and the same refinement idea, with the deviation
   measured in screen pixels along the curve.
2. Build primitives in `core/plotting.py`. Add `parametric_stroke(view, ...)`
   that converts to screen space and clips with `geometry.clip_paths`,
   returning a `Stroke`.
3. Expose it in `plane.py`. Add `Plane.parametric(surface, fx, fy, ...)`:
   resolve the style with `with_overrides`, pick the colour with
   `self._colors.resolve`, build the stroke, and call `self._draw(surface,
   stroke)`.
4. Add tests in `tests/core/` for the sampler and builder, and in
   `tests/test_plane_plotting.py` for an end-to-end check (pan rigidity
   included).

**A new marker shape.** Add the name to `MarkerShape` and `MARKER_SHAPES`
in `core/style.py`, then add a signed-distance branch to
`render/raster._marker_sdf`. If the shape reaches beyond `1.45 × radius`,
widen the stamp in `marker_coverage`. Test its area in
`tests/render/test_raster.py`.

**A new theme option.** Add a field (with validation in `__post_init__`) to
`GridTheme` in `core/theme.py`, use it where the grid is built, and override it
in `DARK` if it is a colour. Every visual setting should live in the theme
rather than being hard-coded.

**A new backend.** Implement the `Renderer` protocol (`render/base.py`):
`measure_text`, `draw_grid`, `draw_stroke`, `draw_fill` and `draw_markers`,
working from the primitives. `Plane` and `interaction.py` are the
pygame-specific integration: `Plane` also uses the renderer's `clipped()`
context manager and pygame events. A different windowing library would need
its own thin façade, or a `Plane` generalised over the renderer type.
Everything in `core` stays as it is.

## Testing and tooling

```bash
pytest                                  # ~340 tests, a few seconds, fully headless
ruff check . && ruff format --check .   # lint and formatting (Google docstrings required)
mypy                                    # strict type checking of src/
```

- **Unit tests** mirror the package (`tests/core`, `tests/render`).
  `conftest.py` sets SDL's dummy drivers and provides `FakeMeasurer` (fixed
  label sizes) for testing layout without fonts.
- **Property tests** (Hypothesis) check the important invariants:
  - grid lines stay in the plane, labels never overlap, and zooming never
    pops labels;
  - panning moves grid lines and plots rigidly;
  - clipping keeps shapes in their box, and splitting a stroke never changes
    it;
  - fills preserve polygon areas.
- **Pixel tests** render to real surfaces: crisp grid colours, exact stroke
  widths, alpha blending, clipping, constant point size and polygon scaling.
- **Visual tools**, for things tests can't judge:
  - `scripts/render_gallery.py --out DIR` renders 24 tricky views plus a
    contact sheet. Diff two runs by eye after a change.
  - `examples/zoom_sweep.py --save-frames DIR` dumps frames of an automated
    zoom and pan.
  - `examples/grid_playground.py` toggles theme settings live and shows
    timings.

Conventions: every module, class and function has a Google-style docstring;
`src/` passes `mypy --strict`; constants are named, documented, and not
hard-coded in drawing code; ruff never touches Markdown (`extend-exclude`).

## Known limitations

- Switching an axis between fixed and scientific notation (around 10⁶ and
  10⁻⁴) changes its labels instantly rather than fading.
- Infinitely oscillating functions (`sin(1/x)` near 0) are expensive
  (10–20 ms per frame when zoomed in), because their curve covers much of the
  screen.
- High-DPI scaling is not handled. Sizes are in physical pixels.
- `Plane` is pygame-specific (see [A new backend](#extending-pygridplot)).
- Roadmap items from the project plan not yet started: parametric, polar and
  implicit curves, inequalities, a retained-mode layer, export, and
  draggable points.
