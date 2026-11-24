# Architecture

## Design principles

These are the technical side of the [philosophy](../philosophy.md) (one
way to do things, Python first, a library not a framework, immediate
mode): how the code is arranged so that the library keeps its promises.

1. **Library, not framework.** The user owns the main loop. The plane is an
   object they call (`handle_event`, `draw`, `plot`, ...); it never
   takes control, never sleeps and never polls events itself.
2. **Immediate mode.** Everything is recomputed from the view every frame.
   There is no retained scene to keep in sync, so animation and interaction
   come for free. The only state kept between frames is small and explicit:
   the view, the easing springs, snap hysteresis, the colour cycle's
   position, the framing (home view, pending fits), the draggable points,
   the sliders and caches. There are no retained plot objects (see
   [Retired experiments](#retired-experiments)).
3. **Pure maths, thin backend.** All geometry, sampling, layout and
   formatting lives in `pygridplot.core`, which never imports pygame. Code
   there describes *what* to draw as backend-neutral **primitives** in
   screen space; a renderer decides *how*.
4. **Exact where it matters.** Grid spacings are integers on a 1-2-5
   ladder, grid lines come from integer indices, and labels are formatted
   from exact decimals. Float drift (`0.30000000000000004`, `-0`) cannot
   appear.
5. **No pops, no jumps.** Every visible change while panning or zooming is
   continuous. Spacings cross-fade, labels fade rather than appear or
   vanish, labels near edges are pushed rather than flipped, and samples are
   anchored so that curves move rigidly. Property tests check these
   guarantees directly (see [Testing](testing.md#property-tests)).
6. **Robust by construction.** User functions may raise, return NaN or
   complex values, or be scalar-only; none of that can break a frame.
   Invalid settings are rejected with a clear `ValueError` when they are
   created, not while drawing.

## Layers

```text
   user program
        │
   pygridplot (__init__.py: the public API)
        │
   grid.py ── interaction.py ── dragging.py ── quick.py     pygame-specific integration
     │      │
     │      └── render/        the backend: primitives -> pixels
     │              │
     └──────────────┴── core/  pure maths: never imports anything above it
                         ├── grid/     the adaptive grid and its labels
                         ├── plot/     plotted objects -> primitives
                         └── handles/  draggable points, sliders (uses grid/ and plot/)
```

`core` may import only the standard library, NumPy, contourpy (a small,
pure contouring library, used for implicit curves) and other `core`
modules. `tests/test_architecture.py` parses the imports of every module
under `core/` and fails if this is broken. Keeping `core` pure means it is
fast to test without a display, easy to reason about, and reusable by
another backend (OpenGL, SVG export) without changes.

Within `core` there are two more rules, also enforced by
`tests/test_architecture.py`:

- The **foundation modules** (`core/*.py`: the view, transforms, primitives,
  ...) never import `core.grid` or `core.plot`. They are shared by both.
- **`core.grid` and `core.plot` never import each other.** Each can change or
  grow without touching the other. `core.handles` (draggable points) builds
  on both, and nothing in `core` imports it. Anything both need (such as the label
  text and marker vocabulary in `primitives.py`) belongs in a foundation
  module.

The boundary between `core` and a renderer is deliberately small:

- **Primitives** (`core/primitives.py`), all in continuous screen pixels:
  `LineBatch` (axis-aligned grid lines with a paint stack), `TextItem` (a
  placed label), `Stroke` (polylines with a round pen), `Fill` (polygons
  with a fill rule) and `Markers` (identical point markers).
- **`TextMeasurer`**: the one thing core needs *from* a renderer, because
  laying out labels requires their sizes.

## Package map

```text
src/pygridplot/
├── __init__.py          The public API: Grid, styles, themes, configs, View, show
├── grid.py              Grid: the façade tying view, theme, controls and renderer together
├── interaction.py       pygame events -> pan/zoom (anchored drag, eased wheel and keys)
├── dragging.py          pygame events -> draggable points and sliders (hit test, drag, hover, cursor)
├── quick.py             show(): a ready-made main loop
├── color_maps.py        color_map(): colours for values (resolves colour names with pygame)
├── _pygame_ce.py        On import: refuses the original pygame, with advice
├── py.typed             Marks the package as typed (PEP 561)
├── core/                PURE: stdlib + NumPy (+ contourpy) only
│   ├── coords.py        ScreenRect, Bounds, Axis, ColorLike
│   ├── pixels.py        The pixel-snapping conventions (single source of truth)
│   ├── transforms.py    AxisTransform: linear and log10 axes
│   ├── view.py          View: transforms, pan/zoom maths, fitting, ViewLimits, ViewState
│   ├── framing.py       ViewFraming: initial ranges, the home view, switching axis kinds
│   ├── easing.py        Critically damped Spring for eased motion
│   ├── primitives.py    The drawing primitives and the TextMeasurer protocol
│   ├── grid/            The adaptive grid
│   │   ├── nice.py          NiceStep: exact 1-2-5 spacings as integer ladder levels
│   │   ├── spacing.py       Choosing a spacing: cross-fades (fade) or hysteresis (snap)
│   │   ├── ticks.py         Linear axes: candidate lines from integer indices, classified
│   │   ├── ticklabels.py    Linear axes: which labels show and how opaque
│   │   ├── logticks.py      Log axes: decades, log-paper lines and subdivisions
│   │   ├── loglabels.py     Log axes: their labels
│   │   ├── gridlines.py     AxisLines (per-line opacities) -> LineBatch primitives, axes, arrowheads
│   │   ├── formatting.py    Exact number formatting -> LabelText (with superscripts)
│   │   ├── layout.py        Label and axis title placement: push-and-pin, collision fades
│   │   ├── categories.py    Names in place of numbers along an axis, thinned
│   │   ├── scene.py         build_grid(): the pipeline, into a GridScene
│   │   └── theme.py         GridTheme, LIGHT, DARK
│   ├── plot/            Plotted objects -> primitives
│   │   ├── style.py         Line, Point, Polygon, Fill, Arrow, Sequence, Bar and TextStyle; ColorCycle
│   │   ├── evaluation.py    Calling user functions robustly (errors/complex -> NaN)
│   │   ├── sampling.py      Adaptive sampling of y = f(x): jumps, poles, domain edges
│   │   ├── curves.py        Adaptive sampling of parametric curves on screen
│   │   ├── jets.py          Taylor arithmetic: exact derivatives through NumPy code
│   │   ├── derivatives.py   derivative(), derivatives(): jets, point by point, numerical
│   │   ├── quadrature.py    Adaptive Gauss-Kronrod quadrature, many integrals at once
│   │   ├── integrals.py     integral(), antiderivative(): infinite limits, cumulative sums
│   │   ├── series.py        taylor(), fourier(): Taylor polynomials, Fourier partial sums
│   │   ├── lattice.py       Anchored coarse and fine grids, and their budget
│   │   ├── implicit.py      Implicit curves F(x, y) = 0: marching squares (contourpy), checked
│   │   ├── regions.py       Regions where a condition holds: filled contours, refined
│   │   ├── areas.py         The area between two graphs, as rings
│   │   ├── fields.py        The fading lattice of vector and slope fields; directions
│   │   ├── sequences.py     Which terms of a sequence to draw: thinned by powers of 3
│   │   ├── arrows.py        Arrows as polygons, shaft and head in one ring
│   │   ├── text.py          Text at a world point: anchors, gaps, lines
│   │   ├── data.py          Data: colours and sizes per point, error bars, steps, bars
│   │   ├── histograms.py    Binning samples: anchored bins, an automatic 1-2-5 width
│   │   ├── distributions.py Box plot statistics, violin densities (kernel density)
│   │   ├── color_maps.py    Colour maps: values to colours, the built-in maps
│   │   ├── geometry.py      Point input; clipping polylines and polygons
│   │   ├── dashes.py        Dash patterns, anchored to the plane, cut and clipped
│   │   ├── memo.py          PlotCache: reusing unchanged plots between frames, exactly
│   │   └── builders.py      One function per kind of plot -> clipped primitives
│   └── handles/         Draggable points, sliders and the legend: drawn over the plots
│       ├── point.py         DraggablePoint: position, constraint, style, state
│       ├── slider.py        Slider: a value on a track, exact steps, log tracks, state
│       ├── panel.py         The corner panels of sliders: layout and primitives
│       ├── legend.py        The legend: entries, its corner panel, swatches
│       ├── color_bar.py     The colour bar: its corner panel, strip and ticks
│       ├── constraints.py   Free, locked and on-curve movement, with bounds
│       ├── tracking.py      Following a curve with the cursor, branch by branch
│       ├── labels.py        The coordinate label: rounded to a pixel, placed
│       └── drawing.py       Markers, hover ring and label as primitives
├── render/              The pygame backend
│   ├── base.py          The Renderer protocol (what any backend must provide)
│   ├── pygame_renderer.py   PygameRenderer: draws every primitive type
│   ├── raster.py        Antialiased coverage: strokes, fills, markers (pure NumPy)
│   ├── composite.py     Blending coverage onto pygame surfaces
│   ├── coverage_cache.py    The coverage of reused primitives, kept between frames
│   ├── markers.py       Cached, sub-pixel marker stamps
│   ├── text.py          Label typesetting (runs, superscripts, halos), cached
│   ├── fonts.py         Font loading; the bundled DejaVu Sans
│   └── colors.py        Colour resolution and paint-stack flattening
└── assets/fonts/        DejaVuSans.ttf and its licence

tests/                   Mirrors the package (tests/core/grid, tests/core/plot, tests/render, ...)
examples/                Runnable programs (each smoke-tested)
scripts/                 render_gallery.py, benchmark.py, build_api_docs.py
docs/                    These pages (user/ and dev/)
```

## What happens in a frame

### `grid.handle_event(event)`

`dragging.SliderController` sees every event first: a press on a
[slider](dragging.md#sliders) sets it, and a press or the wheel anywhere on
a slider panel, or on the legend's panel, is used up. Then `dragging.DragController`: a press on a
[draggable point](dragging.md) grabs it and is used up, and motion then
drags the point. Everything else goes on to
`interaction.InteractionController`, which turns events into view changes:

- **Drag.** On button-down inside the plane it records the grabbed point
  in *transformed* coordinates (see [Coordinates](#coordinates)); every
  mouse motion then *pins* that point under the cursor
  (`View.pin_transformed`). The view is recomputed from the anchor each
  time, so a drag can never drift.
- **Wheel and keys** add to the targets of critically damped springs
  (`core/easing.Spring`): one for zoom, two for pan. Nothing moves yet.

### `grid.draw(surface)`

1. **Rect.** An automatic plane takes the surface's size; the first time,
   `ViewFraming.settle()` redoes any provisional fit at the real size.
2. **Easing.** `InteractionController.update()` advances the springs by the
   real elapsed time (an exact, frame-rate-independent solution) and
   applies the change to the `View`, zooming about the anchor.
3. **Frame.** `ColorCycle.reset()` restarts the palette, and the legend's
   list of labelled calls starts empty.
4. **Grid.** `core.grid.scene.build_grid(view, theme, renderer, state,
   x_title=..., y_title=...)` builds a `GridScene`; [The grid](grid.md)
   explains each step. An unchanged grid is reused, and an arrowhead that
   did not move is swapped for last frame's (equal) one, so its coverage
   is reused.
5. **Draw.** `PygameRenderer.draw_grid` fills the background, draws the line
   batches as crisp 1-pixel rectangles, fills the arrowheads and blits
   cached label surfaces.

### `grid.plot(surface, f)` and the other plotting methods

1. **Style.** `style.with_overrides()` merges the keyword arguments into the
   style, and `ColorCycle.resolve()` picks the colour (the next palette
   colour if none was given). A `label=` adds a `LegendEntry` (the label,
   style and colour) to the frame's list, for `grid.legend`.
2. **Primitives.** A builder in `core/plot/builders.py` turns the object
   into clipped screen-space primitives: `function_stroke` (via
   `sampling.sample_function`), `parametric_stroke` and `polar_stroke` (via
   `curves.sample_curve`), `implicit_stroke` (via
   `implicit.sample_implicit`), `polyline_stroke`, `polygon_shapes` or
   `point_markers`. [Plotting](plotting.md) explains the samplers.
3. **Draw.** `Grid._draw()` clips the surface to the plane's rectangle and
   dispatches each primitive to the renderer (`draw_stroke`, `draw_fill`,
   `draw_markers`); [Rendering](rendering.md) explains how they become
   pixels.

`grid.text` builds `TextItem`s with `builders.text_items` (anchored by
`core/plot/text.py`) and draws them with `Grid._draw_texts`; it is not
cached, as measuring and placing text is cheap and the renderer caches the
rendered text. `grid.legend` lays out the frame's entries in a corner
panel (`core/handles/legend.py`), beyond the sliders' panel if they share
the corner, and draws it like the slider panels. `grid.color_bar` does the
same with a colour bar (`core/handles/color_bar.py`), beyond the sliders
and the legend.

## Coordinates

Three coordinate systems are involved, and **every conversion goes through
`View`**, so the conventions live in one place:

- **World coordinates**: the user's, with y pointing up.
- **Transformed coordinates** `u = T(x)`: the world value on a linear axis,
  `log10` of it on a log axis (`core/transforms.py`). The screen is linear
  in them, so the view stores its centre and scale in transformed
  coordinates (a log axis' scale is in pixels per decade), and panning,
  zooming, fitting, drag anchors and the zoom limits need no special cases
  for log axes. Anything that must be evenly spaced on screen (grid lines,
  function samples) works in transformed coordinates.
- **Screen coordinates**: pygame pixels, y pointing down. Integer values are
  pixel *centres*; `core/pixels.py` defines snapping (`floor(p + 0.5)`,
  never banker's rounding) and is used by both the layout and the renderer,
  so labels sit a fixed number of pixels from their lines.

The view centre is drawn at a whole *origin pixel* (`rect.x + w // 2`), so
the default view puts grid lines exactly on pixels.

## State between frames

| State | Where | Why |
|---|---|---|
| Centre, scale, axis kinds | `View` | What is visible. |
| Springs, drag anchor | `InteractionController` | Eased motion and anchored dragging. |
| Draggable points | `DraggablePoint`, listed by the `Grid` | Positions the user changes; the grabbed point and hover in `DragController`. |
| Sliders | `Slider`, listed by the `Grid`; the last frame's panels (`Grid._panels`) | Values the user changes; what a press hits is what was drawn; the dragged slider and hover in `SliderController`; the last primitives, for reusing their coverage. |
| Home view, pending fits | `ViewFraming` | `reset_view`, and fitting ranges once the size is known. |
| Snap levels | `GridState` | Hysteresis in `"snap"` mode only (fades are stateless). |
| Palette position | `ColorCycle` | Colours in drawing order, restarted every frame. |
| The colour bar's panel | `Grid._color_bar_panel`, `Grid._color_bar_drawing` | As drawn this frame (presses do not go through it), and its last primitives, for reuse. |
| Legend entries, the legend's panel | `Grid._legend_entries`, `Grid._legend_panel`, `Grid._legend_layout` | The labelled calls of the frame (emptied every frame), and the panel as drawn, which presses do not go through; the last primitives of its panel and swatches, for reusing their coverage. |
| The previous frame's plots and grid | `PlotCache`, `Grid._grid_scene` | Reusing unchanged plots (exactly: checked against the functions' values). |
| Coverage of reused primitives | `CoverageCache` (renderer) | Blending reused plots without rasterising them again. |
| Fonts, label surfaces, marker stamps | renderer caches | Speed only; `clear_caches()` frees them. |

Everything else is a pure function of these and the user's calls, which is
what makes the property tests possible.

## Retired experiments

Three larger features were built on top of the plane, tried, and removed
again, because they went against the [philosophy](../philosophy.md#explored-and-retired)
(a second way to draw, or a second language next to Python). They are not
planned to come back, and new features should not reintroduce them in
another form. Their code is still in the history, for reference:

| Feature | What it was | Last commit with it |
|---|---|---|
| Graphs | `pgp.Graph`: a Desmos-like list of named items written as lines of maths (`"y = a sin(x - P.x)"`), names usable in any order, errors per item, saving and loading JSON | `3c76ab5` ("persistent plot objects"); first version in `8aaee2b` ("support for expressions") |
| The expression language | `core/expr`: a safe parser and compiler from text to NumPy functions, with exact derivatives (`f'(x)`), behind `pgp.expression` | `3c76ab5` |
| Retained plot objects | `pygridplot/objects`: `pgp.Function`, `pgp.Point`, ... kept by the plane (`plane.add`, `plane.draw` on what was then `Plane`) and changed through their attributes | `cc116e8` ("updated docs and tests"), Python only; with text support in `3c76ab5` |

What they taught is kept: a draggable point no longer keeps its plane
alive (`Grid.draggable` gives points a weak reference to their plane),
and the derivative of a function is undefined wherever the function is
(`Jet.constant`).
