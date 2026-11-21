# Architecture

## Design principles

1. **Library, not framework.** The user owns the main loop. The plane is an
   object they call (`handle_event`, `draw_grid`, `plot`, ...); it never
   takes control, never sleeps and never polls events itself.
2. **Immediate mode.** Everything is recomputed from the view every frame.
   There is no retained scene to keep in sync, so animation and interaction
   come for free. The only state kept between frames is small and explicit:
   the view, the easing springs, snap hysteresis, the colour cycle's
   position, the framing (home view, pending fits) and caches. The retained
   layers, [objects](graphs.md#the-objects) and [graphs](graphs.md#the-graph),
   sit *on top* of the immediate API and draw through it, so nothing below
   them knows they exist.
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
   graph/                  named lists of objects, text, JSON (uses objects/)
        │
   objects/                persistent functions, curves, points (use plane.py's API)
        │
   plane.py ── interaction.py ── dragging.py ── quick.py     pygame-specific integration
     │      │
     │      └── render/        the backend: primitives -> pixels
     │              │
     └──────────────┴── core/  pure maths: never imports anything above it
                         ├── grid/     the adaptive grid and its labels
                         ├── plot/     plotted objects -> primitives
                         ├── handles/  draggable points (uses grid/ and plot/)
                         └── expr/     the expression language (uses plot/ for derivatives)
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
  on both, `core.expr` (the expression language) on `core.plot` alone (for
  exact derivatives), and nothing in `core` imports either. Anything both
  grid and plot need (such as the label text and marker vocabulary in
  `primitives.py`) belongs in a foundation module.

Above the plane, `pygridplot.objects` uses only the public plane API (the
same calls a program makes) and `core`, and `pygridplot.graph` builds on
the objects. The plane keeps objects through a small protocol of its own
(`Drawable`), so the plane, the renderer and `core` never import either,
and objects never import graphs; the architecture tests check both.

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
├── __init__.py          The public API: Plane, styles, themes, configs, View, Graph, show
├── plane.py             Plane: the façade tying view, theme, controls and renderer together
├── interaction.py       pygame events -> pan/zoom (anchored drag, eased wheel and keys)
├── dragging.py          pygame events -> draggable points (hit test, drag, hover, cursor)
├── quick.py             show(): a ready-made main loop
├── py.typed             Marks the package as typed (PEP 561)
├── objects/             Persistent objects, drawn through the plane's API
│   ├── base.py          PlotObject (two-phase drawing, errors), StyledObject, Owner
│   ├── values.py        Formula (a function, a number or text), argument checks
│   ├── curves.py        Function, ParametricCurve, PolarCurve, ImplicitCurve
│   ├── points.py        Point: fixed, derived or draggable; handles kept in step
│   ├── shapes.py        Points, Polyline, Polygon
│   └── errors.py        PlotWarning, and warning at the program's own call
├── graph/               Named lists of objects, with text and saving
│   ├── graph.py         Graph: the list, names, registered functions, drawing, saving
│   ├── definitions.py   Parameter, Definition: the items only a graph has
│   ├── namespace.py     Compiling objects: dependency order, cycles, errors, bindings
│   ├── lines.py         Reading "y = x^2" and friends into an object (Graph.add)
│   ├── view.py          GraphView: the ranges a graph is shown at
│   ├── serialize.py     JSON: writing, strict reading, the format's version
│   └── errors.py        GraphError
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
│   │   ├── gridlines.py     AxisLines (per-line opacities) -> LineBatch primitives, axes
│   │   ├── formatting.py    Exact number formatting -> LabelText (with superscripts)
│   │   ├── layout.py        Label placement: push-and-pin, collision fades
│   │   ├── scene.py         build_grid(): the pipeline, into a GridScene
│   │   └── theme.py         GridTheme, LIGHT, DARK
│   ├── plot/            Plotted objects -> primitives
│   │   ├── style.py         LineStyle, PointStyle, PolygonStyle, ColorCycle (palette)
│   │   ├── evaluation.py    Calling user functions robustly (errors/complex -> NaN)
│   │   ├── sampling.py      Adaptive sampling of y = f(x): jumps, poles, domain edges
│   │   ├── curves.py        Adaptive sampling of parametric curves on screen
│   │   ├── jets.py          Taylor arithmetic: exact derivatives through NumPy code
│   │   ├── derivatives.py   derivative(), derivatives(): jets, point by point, numerical
│   │   ├── implicit.py      Implicit curves F(x, y) = 0: marching squares (contourpy), checked
│   │   ├── geometry.py      Point input; clipping polylines and polygons
│   │   ├── memo.py          PlotCache: reusing unchanged plots between frames, exactly
│   │   └── builders.py      One function per kind of plot -> clipped primitives
│   ├── handles/         Draggable points
│   │   ├── point.py         DraggablePoint: position, constraint, style, state
│   │   ├── constraints.py   Free, locked and on-curve movement, with bounds
│   │   ├── tracking.py      Following a curve with the cursor, branch by branch
│   │   ├── labels.py        The coordinate label: rounded to a pixel, placed
│   │   └── drawing.py       Markers, hover ring and label as primitives
│   └── expr/            The expression language: text -> NumPy functions
│       ├── lexer.py         Text -> tokens (numbers, names with primes, operators)
│       ├── parser.py        Tokens -> syntax trees with source spans; the grammar
│       ├── builtins.py      The built-in constants and functions; NaN-aware logic
│       ├── compiler.py      Trees -> closures over ufuncs; bindings, folding, suggestions
│       ├── function.py      expression(): text to a function in one call
│       └── errors.py        ExpressionError, with a caret under the mistake
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

### `plane.handle_event(event)`

`dragging.DragController` sees every event first: a press on a
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

### `plane.draw_grid(surface)`

1. **Rect.** An automatic plane takes the surface's size; the first time,
   `ViewFraming.settle()` redoes any provisional fit at the real size.
2. **Easing.** `InteractionController.update()` advances the springs by the
   real elapsed time (an exact, frame-rate-independent solution) and
   applies the change to the `View`, zooming about the anchor.
3. **Palette.** `ColorCycle.reset()` restarts the palette.
4. **Grid.** `core.grid.scene.build_grid(view, theme, renderer, state)`
   builds a `GridScene`; [The grid](grid.md) explains each step.
5. **Draw.** `PygameRenderer.draw_grid` fills the background, draws the line
   batches as crisp 1-pixel rectangles and blits cached label surfaces.

### `plane.plot(surface, f)` and the other plotting methods

1. **Style.** `style.with_overrides()` merges the keyword arguments into the
   style, and `ColorCycle.resolve()` picks the colour (the next palette
   colour if none was given).
2. **Primitives.** A builder in `core/plot/builders.py` turns the object
   into clipped screen-space primitives: `function_stroke` (via
   `sampling.sample_function`), `parametric_stroke` and `polar_stroke` (via
   `curves.sample_curve`), `implicit_stroke` (via
   `implicit.sample_implicit`), `polyline_stroke`, `polygon_shapes` or
   `point_markers`. [Plotting](plotting.md) explains the samplers.
3. **Draw.** `Plane._draw()` clips the surface to the plane's rectangle and
   dispatches each primitive to the renderer (`draw_stroke`, `draw_fill`,
   `draw_markers`); [Rendering](rendering.md) explains how they become
   pixels.

### `plane.draw(surface)` with objects

`draw_grid` first, then the objects added with `plane.add`, in two phases
(see [Drawing in two phases](graphs.md#drawing-in-two-phases)): every
object's `update` (a draggable point catches up with this frame's drag; a
graph compiles if anything changed), then every object's `render` (one of
the plotting methods above per visible object), then all the draggable
handles on top with `draw_draggables`. From the plane's point of view the
objects are a program like any other.

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
| Draggable points | `DraggablePoint`, listed by the `Plane` | Positions the user changes; the grabbed point and hover in `DragController`. |
| Home view, pending fits | `ViewFraming` | `reset_view`, and fitting ranges once the size is known. |
| Snap levels | `GridState` | Hysteresis in `"snap"` mode only (fades are stateless). |
| Palette position | `ColorCycle` | Colours in drawing order, restarted every frame. |
| The previous frame's plots and grid | `PlotCache`, `Plane._grid_scene` | Reusing unchanged plots (exactly: checked against the functions' values). |
| Coverage of reused primitives | `CoverageCache` (renderer) | Blending reused plots without rasterising them again. |
| Fonts, label surfaces, marker stamps | renderer caches | Speed only; `clear_caches()` frees them. |
| Objects and their palette slots | `Plane._objects`, each object | What to draw every frame, and its colour. |
| A draggable `Point`'s handles | `Point._handles` (weak, by plane) | Keeping the point and the plane's handle in step. |
| Objects by name, compiled namespace | `Graph` | Names, text compiled with them, and saving. |

Everything else is a pure function of these and the user's calls, which is
what makes the property tests possible.
