# Testing and tools

```bash
pytest                                   # everything, headless, about 25 s
pytest tests/core/grid                   # one area
pytest -k log                            # by name
ruff check . && ruff format --check .    # lint (Google docstrings required) and formatting
mypy                                     # strict type checking of src/
python scripts/build_api_docs.py         # regenerate docs/user/reference.md
```

## Layout of the tests

The tests mirror the package:

| Folder | Tests |
|---|---|
| `tests/core/` | The foundations: view, transforms, framing, easing, coordinates and pixels. |
| `tests/core/grid/` | Spacing, ticks, formatting, layout, log axes and the whole grid (`test_scene.py`). |
| `tests/core/plot/` | Evaluation, the samplers, jets and derivatives, geometry, the plot cache, styles and the builders. |
| `tests/core/handles/` | Draggable points: constraints, curve tracking, labels. |
| `tests/core/expr/` | The expression language: lexer, parser, compiler, built-ins, `pgp.expression`. |
| `tests/objects/` | Objects: drawing like the plane's methods, attributes, colours, errors, fixed, derived and draggable points. |
| `tests/graph/` | Graphs: reading lines, names and errors, drawing, draggable points, saving and loading, `show`. |
| `tests/render/` | Rasterising and compositing (pure NumPy plus small surfaces). |
| `tests/` | Integration: the renderer, `Plane`, plotting end to end, interaction, the examples, the docs and the architecture rules. |

`tests/conftest.py` sets SDL's dummy video and audio drivers (so nothing
opens a window) and provides `FakeMeasurer`, a text measurer with fixed
character sizes (7 px wide, 9 px tall digit box, 3 px superscript
overflow), so layout can be tested exactly without fonts. The
`pygame_display` fixture gives tests that need real surfaces a small dummy
display.

## Kinds of tests

**Unit tests** check behaviour and edge cases function by function, for
example that `format_value` never writes `-0`, or that a fit keeps a range
visible with at most a pixel to spare.

### Property tests

[Hypothesis](https://hypothesis.readthedocs.io/) generates random views
(centres, scales, axis kinds) and checks the guarantees the design rests
on. The important ones:

- grid lines stay in the plane, and opacities stay in `(0, 1]`;
- the major spacing stays near the target;
- labels never overlap (two overlapping labels together are never more
  opaque than one), on linear and log axes;
- panning by whole pixels moves every grid line and every plotted pixel
  rigidly;
- reusing plots between frames never changes a single pixel, through any
  sequence of changes;
- a tiny zoom step changes every line's and label's opacity only a little
  (no pops), on linear and log axes;
- clipping keeps shapes in their box, splitting a stroke never changes it,
  and fills preserve polygon areas;
- curves stay within the sampling tolerance of the true curve;
- compiled expressions (random trees of operators and functions) agree
  with the same computation written in NumPy, and so do their exact
  derivatives; no text makes the parser raise anything but an
  `ExpressionError`;
- any graph saved as JSON reads back exactly the same.

A failing property test prints the smallest example it found; Hypothesis
remembers it (in `.hypothesis/`) and tries it first next time. When an
example exposes a *documented* limitation rather than a bug (for instance
the instant fixed/scientific notation switch on linear axes), the test
skips those cases with `assume(...)` and a comment saying why.

**Pixel tests** render into real surfaces and check exact results: grid
pixels are exactly theme colours (no blur), strokes have the right width,
markers keep their size when zooming, polygons scale with the zoom, alpha
blends correctly onto transparent layers, drawing respects clip areas.
Allow `2 / 255` for antialiased pixels after a pan: float round-off can
move a value across an 8-bit rounding boundary.

**Architecture tests** (`test_architecture.py`) parse the imports of every
module under `core/`: core imports only the standard library, NumPy,
contourpy and itself; the foundation modules never import `core.grid` or
`core.plot`; those two never import each other; and `core.handles` and
`core.expr` import only what their entries in `ALLOWED_SUBPACKAGES` list;
and the layers above the plane (`objects`, then `graph`) are imported only
from above (`LAYERS_ABOVE_THE_PLANE`).

**Equivalence tests** check that a layer built on another adds nothing it
should not: objects and graphs draw pixel for pixel what the same plane
calls draw (`tests/objects/test_objects.py`, `tests/graph/test_graph.py`),
and a graph read back from JSON draws the same pixels as the original.

**Example smoke tests** (`test_examples.py`) run every program in
`examples/` headless for a few frames with a scripted event queue that
presses some of the keys the examples use, zooms, and closes the window.

**Documentation tests** (`test_docs.py`):

- the API reference must match what `scripts/build_api_docs.py` generates
  from the docstrings, so after changing a public signature or docstring,
  run the script and commit the result;
- every ```` ```python ```` block in the README and `docs/` must run. The
  blocks of a page run in order in one namespace (with `np`, `math`,
  `pygame`, `pgp`, `screen`, `surface` and `plane` predefined), and main
  loops end after a few frames. Use ```` ```py ```` for signature listings
  and fragments that are not meant to run;
- every relative link and `#anchor` must resolve.

## Visual and performance tools

Some qualities can only be judged by eye, or are about speed:

- `python scripts/render_gallery.py --out DIR` renders 54 tricky views
  (deep zooms, far offsets, axes at the edges, cross-fades, themes,
  functions, derivatives, shapes, log axes, ranges, curves, implicit curves
  and draggable points) to PNG files plus a contact sheet `_sheet.png`.
  Render before and after a change and compare. `--only NAME ...` renders
  a subset.
- `python scripts/benchmark.py` prints the median time of the main
  per-frame operations (grids, functions, curves, implicit curves, points,
  fills, dragging, derivatives); `--only TEXT` selects cases and
  `--repeat N` steadies the numbers. Compare on the same machine, over a
  few runs.
- `python examples/zoom_sweep.py --save-frames DIR` dumps frames of an
  automated zoom and pan, for checking smoothness frame by frame.
- `python examples/grid_playground.py` toggles theme settings, log axes and
  plot overlays live, and shows the scale, spacings and timings.

## The checklist

Before a change is finished:

1. `pytest` passes, with new tests for new behaviour (and a property test
   when the change touches a guarantee such as rigid panning).
2. `ruff check .`, `ruff format --check .` and `mypy` are clean.
3. Public API changes: docstrings updated, `python scripts/build_api_docs.py`
   run, and the user guide updated; examples still run.
4. Visual changes: the gallery looked at before and after.
5. Performance-sensitive changes: `scripts/benchmark.py` before and after.
6. `CHANGELOG.md` updated.
