# Testing and tools

```bash
pytest                                   # everything, headless, about 40 s
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
| `tests/core/plot/` | Evaluation, both samplers, geometry, styles and the builders. |
| `tests/render/` | Rasterising and compositing (pure NumPy plus small surfaces). |
| `tests/` | Integration: the renderer, `Grid`, plotting end to end, interaction, the examples, the docs and the architecture rules. |

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
  opaque than one), on linear and log axes, axis titles included, and
  titles stay inside the plane;
- panning by whole pixels moves every grid line, axis title, text and
  plotted pixel rigidly (bars, histograms and error bars included);
- adding samples to a histogram never moves a bin, only adds to its count;
- reusing plots between frames never changes a single pixel, through any
  sequence of changes;
- a tiny zoom step changes every line's and label's opacity only a little
  (no pops), on linear and log axes;
- clipping keeps shapes in their box, splitting a stroke never changes it,
  and fills preserve polygon areas;
- curves stay within the sampling tolerance of the true curve.

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
module under `core/`: core imports only the standard library, NumPy and
itself; the foundation modules never import `core.grid` or `core.plot`; and
those two never import each other.

**Example smoke tests** (`test_examples.py`) run every program in
`examples/` headless for a few frames with a scripted event queue that
presses some of the keys the examples use, zooms, and closes the window.

**Package tests** (`test_package.py`) check the version string and that
importing PyGridPlot with the original pygame (a stand-in without
`pygame.IS_CE`, in a subprocess) fails with advice. CI also builds the
package and runs the suite from the source archive; see
[Releasing](releasing.md).

**Documentation tests** (`test_docs.py`):

- the API reference must match what `scripts/build_api_docs.py` generates
  from the docstrings, so after changing a public signature or docstring,
  run the script and commit the result;
- every ```` ```python ```` block in the README and `docs/` must run. The
  blocks of a page run in order in one namespace (with `np`, `math`,
  `pygame`, `pgp`, `screen`, `surface` and `grid` predefined), and main
  loops end after a few frames. Use ```` ```py ```` for signature listings
  and fragments that are not meant to run;
- every relative link and `#anchor` must resolve (pictures included);
- the [examples page](../user/examples.md) must match what
  `scripts/render_examples.py --page` writes from the examples'
  docstrings, and every example must be on it, with a picture.

## Visual and performance tools

Some qualities can only be judged by eye, or are about speed:

- `python scripts/render_gallery.py --out DIR` renders the gallery of
  tricky views (deep zooms, far offsets, axes at the edges, cross-fades,
  themes, functions, derivatives, shapes, log axes, ranges, curves,
  implicit curves, draggable points, inequalities and areas, vector and
  slope fields and arrows, integrals and series, sequences, data plots)
  to PNG files plus a contact sheet `_sheet.png`. Render before and after a
  change and compare; for refactors, the views must stay pixel-identical
  (see [comparing the gallery](#comparing-the-gallery-before-and-after)).
  `--only NAME ...` renders a subset.
- `python scripts/render_examples.py` renders every example headless (time
  comes from the frame count, so the pictures are the same on every run)
  into `docs/images/examples/`, the README's montage, and the examples
  page; `NAME ...` renders some, `--page` writes only the page.
- `python scripts/benchmark.py` prints the median time of the main
  per-frame operations (grids, functions, curves, implicit curves,
  inequalities and areas, vector and slope fields, arrows, points, fills,
  dragging, derivatives, integrals, antiderivatives, series and sequences,
  and idle
  frames reused by the cache); `--only TEXT` selects cases and
  `--repeat N` steadies the numbers. Compare on the same machine, over a
  few runs.
- `python scripts/zoom_sweep.py --save-frames DIR` dumps frames of an
  automated zoom and pan, for checking smoothness frame by frame.
- `python scripts/grid_playground.py` toggles theme settings, log axes and
  plot overlays live, and shows the scale, spacings and timings.

### Comparing the gallery before and after

Render a baseline from the last commit in a separate worktree (with
`PYTHONPATH=src` it uses that checkout's package, not the editable
install), render the working tree, and compare every view pixel by pixel:

```bash
git worktree add /tmp/pgp-baseline HEAD
(cd /tmp/pgp-baseline && PYTHONPATH=src SDL_VIDEODRIVER=dummy \
  "$OLDPWD"/venv/bin/python scripts/render_gallery.py --out /tmp/gallery-before)
SDL_VIDEODRIVER=dummy venv/bin/python scripts/render_gallery.py --out /tmp/gallery-after
git worktree remove --force /tmp/pgp-baseline
```

```py
import pathlib, numpy as np, pygame
before, after = pathlib.Path("/tmp/gallery-before"), pathlib.Path("/tmp/gallery-after")
for path in sorted(before.glob("[0-9]*.png")):
    a = pygame.surfarray.array3d(pygame.image.load(path)).astype(int)
    b = pygame.surfarray.array3d(pygame.image.load(after / path.name)).astype(int)
    if a.shape != b.shape or (a != b).any():
        print(path.name, np.abs(a - b).max(), (np.abs(a - b).max(axis=2) > 0).sum())
```

It prints each view that differs, with the largest change in a channel
and the number of pixels changed. New features add views at the end; look
at them as images. A change meant to alter the picture should differ only
where intended.

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
