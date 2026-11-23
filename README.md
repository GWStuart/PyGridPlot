# PyGridPlot

An interactive Cartesian plane for [pygame-ce](https://pyga.me/), in the spirit
of the Desmos graph view but driven entirely from Python code. It gives you:

- an adaptive grid with clean tick labels at every zoom level;
- linear or logarithmic axes, set up from a centre and scale or from x and y
  ranges;
- smooth, eased panning and zooming;
- robust, antialiased plotting of functions (and their exact derivatives),
  parametric, polar and implicit curves, points, polylines and polygons;
- shaded areas between graphs, and regions where inequalities hold;
- arrows, and vector and slope fields;
- sequences, as stems and markers: a list of values, a spectrum, or a
  function of `n`;
- data: scatter plots coloured and sized by a value with a colour bar,
  error bars, step plots, bar charts (stacked, floating, horizontal),
  histograms, box plots and violin plots, and names for categories along
  an axis;
- calculus to plot: integrals and antiderivatives, Taylor polynomials and
  Fourier series;
- draggable points: free, locked to an axis, or running along a curve;
- sliders for values that are not positions, in a panel that stays put;
- axis titles and arrowheads, a legend of the labelled plots, and text
  written at points of the plane.

PyGridPlot is a *library, not a framework*: your program owns the main loop,
and the plane, a `pgp.Grid`, is an object you call each frame.

## Installing

```bash
pip install pygridplot
```

PyGridPlot needs Python 3.10 or newer, and installs
[pygame-ce](https://pyga.me/), NumPy and contourpy. It uses pygame
*community edition*, which installs under the same name as the original
`pygame`: if that is installed, uninstall it first (`pip uninstall pygame`).

## A first program

```python
import numpy as np
import pygame
import pygridplot as pgp

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
grid = pgp.Grid()                  # fills the window and follows resizes
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        grid.handle_event(event)    # drag to pan, scroll to zoom

    grid.draw(screen)
    grid.plot(screen, np.tan)       # asymptotes are not joined up
    grid.points(screen, [(0, 0), (1, 1)], edge_color="white")
    grid.polygon(screen, [(-3, -1), (-1, -1), (-2, 1)])
    pygame.display.flip()
    clock.tick(60)
```

For a quick look without writing a loop: `pgp.show(np.sin, lambda x: x**2)`.
Views can be given like a graphing calculator's window settings, and either
axis can be logarithmic:

```python
grid = pgp.Grid(x_range=(-2, 8), y_range=(-1, 3))           # each axis fills its range
grid = pgp.Grid(log_x=True, log_y=True)                     # log-log
pgp.show(np.exp, log_y=True, x_range=(0, 10))               # semi-log, quick mode
```

## Highlights

- **A grid that always looks right.** Spacings are exact 1-2-5 steps that
  cross-fade while you zoom, labels never overlap or pop, and labels stick to
  the edges when an axis is off-screen.
- **Log axes done properly.** Decades thin out when zoomed far out, classic
  log paper appears at ordinary zooms, and finer subdivisions fade in as you
  zoom in, until a deeply zoomed log axis looks linear. Power laws and
  exponentials plot as exactly straight lines.
- **Robust function plotting.** Functions can be NumPy or plain `math`, and may
  raise errors or return NaN. Jumps and poles (`floor`, `1/x`, `tan`, `gamma`)
  are never joined by vertical lines, and domain edges are exact.
- **Parametric and polar curves.** `grid.parametric(screen, x, y, t_range)`
  and `grid.polar(screen, r)` sample adaptively on screen, so loops, cusps
  and spirals stay smooth at any zoom, and curves end exactly where they
  become undefined.
- **Implicit curves and contour lines.** `grid.implicit(screen, F, c)`
  draws any curve `F(x, y) = c`: loops, crossings and separate pieces,
  accurate to a fraction of a pixel, with no false lines at poles or jumps.
  Several levels at once make a contour map.
- **Areas and inequalities.** `grid.fill_between(screen, f, g)` shades
  between two graphs, stopping cleanly at jumps and asymptotes, and
  `grid.inequality(screen, lambda x, y: x**2 + y**2 < 4)` shades the
  region where a condition holds, holes and all, with edges accurate to a
  fraction of a pixel. Conditions combine with `&`, `|` and `~`.
- **Vector and slope fields.** `grid.vector_field(screen, lambda x, y: (-y, x))`
  and `grid.slope_field(screen, lambda x, y: x - y)` draw arrows or
  segments on a lattice that pans rigidly and keeps its density while you
  zoom, with new arrows growing in smoothly rather than popping. Arrows show
  direction, or magnitude too with `scale=`, and stay tangent to the flow
  lines on log axes. `grid.arrow(screen, start, end)` draws any arrows,
  one or thousands at once.
- **Sequences.** `grid.sequence(screen, terms)` draws the terms `aₙ` of a
  list or array (a Fourier series' `amplitudes` give its spectrum), or of a
  function of `n`, drawn at every visible integer, even zoomed far out,
  where the terms are thinned smoothly without hiding alternating signs.
- **Exact derivatives of any order.** `pgp.derivative(f, n)` is the
  `n`-th derivative of a NumPy function, exact to rounding (Taylor-mode
  automatic differentiation, no dependencies), with clean jumps at kinks
  and steps; plot it or evaluate it, for tangent lines at a draggable
  point.
- **Integrals and series.** `pgp.integral(f, a, b)` is a definite integral
  to about 10 digits (adaptive Gauss-Kronrod quadrature), with infinite
  limits, jumps and integrable singularities, and NaN rather than a wrong
  number when it diverges. `pgp.antiderivative(f, a)` is a function to
  plot, `pgp.taylor(f, a, n)` a Taylor polynomial from the exact
  derivatives, and `pgp.fourier(f, n)` a partial Fourier sum, with its
  coefficients. Together with draggable points: areas between two points,
  Riemann sums, Gibbs' overshoot as you add harmonics.
- **Draggable points.** `a = grid.draggable((1, 2))` creates a point the
  user can drag; read `a.x` and `a.y` every frame to drive your plots.
  Points can be free, locked to one axis or kept on a parametric curve,
  which they follow through crossings without jumping.
- **Sliders.** `n = grid.slider("n", (1, 20), step=1)` creates a slider in
  a panel in a corner of the plane, showing `n = 5`, which stays put while
  you pan and zoom; read `n.value` every frame. Steps are exact (`step=1`
  gives `int`s), tracks can be geometric (`log=True`), and the mouse wheel
  moves a slider a step.
- **Data plots.** `grid.points(screen, data, y_error=errors)` draws
  measurements with error bars, and takes a colour and a size per point;
  `grid.polyline(screen, data, step="post")` draws steps;
  `grid.bars(screen, heights)` a bar chart; and `grid.histogram(screen,
  samples, density=True)` bins raw samples in anchored bins (they never
  shift as samples arrive), with an automatic 1-2-5 width, ready to compare
  with a density drawn by `plot`. `pgp.color_map(values, "viridis")` colours
  points by a value, and `grid.color_bar(...)` is its key;
  `grid.box_plot` and `grid.violin_plot` compare groups of samples;
  `grid.draw(x_labels=["a", "b", ...])` names categories along an axis;
  and `pgp.data_range(values)` gives the view a range that shows them.
- **Annotations.** `grid.draw(screen, x_title="t (s)", y_title="v")`
  titles the axes at their ends, keeping the titles on screen when the axes
  leave it, and a theme with `show_arrows=True` ends the axes in
  arrowheads. Every plotting method takes a `label=`, and
  `grid.legend(screen)` lists the labelled plots of the frame beside
  samples of their lines, fills, markers and arrows. `grid.text(screen,
  "max", (x, y), anchor="bottom", gap=8)` writes text that moves with its
  point.
- **Fast when little changes.** Plots that have not changed since the
  previous frame are reused automatically and exactly (their functions are
  checked at every sampled point), so idle frames cost a fraction of a
  full redraw, with no change to your code.
- **Smooth rendering.** Curves use an antialiased round pen, so joints never
  darken. Lines are crisp, and points keep their pixel size at every zoom.
- **Themeable.** Light and dark themes are built in, and every colour, width,
  spacing and font is a theme field.

## Documentation

The [documentation](docs/index.md) starts with the
[philosophy](docs/philosophy.md) behind the library (one way to do things,
done well; Python first; a library, not a framework; immediate mode), then
has two parts:

- **[User guide](docs/user/index.md)**: [getting started](docs/user/getting-started.md),
  [the plane](docs/user/the-plane.md), [interaction](docs/user/interaction.md),
  [plotting](docs/user/plotting.md), [draggable points](docs/user/draggable-points.md),
  [log axes](docs/user/log-axes.md),
  [styles and themes](docs/user/styles-and-themes.md), [recipes](docs/user/recipes.md),
  and the [API reference](docs/user/reference.md) generated from the code.
- **[Developer guide](docs/dev/index.md)**: [architecture](docs/dev/architecture.md),
  [the grid](docs/dev/grid.md), [plotting](docs/dev/plotting.md),
  [rendering](docs/dev/rendering.md), [testing and tools](docs/dev/testing.md)
  and [extending PyGridPlot](docs/dev/extending.md).

## Examples

```bash
python examples/functions.py        # the function test set and hard cases (Space)
python examples/curves.py           # animated parametric and polar curves (Space)
python examples/implicit.py         # implicit curves and contour lines (Space)
python examples/regions.py          # inequalities and areas between graphs (Space)
python examples/fields.py           # vector and slope fields, arrows (Space)
python examples/draggable.py        # draggable points: free, locked, on curves (Space)
python examples/sliders.py          # sliders: continuous, stepped, integer, geometric (Space)
python examples/derivatives.py      # tangents, Taylor polynomials, osculating circles (Space)
python examples/integrals.py        # areas, antiderivatives, Riemann sums, improper integrals (Space, R)
python examples/series.py           # Fourier series with a slider, and their spectrum (Space)
python examples/sequences.py        # sequences, limits, partial sums, the logistic map (Space)
python examples/annotations.py      # axis titles, arrowheads, legends and text (Space)
python examples/data.py             # scatter, colour bars, error bars, steps, bars, histograms, box and violin plots (Space)
python examples/shapes.py           # points, polylines, polygons, fill rules
python examples/log_axes.py         # log-log, semi-log and range-based grids (Space, X, Y)
python examples/animation.py        # everything moving, every frame
python examples/grid_playground.py  # live theme toggles and timings; X/Y log axes
```

More in [`examples/`](examples/): `basic_grid.py`, `many_functions.py`,
`embedded_rect.py`, `dark_theme.py`, `zoom_sweep.py`; the
[recipes](docs/user/recipes.md#examples) list them all.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                                  # headless unit, property and pixel tests
ruff check . && ruff format --check .
mypy                                    # strict
python scripts/render_gallery.py --out renders/gallery   # visual regression PNGs
python scripts/benchmark.py             # per-frame timings
python scripts/build_api_docs.py        # regenerate the API reference
```

See the [developer guide](docs/dev/index.md) for the architecture and
conventions, and the [changelog](CHANGELOG.md) for what has changed.

## Licence

PyGridPlot is free software: you can redistribute it and modify it under
the terms of the [GNU Lesser General Public License](LICENSE), version 2.1
or (at your option) any later version, the licence pygame-ce uses. In
short, programs that use PyGridPlot may be under any licence, while
changed versions of PyGridPlot itself stay under the LGPL; the licence
text has the terms. The bundled label font, DejaVu Sans, comes with its own
permissive licence
([`LICENSE-DejaVu.txt`](src/pygridplot/assets/fonts/LICENSE-DejaVu.txt)).

Copyright © 2026 GWStuart.
