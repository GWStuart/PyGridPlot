# PyGridPlot

An interactive Cartesian plane for [pygame-ce](https://pyga.me/), in the spirit
of the Desmos graph view but driven entirely from Python code. It gives you:

- an adaptive grid with clean tick labels at every zoom level;
- linear or logarithmic axes, set up from a centre and scale or from x and y
  ranges;
- smooth, eased panning and zooming;
- robust, antialiased plotting of functions (and their exact derivatives),
  parametric, polar and implicit curves, points, polylines and polygons;
- draggable points: free, locked to an axis, or running along a curve;
- objects: functions, curves and points made once and changed through
  their attributes (`f = plane.add(pgp.Function(np.sin))`, `f.color = "red"`).

PyGridPlot is a *library, not a framework*: your program owns the main loop,
and the plane is an object you call each frame.

```python
import numpy as np
import pygame
import pygridplot as pgp

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
plane = pgp.Plane()                  # fills the window and follows resizes
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)    # drag to pan, scroll to zoom

    plane.draw_grid(screen)
    plane.plot(screen, np.tan)       # asymptotes are not joined up
    plane.points(screen, [(0, 0), (1, 1)], edge_color="white")
    plane.polygon(screen, [(-3, -1), (-1, -1), (-2, 1)])
    pygame.display.flip()
    clock.tick(60)
```

For a quick look without writing a loop: `pgp.show(np.sin, lambda x: x**2)`.
Views can be given like a graphing calculator's window settings, and either
axis can be logarithmic:

```python
plane = pgp.Plane(x_range=(-2, 8), y_range=(-1, 3))         # each axis fills its range
plane = pgp.Plane(log_x=True, log_y=True)                   # log-log
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
- **Parametric and polar curves.** `plane.parametric(screen, x, y, t_range)`
  and `plane.polar(screen, r)` sample adaptively on screen, so loops, cusps
  and spirals stay smooth at any zoom, and curves end exactly where they
  become undefined.
- **Implicit curves and contour lines.** `plane.implicit(screen, F, c)`
  draws any curve `F(x, y) = c`: loops, crossings and separate pieces,
  accurate to a fraction of a pixel, with no false lines at poles or jumps.
  Several levels at once make a contour map.
- **Exact derivatives of any order.** `pgp.derivative(f, n)` is the
  `n`-th derivative of a NumPy function, exact to rounding (Taylor-mode
  automatic differentiation, no dependencies), with clean jumps at kinks
  and steps; plot it or evaluate it, for tangent lines and Taylor
  polynomials at a draggable point.
- **Draggable points.** `a = plane.draggable((1, 2))` creates a point the
  user can drag; read `a.x` and `a.y` every frame to drive your plots.
  Points can be free, locked to one axis (sliders) or kept on a parametric
  curve, which they follow through crossings without jumping.
- **Objects you make once.** Every plotting method has an object that
  keeps its arguments as attributes, read every frame: `pgp.Function`,
  `pgp.ParametricCurve`, `pgp.Point`, ... Add them to the plane and draw
  everything with one call; Python closures make objects follow each
  other, and functions can be called and differentiated.

  ```python
  a = plane.add(pgp.Point((1, 2), draggable=True))
  f = plane.add(pgp.Function(lambda x: a.y * np.sin(x - a.x)))  # follows a
  f.color = "crimson"                         # change it whenever you like
  plane.draw(screen)                          # each frame: grid, f, then a
  ```
- **Fast when little changes.** Plots that have not changed since the
  previous frame are reused automatically and exactly (their functions are
  checked at every sampled point), so idle frames cost a fraction of a
  full redraw, with no change to your code.
- **Smooth rendering.** Curves use an antialiased round pen, so joints never
  darken. Lines are crisp, and points keep their pixel size at every zoom.
- **Themeable.** Light and dark themes are built in, and every colour, width,
  spacing and font is a theme field.

## Documentation

The [documentation](docs/index.md) has two parts:

- **[User guide](docs/user/index.md)**: [getting started](docs/user/getting-started.md),
  [the plane](docs/user/the-plane.md), [interaction](docs/user/interaction.md),
  [plotting](docs/user/plotting.md), [draggable points](docs/user/draggable-points.md),
  [objects](docs/user/objects.md),
  [log axes](docs/user/log-axes.md),
  [styles and themes](docs/user/styles-and-themes.md), [recipes](docs/user/recipes.md),
  and the [API reference](docs/user/reference.md) generated from the code.
- **[Developer guide](docs/dev/index.md)**: [architecture](docs/dev/architecture.md),
  [the grid](docs/dev/grid.md), [plotting](docs/dev/plotting.md),
  [draggable points](docs/dev/dragging.md),
  [objects](docs/dev/objects.md),
  [rendering](docs/dev/rendering.md), [testing and tools](docs/dev/testing.md)
  and [extending PyGridPlot](docs/dev/extending.md).

## Examples

```bash
python examples/functions.py        # the function test set and hard cases (Space)
python examples/curves.py           # animated parametric and polar curves (Space)
python examples/implicit.py         # implicit curves and contour lines (Space)
python examples/draggable.py        # draggable points, sliders, points on curves (Space)
python examples/derivatives.py      # tangents, Taylor polynomials, osculating circles (Space)
python examples/two_ways.py         # immediate mode and objects, side by side
python examples/objects.py          # a picture made of objects (Space, W)
python examples/shapes.py           # points, polylines, polygons, fill rules
python examples/log_axes.py         # log-log, semi-log and range-based planes (Space, X, Y)
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
conventions, and the [roadmap](ROADMAP.md) for ideas for the future. The bundled label font is DejaVu Sans; see
`src/pygridplot/assets/fonts/LICENSE-DejaVu.txt`.
