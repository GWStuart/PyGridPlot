# PyGridPlot

An interactive Cartesian plane for [pygame-ce](https://pyga.me/), in the spirit
of the Desmos graph view but driven entirely from Python code. It gives you:

- an adaptive grid with clean tick labels at every zoom level;
- smooth, eased panning and zooming;
- robust, antialiased plotting of functions, points, polylines and polygons.

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

## Highlights

- **A grid that always looks right.** Spacings are exact 1-2-5 steps that
  cross-fade while you zoom, labels never overlap or pop, and labels stick to
  the edges when an axis is off-screen.
- **Robust function plotting.** Functions can be NumPy or plain `math`, and may
  raise errors or return NaN. Jumps and poles (`floor`, `1/x`, `tan`, `gamma`)
  are never joined by vertical lines, and domain edges are exact.
- **Smooth rendering.** Curves use an antialiased round pen, so joints never
  darken. Lines are crisp, and points keep their pixel size at every zoom.
- **Themeable.** Light and dark themes are built in, and every colour, width,
  spacing and font is a theme field.

## Documentation

- **[User guide](docs/user-guide.md)**: how to use everything, with every
  parameter and its default.
- **[Developer guide](docs/developer-guide.md)**: how the code is organised,
  how a frame flows through it, and how to extend it.

## Examples

```bash
python examples/functions.py        # the function test set and hard cases (Space)
python examples/shapes.py           # points, polylines, polygons, fill rules
python examples/animation.py        # everything moving, every frame
python examples/grid_playground.py  # live theme toggles and timings; P cycles plots
```

More in [`examples/`](examples/): `basic_grid.py`, `many_functions.py`,
`embedded_rect.py`, `dark_theme.py`, `zoom_sweep.py`.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                                  # headless unit, property and pixel tests
ruff check . && ruff format --check .
mypy                                    # strict
python scripts/render_gallery.py --out renders/gallery   # visual regression PNGs
```

See the [developer guide](docs/developer-guide.md) for the architecture and
conventions. The bundled label font is DejaVu Sans; see
`src/pygridplot/assets/fonts/LICENSE-DejaVu.txt`.
