# PyGridPlot

An interactive Cartesian plane for [pygame-ce](https://pyga.me/): smooth panning
and zooming, and an adaptive grid with clean tick labels at every zoom level,
in the spirit of the Desmos graph view, but driven entirely from Python code.

PyGridPlot is a *library, not a framework*: your program owns the main loop,
and the plane is an object you call each frame.

```python
import pygame
import pygridplot as pgp

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
plane = pgp.Plane()          # fills the window and follows resizes
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)   # drag to pan, scroll to zoom
    plane.draw_grid(screen)
    pygame.display.flip()
    clock.tick(60)
```

For a quick look: `python -c "import pygridplot; pygridplot.show()"`.

## Status

Milestones 0–2 of `PROJECT_PLAN.md` are done: the view, interaction and the
adaptive grid. Function, point and polygon plotting come next.

## Controls

| Input | Action |
|---|---|
| Left-drag | Pan (the grabbed point stays exactly under the cursor) |
| Mouse wheel / trackpad | Zoom about the cursor (eased) |
| Arrow keys | Pan (eased) |
| `+` / `-` | Zoom about the middle (eased) |
| `Home` / `0` | Reset the view |

## How the grid stays smooth

* **Exact spacings.** Grid spacings live on the 1-2-5 ladder and are stored
  as integers (`mantissa × 10^exponent`). Line positions come from integer
  indices, never from accumulated floats, and labels are formatted from exact
  decimals, so `0.30000000000000004` and `-0` cannot appear.
* **Cross-fading.** When the zoom crosses from one spacing to the next, both
  grids are drawn and blended in a narrow band of zoom. Labels fade
  sequentially, so the two sets never crowd each other. Snapping with
  hysteresis is available too (`GridTheme(transition="snap")`).
* **No popping labels.** Labels that would collide are thinned out (every
  2nd/5th/10th), and the ones about to go fade out first. Labels near other
  labels fade instead of overlapping, and labels slide in and out across
  the edges.
* **Pinned labels.** When an axis is off-screen its labels stick to the
  nearest edge. Near the edge they are pushed along continuously instead of
  jumping.
* **Crisp pixels.** Lines are drawn as whole-pixel rectangles (no blur), and
  labels are aligned to the same pixels as their lines, so they never wobble.
* **Eased input.** Wheel and keyboard movement follows a critically damped
  spring: notched wheels glide like trackpads, with no overshoot.
* **Precision limits.** Zoom is clamped where `float64` stops being able to
  place pixels accurately (see `ViewLimits`).

## Theming

Themes are frozen dataclasses. Derive variations with `dataclasses.replace`:

```python
from dataclasses import replace
import pygridplot as pgp

theme = replace(pgp.DARK, axis_color=(120, 190, 255), target_major_px=120)
plane = pgp.Plane(theme=theme)
```

The default label font is the bundled DejaVu Sans, so labels look the same on
every platform. Set `label_font` to a font file path or a system font name to
change it.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                               # unit, property and headless rendering tests
ruff check . && ruff format --check .
mypy                                 # strict
```

Visual tools:

```bash
python examples/grid_playground.py   # interactive, with a debug overlay and hotkeys
python examples/zoom_sweep.py        # automated zoom/pan stress test
python scripts/render_gallery.py --out renders/gallery   # PNGs of tricky views
```

Layout:

```
src/pygridplot/
├── core/              # pure maths: no pygame imports (enforced by a test)
│   ├── coords.py      # ScreenRect, Bounds, type aliases
│   ├── pixels.py      # pixel-snapping conventions
│   ├── view.py        # View: transforms, pan/zoom, limits
│   ├── easing.py      # critically damped spring for eased motion
│   ├── nice.py        # exact 1-2-5 spacing ladder
│   ├── spacing.py     # choosing spacings while zooming (fade / snap)
│   ├── ticks.py       # grid line candidates from integer indices
│   ├── gridlines.py   # painting grid lines and axes
│   ├── labels.py      # exact tick label formatting
│   ├── ticklabels.py  # which labels to show: thinning, fades, notation
│   ├── layout.py      # label placement, pinning, collisions
│   ├── primitives.py  # backend-neutral drawing primitives
│   ├── theme.py       # GridTheme, LIGHT, DARK
│   └── grid.py        # build_grid(): one frame of grid primitives
├── render/            # pygame backend (fonts, text, colours, drawing)
├── interaction.py     # events -> pan/zoom
├── plane.py           # public Plane class
└── quick.py           # show()
```

The bundled label font is DejaVu Sans; see
`src/pygridplot/assets/fonts/LICENSE-DejaVu.txt`.
