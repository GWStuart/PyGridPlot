# Recipes

Worked examples of common tasks. Each snippet shows the part that goes
inside your main loop (after `plane.draw_grid(screen)`) unless it says
otherwise; `screen` is the display surface and `plane` your plane.

- [Showing the mouse position](#showing-the-mouse-position)
- [Marking a curve under the mouse](#marking-a-curve-under-the-mouse)
- [Shading the area under a curve](#shading-the-area-under-a-curve)
- [Plotting data](#plotting-data)
- [Animating](#animating)
- [Adding a legend](#adding-a-legend)
- [Saving a screenshot](#saving-a-screenshot)
- [Drawing onto a transparent layer](#drawing-onto-a-transparent-layer)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Showing the mouse position

Convert the mouse position to world coordinates and draw it with a pygame
font. Create the font once, before the loop:

```python
font = pygame.font.Font(None, 22)          # before the loop

mx, my = pygame.mouse.get_pos()
if plane.rect.contains_point(mx, my):
    x, y = plane.to_world(mx, my)
    text = font.render(f"({x:.3f}, {y:.3f})", True, (30, 30, 30))
    screen.blit(text, (mx + 12, my + 12))
```

`plane.rect.contains_point` keeps the readout to the plane when it only
covers part of the window.

## Marking a curve under the mouse

Evaluate the function at the mouse's x, mark the point, and draw the tangent
line through it:

```python
def f(x):
    return np.sin(x) + 0.3 * x

plane.plot(screen, f)
x, _ = plane.to_world(*pygame.mouse.get_pos())
y = f(x)
slope = (f(x + 1e-6) - f(x - 1e-6)) / 2e-6          # numerical derivative
plane.polyline(screen, [(x - 2, y - 2 * slope), (x + 2, y + 2 * slope)],
               color="gray", width=1.5)
plane.points(screen, (x, y), color="crimson", edge_color="white", radius=5)
```

The point keeps its size at every zoom, and the tangent's length is in world
units, so it grows as you zoom in.

## Shading the area under a curve

Build a polygon from the curve's points and the x axis. Draw it before the
curve, so the line stays on top:

```python
a, b = 0.0, np.pi
xs = np.linspace(a, b, 200)
area = np.column_stack((np.r_[a, xs, b], np.r_[0.0, np.sin(xs), 0.0]))
plane.polygon(screen, area, color="steelblue", width=0, fill_opacity=0.3)
plane.plot(screen, np.sin, color="steelblue")
```

200 points is plenty for a smooth outline on screen. On a log y axis the
points at `y = 0` lie infinitely far down, so the area fills to the bottom
of the plane.

## Plotting data

Measurements are points, optionally joined by a polyline. NumPy arrays go
in directly:

```python
xs = np.arange(0, 10, 0.5)
ys = 2 + 0.5 * xs + np.random.default_rng(1).normal(0, 0.3, xs.size)
data = np.column_stack((xs, ys))

plane.polyline(screen, data, color="gray", width=1)
plane.points(screen, data, marker="square", radius=3.5)
fit = np.polyfit(xs, ys, 1)
plane.plot(screen, lambda x: np.polyval(fit, x), color="crimson", width=1.5)
```

To show all of the data at first, fit the view to it once, before the loop:

```python
plane.set_view(x_range=(xs.min() - 1, xs.max() + 1), y_range=(ys.min() - 1, ys.max() + 1))
```

Reading data from a file is ordinary NumPy, for example
`data = np.loadtxt("measurements.csv", delimiter=",")`. Gaps in a polyline
come from rows of NaN.

## Animating

Draw something that depends on time. Measure time once per frame and use it
in the functions you plot:

```python
t = pygame.time.get_ticks() / 1000            # seconds since pygame.init()

plane.plot(screen, lambda x: np.sin(x - t))   # a travelling wave
angle = 0.5 * t
plane.points(screen, (2 * np.cos(angle), 2 * np.sin(angle)))
plane.polar(screen, lambda a: 2 + 0.5 * np.sin(5 * a + t))
```

The lambdas above use `t` straight away, inside the same frame, so late
binding is not a problem here. It is when you create lambdas in a loop
(see [Late binding](plotting.md#late-binding)). See `examples/animation.py`
and `examples/curves.py`.

## Adding a legend

Plots take palette colours in drawing order, so a legend can read them from
the theme:

```python
font = pygame.font.Font(None, 22)
names = ["sin x", "cos x", "x / 3"]
for name, func in zip(names, (np.sin, np.cos, lambda x: x / 3)):
    plane.plot(screen, func)

for i, name in enumerate(names):
    color = pygame.Color(plane.theme.palette[i % len(plane.theme.palette)])
    y = 16 + 22 * i
    pygame.draw.line(screen, color, (16, y + 7), (40, y + 7), 3)
    screen.blit(font.render(name, True, (40, 40, 40)), (48, y))
```

Draw the legend after the plots so it is on top. A translucent panel behind
it (a `pygame.Surface` with `pygame.SRCALPHA`, filled with an alpha below
255) keeps it readable over the grid; `examples/functions.py` does this.

## Saving a screenshot

Everything is drawn onto ordinary pygame surfaces, so saving is pygame's
job. After drawing a frame:

```python
pygame.image.save(screen, "plot.png")
```

To render an image without a window (on a server, say), set the
environment variable `SDL_VIDEODRIVER=dummy` before `pygame.init()`, create
a display of any size, and draw onto your own surface:

```python
image = pygame.Surface((1200, 800))
poster = pgp.Plane(x_range=(-2 * np.pi, 2 * np.pi))
poster.draw_grid(image)
poster.plot(image, np.sin, width=3)
pygame.image.save(image, "poster.png")
```

## Drawing onto a transparent layer

A theme without a background draws only lines and labels, so the plane can
sit on top of other content, or onto a transparent surface that you
composite later:

```python
from dataclasses import replace

layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)   # fully transparent
overlay = pgp.Plane(theme=replace(pgp.LIGHT, background=None))
overlay.draw_grid(layer)
overlay.plot(layer, np.sin)
screen.blit(layer, (0, 0))
```

Antialiased edges are blended correctly into surfaces with per-pixel alpha,
so there is no dark fringe when the layer is composited.

## Performance

A typical frame (the grid and a few functions) takes a few milliseconds. To
keep it that way:

- **Write functions with NumPy.** Vectorised functions are called once per
  batch of points; scalar-only ones once per point, which is much slower.
- **Mind very long curves.** Drawing costs roughly half a millisecond per
  thousand pixels of line on screen, so a spiral with dozens of turns or a
  dense spirograph costs a few milliseconds each. Functions that oscillate
  infinitely fast (`sin(1/x)` near 0) are the most expensive of all.
- **Big translucent fills** cost about 4 ms for a polygon covering most of
  a large window.
- **Thin lines are cheaper** than wide ones.
- **Share a renderer** between planes (`Plane(renderer=...)`), so they
  share font and label caches; `renderer.clear_caches()` frees them.

`scripts/benchmark.py` in the repository times all of these.

## Troubleshooting

**Nothing appears.** Draw the grid first and the plots after it; the grid
fills the background and would cover anything drawn before it. Remember
`pygame.display.flip()` at the end of each frame.

**Colours change from frame to frame.** Palette colours are given out in
drawing order. If you skip a plot in some frames, the ones after it change
colour; give them explicit colours instead.

**All my functions look the same.** Functions created in a loop need the
loop variable bound as a default argument: `lambda x, k=k: x**k` (see
[Late binding](plotting.md#late-binding)).

**A function draws nothing.** It may fail for every x (a wrong name, for
example): failures are silently left out of the plot. Call it yourself once
with a NumPy array to see the error, for example `f(np.linspace(0, 1, 5))`.

**Eased zooming stops halfway.** Eased movement advances in `draw_grid`. If
your program only redraws when events arrive, use
`InteractionConfig(smoothing=0)`.

**The window looks blurry or small on a high-DPI screen.** Sizes are in
physical pixels; PyGridPlot does not scale for high-DPI displays. Use a
larger window, `label_size` and line widths.

**`ImportError` or odd rendering.** Make sure `pygame-ce` is installed and
the original `pygame` package is not (they share the import name).

## Examples

The `examples/` folder has a runnable program for every topic:

| Example | Shows |
|---|---|
| `basic_grid.py` | The smallest program: a plane in a window. |
| `functions.py` | The function test set and hard cases (`Space` switches sets). |
| `curves.py` | Animated parametric and polar curves (`Space` switches, `P` pauses). |
| `shapes.py` | Points, polylines, polygons and fill rules. |
| `log_axes.py` | Log-log, semi-log and range-based planes (`X`/`Y` toggle log). |
| `animation.py` | Everything moving, every frame. |
| `many_functions.py` | Many functions at once, for performance. |
| `embedded_rect.py` | A plane in part of the window. |
| `dark_theme.py` | The dark theme. |
| `grid_playground.py` | Live theme toggles, log axes and timings (`F1` shows the keys). |
| `zoom_sweep.py` | An automated zoom and pan, optionally saving frames. |
