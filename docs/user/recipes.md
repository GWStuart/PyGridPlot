# Recipes

Worked examples of common tasks. Each snippet shows the part that goes
inside your main loop (after `grid.draw(screen)`) unless it says
otherwise; `screen` is the display surface and `grid` your plane.

- [Showing the mouse position](#showing-the-mouse-position)
- [Marking a curve under the mouse](#marking-a-curve-under-the-mouse)
- [Shading the area under a curve](#shading-the-area-under-a-curve)
- [A tangent line at a draggable point](#a-tangent-line-at-a-draggable-point)
- [Velocity and acceleration vectors](#velocity-and-acceleration-vectors)
- [Plotting data](#plotting-data)
  ([a histogram that fills in](#a-histogram-that-fills-in),
  [a live signal](#a-live-signal))
- [Animating](#animating)
- [Adding a legend](#adding-a-legend)
- [Saving a screenshot](#saving-a-screenshot)
- [Drawing onto a transparent layer](#drawing-onto-a-transparent-layer)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Showing the mouse position

Convert the mouse position to world coordinates and write them beside the
cursor with [`grid.text`](shapes-and-text.md#text):

```python
mx, my = pygame.mouse.get_pos()
if grid.rect.contains_point(mx, my):
    x, y = grid.to_world(mx, my)
    grid.text(screen, f"({x:.3f}, {y:.3f})", (x, y), anchor="top-left", gap=12)
```

`grid.rect.contains_point` keeps the readout to the plane when it only
covers part of the window. For text in a fixed place on the screen (a
caption), render it with a pygame font instead.

## Marking a curve under the mouse

Evaluate the function at the mouse's x, mark the point, and draw the tangent
line through it:

```python
def f(x):
    return np.sin(x) + 0.3 * x

grid.plot(screen, f)
x, _ = grid.to_world(*pygame.mouse.get_pos())
y = f(x)
slope = (f(x + 1e-6) - f(x - 1e-6)) / 2e-6          # numerical derivative
grid.polyline(screen, [(x - 2, y - 2 * slope), (x + 2, y + 2 * slope)],
               color="gray", width=1.5)
grid.points(screen, (x, y), color="crimson", edge_color="white", radius=5)
```

The point keeps its size at every zoom, and the tangent's length is in world
units, so it grows as you zoom in.

## Shading the area under a curve

Use [`fill_between`](regions-and-fields.md#areas-between-graphs), which shades down
to the x axis by default. Draw it before the curve, so the line stays on
top:

```python
grid.fill_between(screen, np.sin, domain=(0, np.pi), color="steelblue")
grid.plot(screen, np.sin, color="steelblue")
```

The area meets the curve exactly at every zoom, and stops at jumps and
asymptotes. On a log y axis, `0` lies infinitely far down, so the area
fills to the bottom of the plane.

## A tangent line at a draggable point

Put a [draggable point](draggable-points.md) on the graph of `f` (a curve
`(t, f(t))`) and draw the tangent line from the
[derivative](functions.md#derivatives) at the point. Create the point once,
before the loop:

```python
f = np.cos
point = grid.draggable(curve=(lambda t: t, f, (-10, 10)), t=0.5)   # before the loop

value, slope = pgp.derivatives(f, point.x, 1)
grid.plot(screen, f)
grid.plot(screen, lambda x: value + slope * (x - point.x), width=1.5)
grid.draw_controls(screen)
```

## Velocity and acceleration vectors

For a point moving along a parametric curve `(x(t), y(t))`, the velocity
is `(x'(t), y'(t))` and the acceleration `(x''(t), y''(t))`. Put a
draggable point on the curve and draw both as
[arrows](shapes-and-text.md#arrows) from it, with the exact
[derivatives](functions.md#derivatives):

```python
x = lambda t: 3 * np.cos(t)
y = lambda t: 2 * np.sin(2 * t)
mover = grid.draggable(curve=(x, y, (0, 2 * np.pi)), t=1.0)   # before the loop

t = mover.t
(_, vx, ax), (_, vy, ay) = pgp.derivatives(x, t, 2), pgp.derivatives(y, t, 2)
grid.parametric(screen, x, y, (0, 2 * np.pi), color="gray", width=1.5)
grid.arrow(screen, (mover.x, mover.y), (mover.x + vx, mover.y + vy), color="crimson")
grid.arrow(screen, (mover.x, mover.y), (mover.x + ax / 4, mover.y + ay / 4),
            color="seagreen")                           # a quarter of the acceleration
grid.draw_controls(screen)
```

## Plotting data

Measurements are points, optionally joined by a polyline. NumPy arrays go
in directly:

```python
xs = np.arange(0, 10, 0.5)
ys = 2 + 0.5 * xs + np.random.default_rng(1).normal(0, 0.3, xs.size)
data = np.column_stack((xs, ys))

grid.polyline(screen, data, color="gray", width=1)
grid.points(screen, data, marker="square", radius=3.5)
fit = np.polyfit(xs, ys, 1)
grid.plot(screen, lambda x: np.polyval(fit, x), color="crimson", width=1.5)
```

To show all of the data at first, fit the view to it once, before the loop:

```python
grid.set_view(x_range=(xs.min() - 1, xs.max() + 1), y_range=(ys.min() - 1, ys.max() + 1))
```

Reading data from a file is ordinary NumPy, for example
`data = np.loadtxt("measurements.csv", delimiter=",")`. Gaps in a polyline
come from rows of NaN. Measurements with uncertainties take error bars,
`grid.points(screen, data, y_error=errors)`; see
[Plotting data](data.md#which-method-to-use) for scatter plots coloured by a
value, step plots, bar charts and histograms.

### A histogram that fills in

Samples added every frame (a simulation, a Monte Carlo estimate) go into a
list or array that grows; the histogram's bins are anchored, so the bars
grow in place instead of shifting:

```python
rng = np.random.default_rng(7)
samples = np.zeros(0)                                  # before the loop

samples = np.append(samples, rng.uniform(size=(50, 12)).sum(axis=1) - 6)
grid.histogram(screen, samples, width=0.25, density=True)
grid.plot(screen, lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))
```

(Sums of twelve uniform numbers minus six: the central limit theorem at
work.) A density is the right comparison as the number of samples grows;
counts grow with it.

### A live signal

A signal sampled over time is a polyline through the latest samples, kept
in a `collections.deque`; to follow it, set the view's x range every
frame:

```python
from collections import deque

history = deque(maxlen=300)                            # before the loop
t = 0.0

t += 1 / 60
history.append((t, math.sin(3 * t) + 0.1 * math.sin(40 * t)))
grid.polyline(screen, list(history))
grid.set_view(x_range=(t - 5, t))
```

`set_view` every frame stops the user panning along x; leave it out to let
them look back.

## Animating

Draw something that depends on time. Measure time once per frame and use it
in the functions you plot:

```python
t = pygame.time.get_ticks() / 1000            # seconds since pygame.init()

grid.plot(screen, lambda x: np.sin(x - t))    # a travelling wave
angle = 0.5 * t
grid.points(screen, (2 * np.cos(angle), 2 * np.sin(angle)))
grid.polar(screen, lambda a: 2 + 0.5 * np.sin(5 * a + t))
```

The lambdas above use `t` straight away, inside the same frame, so late
binding is not a problem here. It is when you create lambdas in a loop
(see [Late binding](functions.md#late-binding)). See `examples/animation.py`,
`examples/curves.py` and `examples/implicit.py`.

## Adding a legend

Give the plots a `label=` and draw the [legend](shapes-and-text.md#legends) after
them:

```python
a = 1.5                                       # a value read from a slider, say
grid.plot(screen, np.sin, label="sin x")
grid.plot(screen, lambda x: a * np.cos(x), label=f"{a:g} cos x")
grid.legend(screen, corner="top-left")
```

Labels are ordinary strings, so they can show values, and the legend
follows them frame by frame. For something drawn another way (with pygame,
say), a labelled call that draws nothing still adds an entry: an empty
`grid.points(screen, [], color=..., label=...)` shows a marker,
`grid.polyline(screen, [], color=..., label=...)` a line.

## Saving a screenshot

To save the plane, at the screen's resolution or a higher one, use
`grid.save` after drawing a frame; [Saving pictures and videos](export.md)
covers it, and recording videos:

```python
grid.save("plot.png", scale=2)    # the plane alone, twice the size
```

Everything is drawn onto ordinary pygame surfaces, so saving the whole
window, with your own drawing on it, is pygame's job:

```python
pygame.image.save(screen, "window.png")
```

To render an image without a window (on a server, say), set the
environment variable `SDL_VIDEODRIVER=dummy` before `pygame.init()`, create
a display of any size, and draw onto your own surface:

```python
image = pygame.Surface((1200, 800))
poster = pgp.Grid(x_range=(-2 * np.pi, 2 * np.pi))
poster.draw(image)
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
overlay = pgp.Grid(theme=replace(pgp.LIGHT, background=None))
overlay.draw(layer)
overlay.plot(layer, np.sin)
screen.blit(layer, (0, 0))
```

Antialiased edges are blended correctly into surfaces with per-pixel alpha,
so there is no dark fringe when the layer is composited.

## Performance

A typical frame (the grid and a few functions) takes a few milliseconds,
and much less while nothing changes: plots that have not changed are
[reused](plotting.md#reusing-unchanged-plots) automatically. To keep it
that way:

- **Write functions with NumPy.** Vectorised functions are called once per
  batch of points; scalar-only ones once per point, which is much slower.
- **Mind very long curves.** Drawing costs roughly half a millisecond per
  thousand pixels of line on screen, so a spiral with dozens of turns or a
  dense spirograph costs a few milliseconds each. Functions that oscillate
  infinitely fast (`sin(1/x)` near 0) are the most expensive of all.
- **Implicit curves cost more than functions**: a few milliseconds for a
  simple curve, more for curves that fill the plane. Draw several levels
  of one function with a single call (`grid.implicit(screen, F, levels)`),
  and write `F` with NumPy: scalar-only functions are much slower here.
- **Big translucent fills** cost about 4 ms for a polygon covering most of
  a large window, and so do regions (`inequality`) and areas
  (`fill_between`) of that size.
- **Vector fields** cost about 7 ms on a large window, and about twice that
  for the moments while arrows fade in during a zoom; a larger `spacing`
  draws fewer arrows. Slope fields are cheaper (about 2.5 ms). Many arrows
  in one `arrow` call are far cheaper than one call each.
- **Thin lines are cheaper** than wide ones.
- **Share a renderer** between planes (`Grid(renderer=...)`), so they
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
[Late binding](functions.md#late-binding)).

**A function draws nothing.** It may fail for every x (a wrong name, for
example): failures are silently left out of the plot. Call it yourself once
with a NumPy array to see the error, for example `f(np.linspace(0, 1, 5))`.

**Eased zooming stops halfway.** Eased movement advances in `grid.draw`. If
your program only redraws when events arrive, use
`InteractionConfig(smoothing=0)`.

**The window looks blurry or small on a high-DPI screen.** Sizes are in
physical pixels; PyGridPlot does not scale for high-DPI displays. Use a
larger window, `label_size` and line widths.

**`ImportError` or odd rendering.** Make sure `pygame-ce` is installed and
the original `pygame` package is not (they share the import name).

## Examples

The [examples page](examples.md) shows every program in the repository's
`examples/` folder, with a picture and what it shows: functions and
calculus, curves and regions, fields, points, sliders and shapes, data,
log axes, animation and a plane inside a game. Each is one short file to
copy and start from.
