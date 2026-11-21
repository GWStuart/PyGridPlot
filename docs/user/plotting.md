# Plotting

Every plotting method draws onto a surface, on top of whatever is already
there, and is called every frame after `draw_grid`. Everything is redrawn
from the current view each time, so plots follow panning and zooming
exactly, and changing what you draw from one frame to the next animates it.

| Method | Draws |
|---|---|
| [`plot(surface, f)`](#functions) | The graph of `y = f(x)`. |
| [`parametric(surface, x, y, t_range)`](#parametric-curves) | The curve `(x(t), y(t))`. |
| [`polar(surface, r)`](#polar-curves) | The curve `r = r(θ)`. |
| [`implicit(surface, F)`](#implicit-curves) | The curve `F(x, y) = 0` (or `= c`). |
| [`points(surface, points)`](#points) | Markers that keep their size in pixels. |
| [`polyline(surface, points)`](#polylines) | Vertices joined by straight lines. |
| [`polygon(surface, points)`](#polygons) | A filled shape with an outline. |

All of them accept `color`, `opacity` and a reusable `style`; see
[Styles and themes](styles-and-themes.md). Objects drawn without a colour
take the theme's [palette](styles-and-themes.md#the-palette) colours in
order. Everything is clipped to the plane's rectangle.

## Functions

```py
plane.plot(surface, func, *, color=None, width=None, opacity=None,
           domain=None, style=None, sampling=None)
```

```python
plane.plot(screen, np.sin)
plane.plot(screen, lambda x: x**2 / 4, color="crimson", width=4)
plane.plot(screen, np.sqrt, domain=(0, 4))           # only 0 <= x <= 4
plane.plot(screen, math.gamma)                         # scalar-only functions work too
```

`func` maps x to y. It can be:

- **vectorised**, written with NumPy (`np.sin`, `lambda x: x**2`). This is
  the fastest kind: it is called with arrays of x values.
- **scalar-only**, written with `math` or with `if` statements. It is
  detected automatically (the first call with an array fails or returns the
  wrong shape) and then called one x at a time.

It does not need to be well behaved. Wherever it raises an error, returns
NaN or infinity, or returns a complex number with a real imaginary part,
there is simply a gap; a frame never crashes because of a plotted function.
The plot is also faithful where functions are hard to draw:

- **Jumps** (`np.floor`) and **poles** (`1/x`, `np.tan`, `math.gamma`) are
  never joined by vertical lines, and each branch of a pole runs off the
  edge of the plane.
- **Domain edges** are found exactly: `np.sqrt` starts at the origin, and
  `np.log` dives off the bottom of the plane as x approaches 0.
- **Fine detail** is sampled adaptively, so curves stay smooth however far
  you zoom in, and narrow features are not missed.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Line colour. |
| `width` | `2.5` | Line width in pixels (fractional widths are fine). |
| `opacity` | `1.0` | Line opacity, 0 to 1. |
| `domain` | `None` | Only plot where `domain[0] <= x <= domain[1]`. |
| `style` | `None` | A reusable `LineStyle`; the keyword arguments override it. |
| `sampling` | the plane's | A `SamplingConfig` for this call. |

### Sampling settings

Sampling is automatic and you will rarely need to change it. If you do, a
`SamplingConfig` can be passed to `plot` or set for the whole plane
(`Plane(sampling=...)` or `plane.sampling = ...`):

| Field | Default | Meaning |
|---|---|---|
| `samples_per_px` | `1.0` | Initial samples per pixel column. |
| `tolerance_px` | `0.25` | Largest allowed distance between the curve and the drawn line. |
| `max_rounds` | `8` | Refinement depth (jumps are located to 1/256 of a pixel). |
| `max_samples` | `16384` | Budget of evaluations per function per frame. |
| `jump_px` | `1.0` | Smallest unresolved rise treated as a discontinuity. |

### Late binding

When creating functions in a loop, bind the loop variable as a default
argument:

```python
powers = [lambda x, k=k: x**k for k in range(1, 4)]   # x, x², x³
for f in powers:
    plane.plot(screen, f)
```

Without the `k=k`, all three functions would use the last `k`, because a
lambda looks its variables up when it is called, not when it is created. The
same applies to animation variables captured by lambdas inside a loop.

## Parametric curves

```py
plane.parametric(surface, x, y, t_range, *, color=None, width=None,
                 opacity=None, style=None, sampling=None)
```

```python
plane.parametric(screen, np.cos, np.sin, (0, 2 * np.pi))               # unit circle
plane.parametric(screen, lambda t: t - np.sin(t), lambda t: 1 - np.cos(t), (0, 20))
plane.parametric(screen, 2.0, lambda t: t, (-1, 1))                     # the segment x = 2
```

The curve is the point `(x(t), y(t))` as `t` runs over `t_range`, which you
always give (there is no natural default). `x` and `y` are written like the
functions given to `plot`, vectorised or not, and allowed to fail for some
`t`; either can also be a plain number. Points where a coordinate is
undefined are left out, and the curve ends exactly where it becomes
undefined. Jumps and poles are not joined up: `x = tan t` breaks into
branches, each running off the edge of the plane.

Curves are sampled adaptively *on screen*, so tight turns, cusps, loops and
spirals stay smooth at every zoom level, on linear and log axes alike.

## Polar curves

```py
plane.polar(surface, r, theta_range=(0, 2π), *, color=None, width=None,
            opacity=None, style=None, sampling=None)
```

```python
plane.polar(screen, lambda a: 2 * np.cos(3 * a))                  # three-petal rose
plane.polar(screen, lambda a: 1 + np.cos(a))                      # cardioid
plane.polar(screen, lambda a: 0.1 * a, (0, 10 * np.pi))           # spiral, 5 turns
plane.polar(screen, 1.5)                                           # circle of radius 1.5
```

The point at angle `θ` and radius `r(θ)` is `(r cos θ, r sin θ)`. Angles are
in radians, anticlockwise from the positive x axis, and a negative radius
lies on the opposite side of the origin, as usual in mathematics.
`theta_range` defaults to one full turn; spirals, and roses such as
`cos(θ / 3)`, need more. Where `r` is undefined, the curve has a gap that
ends exactly at the edge (`r = sqrt(cos 2θ)`, the lemniscate, reaches the
origin).

### Curve sampling settings

Parametric and polar curves share a `CurveSampling` configuration, passed
as `sampling=`:

| Field | Default | Meaning |
|---|---|---|
| `initial_samples` | `256` | Evenly spaced parameter values sampled first. |
| `tolerance_px` | `0.25` | Largest allowed distance between the curve and the drawn line. |
| `max_segment_px` | `24` | Longest segment drawn without checking in between. A segment that stays longer than this however much it is refined is treated as a jump. |
| `max_rounds` | `12` | Refinement depth beyond what the zoom level needs. |
| `max_samples` | `16384` | Budget of evaluations per curve per frame. |

The initial samples must be dense enough that no feature hides between two
of them, so a curve with a very large parameter range (hundreds of spiral
turns, say) may need more `initial_samples`.

## Implicit curves

```py
plane.implicit(surface, func, level=0, *, color=None, width=None,
               opacity=None, style=None, sampling=None)
```

```python
plane.implicit(screen, lambda x, y: x**2 + y**2, 4)                   # circle, radius 2
plane.implicit(screen, lambda x, y: x**3 + y**3 - 6 * x * y)          # folium of Descartes
plane.implicit(screen, lambda x, y: (x**2 + y**2)**2 - 8 * (x**2 - y**2))  # lemniscate
plane.implicit(screen, lambda x, y: x - np.sin(y))                    # x = sin y
```

An implicit curve is every point where `func(x, y)` equals `level` (0 by
default), so any equation can be drawn by moving everything to one side:
`x² + y² = 4` is `func = x² + y²` with `level = 4`, or `x² + y² - 4` with
the default level. The curve can have any shape: closed loops, several
separate pieces, branches that cross, parts that are not a function of
`x` or of `y`.

`func` takes two arguments, written like the functions given to `plot`:
NumPy (fast) or plain `math` (slower), allowed to raise errors or return
NaN where it is undefined. Where `func` changes sign without passing
through the level, at a pole or a jump, no line is drawn, and the curve
ends where `func` becomes undefined:

```python
plane.implicit(screen, lambda x, y: y - np.tan(x))      # branches, no asymptotes
plane.implicit(screen, lambda x, y: y - np.floor(x))    # steps, no risers
plane.implicit(screen, lambda x, y: y - math.sqrt(x))   # starts at x = 0
```

The curve is found again every frame on a grid of samples 2 pixels apart,
refined where it crosses the grid, so it is accurate to a small fraction
of a pixel at any zoom, on linear and log axes alike.

### Contour lines

Give a sequence of levels to draw one curve for each, all in the same
style, like the height lines of a map. This is much faster than one call
per level, because `func` is evaluated only once:

```python
def height(x, y):
    return np.sin(x) + np.cos(y)

plane.implicit(screen, height, np.linspace(-1.5, 1.5, 7), width=1.5)
plane.implicit(screen, height, 0, width=3)              # emphasise one level
```

### What implicit curves cannot do

- **`func` must change sign across the curve.** A curve where it only
  touches the level is not drawn: `(x - y)**2` is never negative, so
  `(x - y)**2 = 0` draws nothing (and nor does a region where `func`
  equals the level throughout). Write `x - y` instead. For the same reason
  a curve may break where two of its branches touch without crossing.
- **Features smaller than about 2 pixels may be missed**, such as a loop
  less than 3 pixels across, or two curves closer together than a pixel or
  two. Zoom in to see them, or sample more finely (below).
- **A curve stops within a pixel or two of where `func` becomes
  undefined**, rather than exactly at the edge as for `plot`.
- **A function of `x` is better drawn with `plot`**, and a curve with a
  known parametrisation with `parametric`: both are faster and treat
  domain edges exactly. Use `implicit` for the curves that have no simpler
  form.

### Implicit sampling settings

Pass an `ImplicitSampling` as `sampling=` to trade speed for detail:

| Field | Default | Meaning |
|---|---|---|
| `resolution_px` | `2.0` | Spacing of the sampling grid in pixels. Features smaller than this may be missed; halving it quadruples the work near the curve. |
| `coarse_factor` | `4` | The grid used to find where the curve might be is this many times coarser (1, 2, 4, 8 or 16; 1 samples everywhere). |
| `refine_steps` | `3` | Steps spent checking each crossing of the grid and moving it onto the curve. `0` skips the check, so poles and jumps would be drawn as lines. |
| `max_samples` | `250000` | Budget of evaluations per call per frame. Beyond it the grid is made coarser. |

```python
fine = pgp.ImplicitSampling(resolution_px=1.0)
plane.implicit(screen, lambda x, y: np.sin(x * y) - 0.5, sampling=fine)
```

The budget keeps slow functions usable: a plain `math` function costs
about 0.4 microseconds per evaluation (NumPy is tens of times faster), so
the default budget bounds a call at about 100 ms even in the worst case, a
curve that winds across the whole plane. Dense contour maps
on large windows can reach the budget with NumPy functions; raising
`max_samples` (to `10**6`, say) keeps them at full resolution.

## Points

```py
plane.points(surface, points, *, color=None, radius=None, marker=None,
             edge_color=None, edge_width=None, opacity=None, style=None)
```

```python
plane.points(screen, (1, 2))                                     # one point
plane.points(screen, [(0, 0), (1, 1), (2, 4)], marker="diamond")
xs = np.linspace(-3, 3, 25)
plane.points(screen, np.column_stack((xs, np.sin(xs))), edge_color="white")
```

`points` is a single point `(x, y)`, a sequence of points, or an `(N, 2)`
array (use `np.column_stack((xs, ys))` for separate arrays). Points with a
NaN coordinate are skipped. Markers keep the same size in pixels at every
zoom level, and are positioned to a quarter of a pixel, so they glide
smoothly while panning.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Marker colour. |
| `radius` | `4.5` | Size in pixels (other shapes have the same area as the circle). |
| `marker` | `"circle"` | `"circle"`, `"square"`, `"diamond"`, `"cross"` or `"plus"`. |
| `edge_color` | `None` | Outline colour, drawn *outside* the marker. A background-coloured outline (such as `"white"`) sets points apart from the curves they sit on. |
| `edge_width` | `1.5` | Outline thickness in pixels (only with `edge_color`). |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `style` | `None` | A reusable `PointStyle`. |

## Polylines

```py
plane.polyline(surface, points, *, color=None, width=None, opacity=None,
               closed=False, style=None)
```

```python
plane.polyline(screen, [(0, 0), (1, 2), (3, 1)])
plane.polyline(screen, [(0, 0), (1, 0), (1, 1)], closed=True, width=1.5)
plane.polyline(screen, [(0, 0), (1, 1), (np.nan, np.nan), (2, 0), (3, 1)])  # two pieces
```

The vertices are joined in order by straight lines, with round joins and
ends, antialiased at any width. A vertex with a NaN coordinate breaks the
line, like lifting the pen. `closed=True` joins the last vertex back to the
first. Vertices are in world units, so a polyline grows and shrinks with the
zoom; only the width is in pixels. Takes the `LineStyle` fields.

## Polygons

```py
plane.polygon(surface, points, *, color=None, width=None, opacity=None,
              fill=None, fill_color=None, fill_opacity=None, fill_rule=None,
              style=None)
```

```python
plane.polygon(screen, [(-3, -1), (-1, -1), (-2, 1)])                 # translucent triangle
plane.polygon(screen, [(0, 0), (2, 0), (2, 1), (0, 1)], fill_opacity=0.6, width=0)
```

A polygon is a translucent fill with an outline; it closes itself.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Outline colour, also used for the fill. |
| `width` | `2.5` | Outline width in pixels; `0` for no outline. |
| `opacity` | `1.0` | Outline opacity. |
| `fill` | `True` | Whether to fill. |
| `fill_color` | same as `color` | Fill colour. |
| `fill_opacity` | `0.25` | Fill opacity. |
| `fill_rule` | `"nonzero"` | For outlines that cross themselves: `"nonzero"` fills everything enclosed, `"evenodd"` leaves doubly enclosed areas (such as a star's centre) empty. |
| `style` | `None` | A reusable `PolygonStyle`. |

To fill the area under a curve, build the polygon from the curve's points;
see [Recipes](recipes.md#shading-the-area-under-a-curve).

## Drawing order

Plots are drawn in the order you call the methods, each on top of the ones
before. Draw fills first and points last so that nothing important is
covered, and draw your own overlays (captions, a legend) after the plots.
Opacity blends properly with what is underneath, including on surfaces
with per-pixel alpha, so translucent layers can be composited later.

## Log axes

Everything above works on [logarithmic axes](log-axes.md) too. Functions
and curves are sampled in log space, so a power law is an exactly straight
line on log-log axes, and implicit curves are found on a grid that is even
on screen. Points, polylines and polygons are transformed vertex
by vertex and joined by straight lines on screen.
