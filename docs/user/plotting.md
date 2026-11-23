# Plotting

Every plotting method draws onto a surface, on top of whatever is already
there, and is called every frame after `grid.draw`. Everything is drawn
from the current view each time, so plots follow panning and zooming
exactly, and changing what you draw from one frame to the next animates it.
Plots that have not changed are [reused](#reusing-unchanged-plots)
automatically, so frames where little changes stay cheap.

| Method | Draws |
|---|---|
| [`plot(surface, f)`](#functions) | The graph of `y = f(x)`, or of a [derivative](#derivatives), an [antiderivative](#integrals-and-antiderivatives), a [Taylor polynomial or a Fourier series](#taylor-polynomials-and-fourier-series). |
| [`parametric(surface, x, y, t_range)`](#parametric-curves) | The curve `(x(t), y(t))`. |
| [`polar(surface, r)`](#polar-curves) | The curve `r = r(θ)`. |
| [`implicit(surface, F)`](#implicit-curves) | The curve `F(x, y) = 0` (or `= c`). |
| [`fill_between(surface, f, g)`](#areas-between-graphs) | The area between two graphs (or a graph and the x axis). |
| [`inequality(surface, condition)`](#inequalities-and-regions) | The region where a condition such as `x² + y² < 4` holds. |
| [`vector_field(surface, field)`](#vector-fields) | Arrows showing a vector field `(u, v) = F(x, y)`. |
| [`slope_field(surface, slope)`](#slope-fields) | Segments showing the slopes of `dy/dx = f(x, y)`. |
| [`sequence(surface, terms)`](#sequences) | The terms `aₙ` of a sequence: a list or array, or a function of `n`. |
| [`points(surface, points)`](#points) | Markers that keep their size in pixels: a scatter plot, with a colour and size per point and error bars. |
| [`polyline(surface, points)`](#polylines) | Vertices joined by straight lines, or by steps. |
| [`polygon(surface, points)`](#polygons) | A filled shape with an outline. |
| [`bars(surface, heights)`](#bars) | Bars standing on the x axis: a bar chart. |
| [`histogram(surface, samples)`](#histograms) | How many samples fall in each bin, or the density they estimate. |
| [`box_plot(surface, samples)`](#box-and-violin-plots) | The quartiles, median and spread of groups of samples. |
| [`violin_plot(surface, samples)`](#box-and-violin-plots) | The density of groups of samples, side by side. |
| [`color_bar(surface, colors, value_range)`](#colour-by-value) | Which colour is which value, for points coloured by a value. |
| [`arrow(surface, start, end)`](#arrows) | Arrows between points. |
| [`text(surface, text, position)`](#text) | Text at a point, moving with it. |
| [`legend(surface)`](#legends) | A legend naming the plots drawn with a `label=`. |

All of them accept `color`, `opacity` and a reusable `style`; see
[Styles and themes](styles-and-themes.md). Objects drawn without a colour
take the theme's [palette](styles-and-themes.md#the-palette) colours in
order. Every plotting method also takes a `label=`, the plot's name in the
[legend](#legends). Everything is clipped to the plane's rectangle. Axis
titles and arrowheads on the axes are part of the grid; see
[The plane](the-plane.md#axis-titles-and-arrowheads).

## Functions

```py
grid.plot(surface, func, *, color=None, width=None, opacity=None,
           domain=None, style=None, sampling=None)
```

```python
grid.plot(screen, np.sin)
grid.plot(screen, lambda x: x**2 / 4, color="crimson", width=4)
grid.plot(screen, np.sqrt, domain=(0, 4))             # only 0 <= x <= 4
grid.plot(screen, math.gamma)                         # scalar-only functions work too
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
| `sampling` | the plane's | A `FunctionSampling` for this call. |

### Sampling settings

Sampling is automatic and you will rarely need to change it. If you do, a
`FunctionSampling` can be passed to `plot` or set for the whole plane
(`Grid(sampling=...)` or `grid.sampling = ...`):

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
    grid.plot(screen, f)
```

Without the `k=k`, all three functions would use the last `k`, because a
lambda looks its variables up when it is called, not when it is created. The
same applies to animation variables captured by lambdas inside a loop.

## Derivatives

`pgp.derivative(f, n)` returns the `n`-th derivative of a function of one
variable, itself a function: plot it, call it at a point, or pass it
anywhere a function is expected.

```py
pgp.derivative(func, n=1)          # f⁽ⁿ⁾, a function of x
pgp.derivatives(func, x, n)        # (f(x), f'(x), ..., f⁽ⁿ⁾(x)) at once
```

```python
def f(x):
    return np.sin(x) * np.exp(-x / 4)

grid.plot(screen, f)
grid.plot(screen, pgp.derivative(f))          # f'
grid.plot(screen, pgp.derivative(f, 2))       # f''
slope = pgp.derivative(f)(1.5)                # a number, for a number
value, slope, curvature = pgp.derivatives(f, 1.5, 2)
```

The derivatives are **exact** (to rounding) for functions written with
NumPy, at any order up to 40: the function is not sampled at nearby points
but run with a special number that carries its derivatives along (Taylor
mode automatic differentiation). There is no step size to choose and no
error that grows with the order, and derivatives stay exact right next to
jumps, kinks and poles: the derivative of `np.abs` jumps from -1 to 1 at 0,
that of `np.floor` is 0 between the steps, and neither shows a spike.
Where the function is undefined, so is its derivative.

What can be differentiated exactly:

- arithmetic and powers, `np.sqrt`, `np.cbrt`, `np.exp`, `np.log` (and
  their base-2, base-10, `expm1` and `log1p` forms), trigonometric,
  hyperbolic and inverse functions, `np.arctan2`, `np.hypot`;
- piecewise functions: `np.abs`, `np.sign`, `np.floor` and friends, `%`,
  `np.maximum`/`np.minimum`, `np.where`, `np.clip`, `np.select` and
  `np.piecewise`, with comparisons such as `x > 0`;
- functions written for one number at a time with NumPy and Python `if`
  statements, such as `def f(x): return np.sin(x) if x > 0 else x * x`.
  These are exact too, but differentiated one point at a time, a few
  hundred times more slowly (about 40 µs per point); write them with
  `np.where` if speed matters.

Functions written with the `math` module, or that use something without a
rule (SciPy's special functions, for example), are differentiated
**numerically** instead, by finite differences. That is automatic and works
well for smooth functions (about `1e-10` relative error for a first
derivative, `1e-6` for a fourth), but near a jump or a kink it sees a slope
that is not there (a spike), and it loses accuracy fast at high orders.
Prefer NumPy for anything you differentiate.

A derivative of a derivative simply raises the order:
`pgp.derivative(pgp.derivative(f), 2)` is the third derivative, computed
directly.

### Tangent lines and Taylor polynomials

`derivatives` gives the value and slope together, which is all a tangent
line needs. With a [draggable point](draggable-points.md) on the curve, the
tangent follows the point:

```python
p = grid.draggable(curve=(lambda t: t, f, (-10, 10)), t=1.0)   # before the loop

value, slope = pgp.derivatives(f, p.x, 1)
grid.plot(screen, f)
grid.plot(screen, lambda x: value + slope * (x - p.x))
grid.draw_controls(screen)
```

A Taylor polynomial of degree `n` uses the first `n` derivatives, and
[`pgp.taylor`](#taylor-polynomials-and-fourier-series) builds it:

```python
grid.plot(screen, pgp.taylor(f, p.x, 5))       # degree 5, about the point
```

`examples/derivatives.py` shows both, and the osculating circle.

## Integrals and antiderivatives

```py
pgp.integral(func, a, b)           # ∫ₐᵇ f(x) dx, a number
pgp.antiderivative(func, a=0)      # F(x) = ∫ₐˣ f(t) dt, a function of x
```

```python
area = pgp.integral(np.sin, 0, np.pi)                            # 2.0
bell = pgp.integral(lambda x: np.exp(-x**2), -np.inf, np.inf)    # √π
F = pgp.antiderivative(np.cos)                                   # sin, as F(0) = 0
grid.plot(screen, F)
value = F(1.0)                                                   # a number, for a number
```

`integral` computes a definite integral adaptively (Gauss-Kronrod
quadrature, spending the effort where the function needs it), to about 10
significant digits for smooth functions, for functions with jumps
(`np.floor`) and kinks (`np.abs`), and for singularities that can be
integrated, such as `1/sqrt(x)` or `log(x)` at 0. Either limit may be
infinite, and `b < a` gives the negative of the integral from `b` to `a`.
Functions are written as for `plot`, with NumPy or `math`, and may raise
or return NaN.

When there is no honest number to give, the result is NaN rather than a
wrong one: when the integral diverges (`1/x` across 0, or from 1 to
infinity) and when the function is undefined on part of the interval
(`np.sqrt` from -1). A single point where the function is undefined, like
`sin(x) / x` at 0, makes no difference to an integral and is ignored.

`antiderivative` returns `F(x) = ∫ₐˣ f(t) dt`, with `F(a) = 0`: a function
to plot, evaluate or pass on, like a [derivative](#derivatives). Choose `a`
where the function is defined: the antiderivative of `1/x` from 1 is
`log x`, while from 0 it would be undefined everywhere. Beyond a point where
the integral from `a` diverges, `F` is undefined, so that antiderivative
has no graph for `x < 0`. `a = -np.inf` gives a cumulative distribution,
and `F(np.inf)` the integral to infinity:

```python
normal = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
cdf = pgp.antiderivative(normal, -np.inf)
grid.plot(screen, cdf)
cdf(1.96)                                       # 0.975
log = pgp.antiderivative(lambda x: 1 / x, 1)    # log x, for x > 0
```

The derivative of an antiderivative is the function itself, and
`pgp.derivative(F)` gives exactly that: `pgp.derivative(F)` is `cos`, not a
numerical difference of `F`.

Plotting an antiderivative costs about as much as integrating over the
plotted range a few times, not once per point: the plotted points are
integrated between neighbours, and `F` is their running total from `a`.
Unchanged antiderivatives are [reused](#reusing-unchanged-plots) like
other plots.

### The area under a curve

`fill_between` shades the area and `integral` measures it. With two
[draggable points](draggable-points.md) on the curve, both follow the
points:

```python
p = grid.draggable(curve=(lambda t: t, f, (-10, 10)), t=-1.0)   # before the loop
q = grid.draggable(curve=(lambda t: t, f, (-10, 10)), t=2.0)

grid.fill_between(screen, f, domain=(min(p.x, q.x), max(p.x, q.x)))
grid.plot(screen, f)
grid.draw_controls(screen, p, q)
area = pgp.integral(f, p.x, q.x)        # below the x axis counts as negative
```

Show `area` with a pygame font, as `examples/integrals.py` does. For the
area between two graphs, integrate their difference:
`pgp.integral(lambda x: f(x) - g(x), a, b)`, or `np.abs` of it for the
area regardless of which graph is on top.

### Riemann sums

A Riemann sum is a few lines of NumPy, and its rectangles are a step
function for `fill_between` and their outlines one `polyline` (NaN
vertices separate them):

```python
a, b, n = 0.0, 3.0, 12
x = np.linspace(a, b, n + 1)
h = f(x[:-1])                       # left ends; f(x[1:]) right, f((x[:-1] + x[1:]) / 2) midpoints
riemann = h.sum() * (b - a) / n     # tends to pgp.integral(f, a, b) as n grows

step = lambda t: h[np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 1)]
grid.fill_between(screen, step, domain=(a, b), color="steelblue")
gap = np.full(n, np.nan)
boxes = np.column_stack([x[:-1], 0 * h, x[:-1], h, x[1:], h, x[1:], 0 * h, gap, gap])
grid.polyline(screen, boxes.reshape(-1, 2), width=1.5, color="steelblue")
grid.plot(screen, f)
```

With a [slider](draggable-points.md#sliders) for `n`, the rectangles close in on the
area as you drag; `examples/integrals.py` does this, with left, right and
midpoint sums and the trapezoidal rule (whose trapezoids are
`fill_between(screen, lambda t: np.interp(t, x, f(x)), domain=(a, b))`).

### What integrals cannot do

- **Integrals that converge only by cancellation**, such as `sin(x) / x`
  from 0 to infinity, are NaN: the integrand oscillates forever without
  decaying fast enough to be added up piece by piece.
- **Integrands that oscillate many thousands of times** over the interval
  (`sin(10**6 * x)` from 0 to 1) are NaN: the work is limited, so that no
  integral can freeze a frame.
- **Singularities at an end away from 0**, such as `1 / sqrt(1 - x**2)` at
  1, are integrated to about 8 digits rather than 10, and are NaN beyond
  about ±10, because floating point cannot get closer to the singularity
  than about `1e-13` times its distance from 0. Move the singularity to 0:
  `pgp.integral(lambda t: 1 / np.sqrt(t), 0, 1)` rather than
  `pgp.integral(lambda x: 1 / np.sqrt(x - 10), 10, 11)`.
- **A feature much narrower than the interval** (a spike a millionth as wide)
  can be missed if no sample lands on it, as with any numerical integration.
  Integrate over a shorter interval around it.
- **An antiderivative's last digits** can differ between calls, as `F(x)`
  depends a little on the other points it is evaluated with. The values
  agree to about 10 digits, which is invisible in a plot.
- **Singular integrands at the lower limit** (the antiderivative of
  `1/sqrt(x)` from 0) need more work at every point, so plotting one takes
  around 15 ms when it is not reused.

## Taylor polynomials and Fourier series

```py
pgp.taylor(func, a, n)                     # the Taylor polynomial of degree n about a
pgp.fourier(func, n, interval=None)        # the Fourier series with n harmonics, on (-π, π)
```

```python
p5 = pgp.taylor(np.sin, 0, 5)              # x - x³/6 + x⁵/120
grid.plot(screen, np.sin)
grid.plot(screen, p5)
p5.coefficients                            # [0, 1, 0, -1/6, 0, 1/120]

square = pgp.fourier(np.sign, 15)          # a square wave: sign on (-π, π), repeated
grid.plot(screen, square)
square.sines[:4]                           # [0, 4/π, 0, 4/(3π)]
```

Both return functions of `x`, to plot, evaluate or differentiate like any
other. They are computed once, when created, and are then cheap to
evaluate.

`taylor(func, a, n)` is the polynomial `Σ f⁽ᵏ⁾(a) (x - a)ᵏ / k!`, up to
degree 40, from the [exact derivatives](#derivatives) at `a` (to rounding,
for functions written with NumPy; for `math` functions the derivatives are
numerical, good only to the first few orders). Its `coefficients` are
`f⁽ᵏ⁾(a) / k!`, lowest first, and its `center` is `a`.

`fourier(func, n, interval)` is the partial sum

`S(x) = a₀/2 + Σₖ (aₖ cos kωx + bₖ sin kωx)`,  `k = 1 ... n`,  `ω = 2π/T`,

of the Fourier series of the function on `interval` (default `(-π, π)`),
whose length `T` is the period. The function need not be periodic: its
values on the interval are repeated, so `np.sign` gives a square wave,
`lambda x: x` a sawtooth and `np.abs` a triangle wave. Its `cosines` are
`a₀ ... aₙ` and its `sines` `b₀ ... bₙ` (with `b₀ = 0`), and its
`amplitudes` the size of each harmonic, `√(aₖ² + bₖ²)` (and `|a₀|/2`, the
mean): its spectrum, which [`sequence`](#sequences) draws. The coefficients are integrals over the interval, computed
accurately for smooth functions and for jumps and kinks alike. Near a jump
the partial sums overshoot by about 9% however many harmonics there are:
that is the Gibbs phenomenon, a property of the series, not of the
drawing. `n` can be up to 1000; 100 harmonics take about a millisecond to
compute.

```python
saw = pgp.fourier(lambda x: x, 25)                      # period 2π
wave = pgp.fourier(lambda x: x * (1 - x), 5, (0, 1))    # period 1
grid.plot(screen, saw)
grid.plot(screen, wave)
grid.sequence(screen, saw.amplitudes)                   # its spectrum: 2/k
```

With a [slider](draggable-points.md#sliders) for `n` or the degree, the
approximations can be watched converging; `examples/series.py` does this for
Fourier series, and `examples/derivatives.py` for Taylor polynomials at a
draggable point.

## Parametric curves

```py
grid.parametric(surface, x, y, t_range, *, color=None, width=None,
                 opacity=None, style=None, sampling=None)
```

```python
grid.parametric(screen, np.cos, np.sin, (0, 2 * np.pi))               # unit circle
grid.parametric(screen, lambda t: t - np.sin(t), lambda t: 1 - np.cos(t), (0, 20))
grid.parametric(screen, 2.0, lambda t: t, (-1, 1))                    # the segment x = 2
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
grid.polar(surface, r, theta_range=(0, 2π), *, color=None, width=None,
            opacity=None, style=None, sampling=None)
```

```python
grid.polar(screen, lambda a: 2 * np.cos(3 * a))                   # three-petal rose
grid.polar(screen, lambda a: 1 + np.cos(a))                       # cardioid
grid.polar(screen, lambda a: 0.1 * a, (0, 10 * np.pi))            # spiral, 5 turns
grid.polar(screen, 1.5)                                           # circle of radius 1.5
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
grid.implicit(surface, func, level=0, *, color=None, width=None,
               opacity=None, style=None, sampling=None)
```

```python
grid.implicit(screen, lambda x, y: x**2 + y**2, 4)                        # circle, radius 2
grid.implicit(screen, lambda x, y: x**3 + y**3 - 6 * x * y)               # folium of Descartes
grid.implicit(screen, lambda x, y: (x**2 + y**2)**2 - 8 * (x**2 - y**2))  # lemniscate
grid.implicit(screen, lambda x, y: x - np.sin(y))                         # x = sin y
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
grid.implicit(screen, lambda x, y: y - np.tan(x))      # branches, no asymptotes
grid.implicit(screen, lambda x, y: y - np.floor(x))    # steps, no risers
grid.implicit(screen, lambda x, y: y - math.sqrt(x))   # starts at x = 0
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

grid.implicit(screen, height, np.linspace(-1.5, 1.5, 7), width=1.5)
grid.implicit(screen, height, 0, width=3)              # emphasise one level
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
grid.implicit(screen, lambda x, y: np.sin(x * y) - 0.5, sampling=fine)
```

The budget keeps slow functions usable: a plain `math` function costs
about 0.4 microseconds per evaluation (NumPy is tens of times faster), so
the default budget bounds a call at about 100 ms even in the worst case, a
curve that winds across the whole plane. Dense contour maps
on large windows can reach the budget with NumPy functions; raising
`max_samples` (to `10**6`, say) keeps them at full resolution.

## Areas between graphs

```py
grid.fill_between(surface, f, g=0, *, color=None, opacity=None,
                   domain=None, style=None, sampling=None)
```

```python
grid.fill_between(screen, np.sin, domain=(0, np.pi))            # under sin, above the x axis
grid.fill_between(screen, np.sin, np.cos, color="crimson")      # between two graphs
grid.fill_between(screen, lambda x: x**2, 4)                    # between a parabola and y = 4
grid.fill_between(screen, np.exp, -math.inf, domain=(-1, 1))    # down to the bottom edge
```

`fill_between` shades the area between the graphs of `f` and `g`, without
an outline. Each side is a function of `x`, written as for `plot`, or a
number for a horizontal line; `g` defaults to `0`, the x axis. Draw the
area first and the curves on top:

```python
grid.fill_between(screen, np.sin, np.cos, domain=(-2, 4), color="steelblue")
grid.plot(screen, np.sin, color="steelblue")
grid.plot(screen, np.cos, color="steelblue")
```

Both sides are sampled exactly like a `plot` of them, so the shading
meets the curves exactly, at every zoom. Where the graphs cross, both
parts are shaded. The area covers only the `x` where both sides are
defined, and ends with a straight edge wherever either side jumps or
ends: `fill_between(screen, np.tan)` shades each branch of `tan` without
joining them across the asymptotes, and
`fill_between(screen, np.sqrt, 0)` starts at `x = 0`.

`math.inf` and `-math.inf` shade all the way to the top or bottom of the
plane. On a [log y axis](log-axes.md), `0` (or any number `<= 0`) lies
infinitely far down, so `fill_between(screen, f)` shades down to the
bottom edge there.

| Parameter | Default | Meaning |
|---|---|---|
| `f`, `g` | (`g`: `0`) | The two sides: functions of `x`, or numbers. |
| `color` | next palette colour | Fill colour. |
| `opacity` | `0.25` | Fill opacity, 0 to 1. |
| `domain` | `None` | Only shade where `domain[0] <= x <= domain[1]`. |
| `style` | `None` | A reusable `FillStyle`. |
| `sampling` | the plane's | `FunctionSampling` for both sides, as for `plot`. |

## Inequalities and regions

```py
grid.inequality(surface, condition, *, color=None, opacity=None,
                 style=None, sampling=None)
```

```python
grid.inequality(screen, lambda x, y: x**2 + y**2 < 4)                      # a disc
grid.inequality(screen, lambda x, y: y > np.sin(x), color="seagreen")      # above a curve
grid.inequality(screen, lambda x, y: (y > x**2 - 2) & (y < 2 - x**2))      # both at once
grid.inequality(screen, lambda x, y: (np.abs(x) > 3) | (np.abs(y) > 2))    # either
```

`inequality` shades every point where `condition(x, y)` is true, without
an outline, like the inequalities of a graphing calculator. The condition
is a Python function of `x` and `y`, written with NumPy like the functions
given to `implicit`, and returning booleans. Combine conditions with
NumPy's `&` (and), `|` (or) and `~` (not), with each part in brackets
(`&` binds more tightly than `<`).

Python's own `and`, `or`, `not` and chained comparisons (`0 < x < 1`) work
too, but only on single numbers, so the condition is then evaluated one
point at a time, tens of times more slowly. Any value that is not zero
counts as true, so functions returning 0 and 1 work as well. Where the
condition raises an error or returns NaN, the point is outside: the region
`lambda x, y: np.sqrt(x) < y` starts at `x = 0`.

To show the boundary, draw it with `implicit` (or `plot`):

```python
grid.inequality(screen, lambda x, y: x**2 + y**2 < 4, color="steelblue")
grid.implicit(screen, lambda x, y: x**2 + y**2, 4, color="steelblue")
```

A region can have any shape: several pieces, holes, parts that end at an
asymptote (`y < tan(x)`), or edges where the condition jumps
(`y < floor(x)`). It is found again every frame on the same grid as
implicit curves, with its edges moved onto the true boundary to a small
fraction of a pixel, and only the neighbourhood of the boundary is
sampled finely, so a large region costs no more than a small one with the
same outline. It takes the same `sampling=` settings as implicit curves
([above](#implicit-sampling-settings)).

Islands and holes smaller than about 8 pixels across (the coarse grid's
spacing) may be missed, and so may bands thinner than that; zoom in, or
use a smaller `coarse_factor`, to find them.

| Parameter | Default | Meaning |
|---|---|---|
| `condition` | | `condition(x, y)`: true inside. |
| `color` | next palette colour | Fill colour. |
| `opacity` | `0.25` | Fill opacity, 0 to 1. |
| `style` | `None` | A reusable `FillStyle`. |
| `sampling` | `ImplicitSampling()` | How the region is searched for. |

## Vector fields

```py
grid.vector_field(surface, field, *, spacing=None, scale=None, color=None,
                   width=None, opacity=None, head_size=None, style=None)
```

```python
grid.vector_field(screen, lambda x, y: (-y, x))                    # a rotation
grid.vector_field(screen, lambda x, y: (x, -y), color="seagreen")  # a saddle
grid.vector_field(screen, lambda x, y: (1, np.sin(x * y)))         # u is a number
```

A vector field gives a vector `(u, v)` at every point of the plane, and is
drawn as arrows on a lattice of points. `field` takes `x` and `y` and
returns the vector there as a pair, written with NumPy like the functions
given to `implicit` (or plain `math`, more slowly). A component may be a
plain number, as in `(1, np.sin(x * y))`. Where the field raises an error,
returns NaN or an infinite component, or is zero, no arrow is drawn.

The arrows are about `spacing` pixels apart (40 by default) at any zoom.
The lattice is fixed to the plane, so panning moves the arrows with it,
and while zooming, new arrows grow and fade in between the others (or
shrink and fade out), so that the density stays the same and nothing
jumps: at any zoom the arrows are between `spacing` and twice that apart.

By default every arrow has the same length, so the picture shows the
field's *direction*. Give `scale` to show its *magnitude* as well: each
arrow is then `scale` pixels long per unit of magnitude, so with
`scale=10` a vector of length 2 is drawn 20 pixels long. Lengths are capped
at 0.8 × `spacing`, so that arrows never run into each other; arrows
shorter than their head are drawn as a smaller head alone.

```python
def electric(x, y):                   # a charge at the origin: E = r / |r|³
    r3 = np.hypot(x, y) ** 3
    return x / r3, y / r3

grid.vector_field(screen, electric)                       # directions: readable everywhere
grid.vector_field(screen, lambda x, y: (-y, x), scale=8)  # 8 px per unit of magnitude
```

Fields such as the one of a point charge, whose magnitude ranges over many
orders, read best by direction alone; fields of moderate size, by
magnitude too. Each arrow is centred on its point of the lattice and is
tangent to the field's flow lines on screen, on log axes as well (a vector
is taken as a direction in world units at its point).

| Parameter | Default | Meaning |
|---|---|---|
| `spacing` | `40` | Distance between neighbouring arrows in pixels (at least 10). |
| `scale` | `None` | `None` for arrows of one length, or pixels per unit of magnitude. |
| `color` | next palette colour | Arrow colour. |
| `width` | `1.5` | Shaft width in pixels. |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `head_size` | in proportion | Length of the heads in pixels (`0` for none). |
| `style` | `None` | A reusable `ArrowStyle`. |

## Slope fields

```py
grid.slope_field(surface, slope, *, spacing=None, color=None, width=None,
                  opacity=None, style=None)
```

```python
grid.slope_field(screen, lambda x, y: x - y)                           # dy/dx = x - y
grid.plot(screen, lambda x: x - 1 + 2 * np.exp(-x), color="crimson")   # one solution
```

A slope field shows the differential equation `dy/dx = slope(x, y)`: a
short segment at each point of a lattice, with the slope the equation
gives there, so that the solution curves can be seen running along the
segments. `slope` takes `x` and `y` like the conditions of `inequality`.
An infinite slope gives a vertical segment, and where `slope` raises an
error or returns NaN no segment is drawn.

The lattice behaves exactly as for [vector fields](#vector-fields):
`spacing` pixels apart (40 by default), fixed to the plane, with segments
fading in and out while zooming. On log axes the segments are tangent to
the solution curves on screen, so `dy/dx = y / x`, whose solutions are the
lines `y = c x`, has every segment at 45° on log-log axes. For arrows
instead of segments, draw the vector field `lambda x, y: (1, slope(x, y))`.

| Parameter | Default | Meaning |
|---|---|---|
| `spacing` | `40` | Distance between neighbouring segments in pixels (at least 10). |
| `color` | next palette colour | Segment colour. |
| `width` | `1.5` | Line width in pixels. |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `style` | `None` | A reusable `LineStyle`. |

## Sequences

```py
grid.sequence(surface, terms, start=0, *, baseline=0, color=None,
               width=None, radius=None, marker=None, opacity=None, style=None)
```

```python
grid.sequence(screen, [3, 1, 4, 1, 5, 9, 2, 6])                  # a₀ ... a₇
grid.sequence(screen, lambda n: (-1) ** n / n, start=1)          # every n ≥ 1
grid.sequence(screen, pgp.fourier(np.sign, 15).amplitudes)       # a spectrum
grid.sequence(screen, lambda n: 1 - 1 / n, start=1, width=0)     # markers alone
```

Each term `aₙ` is drawn at `x = n` as a stem from the x axis to `(n, aₙ)`
with a marker at its end, as in a spectrum or a stem plot. `terms` is:

- **a list or array**: a finite sequence, whose first term has the index
  `start` (0 by default, so the terms of a Python list sit at their
  indices). It is drawn in full; NaN terms are left out;
- **a function of `n`**: an infinite sequence, drawn at every visible
  integer `n >= start`. It is called with a float array of whole numbers,
  so NumPy arithmetic cannot overflow (`2**n` is fine), or, if it only
  works on one value at a time, with each `n` as a Python `int`, so
  `math.factorial`, `math.comb` and list indexing work. Where it raises,
  returns NaN or infinity, the term is left out: `1/n` simply has no term
  at 0.

```python
fib = [1, 1]
while len(fib) < 30:
    fib.append(fib[-1] + fib[-2])
grid.sequence(screen, lambda n: fib[n + 1] / fib[n], width=0)   # → the golden ratio
grid.sequence(screen, lambda n: math.comb(10, n) / 50)          # a row of Pascal's triangle
partial = np.cumsum(1 / np.arange(1, 50) ** 2)
grid.sequence(screen, partial, start=1)                         # → π²/6
```

A limit is a horizontal line, `grid.plot(screen, lambda x: 0 * x + np.pi**2 / 6)`;
with `baseline=` the stems start from it instead of the x axis, so they
shrink as the terms close in:

```python
grid.sequence(screen, partial, start=1, baseline=np.pi**2 / 6)
```

A recurrence `xₙ₊₁ = g(xₙ)` is a few lines of Python building a list:

```python
r, x = 3.5, [0.2]
for _ in range(60):
    x.append(r * x[-1] * (1 - x[-1]))       # the logistic map
grid.sequence(screen, x, width=0)
```

Zoomed in, every term is drawn; zoomed far out, the stems of the terms
that share a pixel column are drawn as one, which covers exactly what they
would together, so even thousands of terms per pixel cost one stem per
pixel. An infinite sequence's terms closer than a quarter of a pixel are
thinned: only multiples of 3, 9, 27, ... are drawn, the others fading out
as you zoom out, so nothing pops and panning stays rigid. Thinning by
powers of 3 keeps odd and even terms alike, so an alternating sequence
stays two-sided and a square wave's odd harmonics stay visible; only a
pattern repeating every 3, 9, ... terms can be misrepresented, and only
when its terms are less than a quarter of a pixel apart. A finite
sequence is never thinned.

On a [log y axis](log-axes.md) the stems reach down to the bottom edge
(a baseline of 0 is infinitely far down) and terms that are not positive
are left out;
markers alone (`width=0`) are often clearer there. On a log x axis the
terms crowd together to the right, and are thinned there in the same way.

| Parameter | Default | Meaning |
|---|---|---|
| `terms` | | A list or 1-D array of numbers, or a function of `n`. |
| `start` | `0` | The index of the first term (a whole number). |
| `baseline` | `0` | Where the stems start: a y value, such as a limit. |
| `color` | next palette colour | Colour of the stems and markers. |
| `width` | `2.0` | Stem width in pixels; `0` for markers alone. |
| `radius` | `3.5` | Marker size in pixels; `0` for stems alone. |
| `marker` | `"circle"` | `"circle"`, `"square"`, `"diamond"`, `"cross"` or `"plus"`. |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `style` | `None` | A reusable `SequenceStyle`. |

## Plotting data

Data (measurements, samples, the state of a running simulation) is drawn
with the same methods as everything else, from the program's own lists and
arrays, every frame:

| To draw | Use |
|---|---|
| A scatter plot | [`points`](#points), with a colour and a size per point if they show a value. |
| Error bars | `points(..., y_error=, x_error=)`; with `radius=0`, the bars alone. |
| A line through data | [`polyline`](#polylines); `step="post"` for steps (counts, held signals, an empirical distribution). |
| A bar chart | [`bars`](#bars), from heights or `(x, height)` pairs. |
| A histogram | [`histogram`](#histograms), from the raw samples (`cumulative=True` for a distribution). |
| Groups of samples compared | [`box_plot`](#box-and-violin-plots) or [`violin_plot`](#box-and-violin-plots). |
| A value shown by colour | [`pgp.color_map`](#colour-by-value) for `points(color=)`, and a [`color_bar`](#colour-by-value). |
| Stacked, floating or horizontal bars | [`bars`](#bars) with `baseline=` per bar, or `horizontal=True`. |
| Categories along an axis | `grid.draw(x_labels=["a", "b", ...])` (see [The plane](the-plane.md#categories-on-an-axis)). |
| A model or fit | [`plot`](#functions) of the function, over the data. |
| A band (a confidence interval) | [`fill_between`](#areas-between-graphs) of `lambda x: np.interp(x, xs, low)` and `... high`. |

```python
rng = np.random.default_rng(1)
xs = np.arange(10.0)
ys = 0.5 * xs + rng.normal(0, 0.4, xs.size)
grid.points(screen, np.column_stack((xs, ys)), y_error=0.4, label="measured")
grid.plot(screen, lambda x: 0.5 * x, label="model")
grid.legend(screen, corner="top-left")
```

The data may change from frame to frame, in place or not: unchanged data
is [reused](#reusing-unchanged-plots) (it is compared, not trusted), and
changed data is drawn again. To show the data at first, set the view from
it with `pgp.data_range`, which gives a range around values with a margin
(see [Setting up the view](the-plane.md#setting-up-the-view)):

```python
grid = pgp.Grid(x_range=pgp.data_range(xs), y_range=pgp.data_range(ys, include=0))
```

## Points

```py
grid.points(surface, points, *, color=None, radius=None, marker=None,
             edge_color=None, edge_width=None, opacity=None, x_error=None,
             y_error=None, error_width=None, error_cap=None, style=None)
```

```python
grid.points(screen, (1, 2))                                     # one point
grid.points(screen, [(0, 0), (1, 1), (2, 4)], marker="diamond")
xs = np.linspace(-3, 3, 25)
grid.points(screen, np.column_stack((xs, np.sin(xs))), edge_color="white")
```

`points` is a single point `(x, y)`, a sequence of points, or an `(N, 2)`
array (use `np.column_stack((xs, ys))` for separate arrays). Points with a
NaN coordinate are skipped. Markers keep the same size in pixels at every
zoom level, and are positioned to a quarter of a pixel, so they glide
smoothly while panning.

### A colour and a size per point

`color=` and `radius=` take one value for every point, or one per point,
for a scatter plot that shows a third value by colour or size:

```python
pts = np.random.default_rng(2).normal(size=(200, 2))
r = np.hypot(pts[:, 0], pts[:, 1])
shade = np.clip(r / 2.5, 0, 1)[:, None]
colors = (1 - shade) * (40, 100, 200) + shade * (220, 60, 40)   # blue to red
grid.points(screen, pts, color=colors, radius=2 + 3 * shade[:, 0])
grid.points(screen, [(-2, 2), (0, 2), (2, 2)], color=["crimson", "seagreen", "gold"])
```

A colour per point is a sequence of colours (names, or `(r, g, b)` and
`(r, g, b, a)` tuples) or an `(N, 3)` or `(N, 4)` array of values from 0
to 255; a sequence of three or four numbers is one colour. Colours per
point use no palette colour, and a legend entry shows the first. A size
per point is in pixels, rounded to a quarter of a pixel; NaN or 0 draws
no marker. Markers are drawn in order, each on top of the ones before. To
colour by category (clusters, say), one call per category is often
better: each gets a palette colour and a legend entry.

### Error bars

```python
grid.points(screen, [(1, 2), (2, 3)], y_error=0.5)                       # ± 0.5
grid.points(screen, [(1, 2), (2, 3)], y_error=[0.2, 0.4])                # one each
grid.points(screen, [(1, 2), (2, 3)], y_error=[[0.2, 0.4], [0.6, 0.1]])  # below, above
grid.points(screen, [(1, 2)], x_error=0.3, y_error=0.3, error_cap=0)     # no caps
```

`y_error` and `x_error` are one length for every point, one per point
(the same below and above), or two rows of one per point (below then
above, or left then right); NaN draws no bar for that point. The bars are
in world units, so they grow with the zoom, while their width and caps are
in pixels. They are drawn in the points' colour, under the markers (with a
colour per point, in the theme's label colour). `radius=0` draws the error
bars alone, to put them on [bars](#bars), say. On a log axis a bar
reaching zero or below runs off the plane, without a cap there.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Marker colour, or a colour per point. |
| `radius` | `4.5` | Size in pixels (other shapes have the same area as the circle), or a size per point; `0` for no marker. |
| `marker` | `"circle"` | `"circle"`, `"square"`, `"diamond"`, `"cross"` or `"plus"`. |
| `edge_color` | `None` | Outline colour, drawn *outside* the marker. A background-coloured outline (such as `"white"`) sets points apart from the curves they sit on. |
| `edge_width` | `1.5` | Outline thickness in pixels (only with `edge_color`). |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `x_error`, `y_error` | `None` | Error bar lengths (above). |
| `error_width` | `1.5` | Thickness of the error bars in pixels. |
| `error_cap` | `8` | Width of the caps across their ends, in pixels; `0` for none. |
| `style` | `None` | A reusable `PointStyle`. |

## Polylines

```py
grid.polyline(surface, points, *, color=None, width=None, opacity=None,
               closed=False, step=None, style=None)
```

```python
grid.polyline(screen, [(0, 0), (1, 2), (3, 1)])
grid.polyline(screen, [(0, 0), (1, 0), (1, 1)], closed=True, width=1.5)
grid.polyline(screen, [(0, 0), (1, 1), (np.nan, np.nan), (2, 0), (3, 1)])  # two pieces
```

The vertices are joined in order by straight lines, with round joins and
ends, antialiased at any width. A vertex with a NaN coordinate breaks the
line, like lifting the pen. `closed=True` joins the last vertex back to the
first. Vertices are in world units, so a polyline grows and shrinks with the
zoom; only the width is in pixels. Takes the `LineStyle` fields.

`step=` joins the vertices by steps instead, across and then up or down:

```python
samples = np.sort(np.random.default_rng(3).normal(size=50))
ecdf = np.column_stack((samples, np.arange(1, 51) / 50))
grid.polyline(screen, ecdf, step="post")     # an empirical distribution function
grid.polyline(screen, [(0, 1), (1, 3), (2, 2)], step="mid")
```

| `step` | Each value holds |
|---|---|
| `"post"` | from its x until the next x (counts, sampled-and-held signals, distribution functions) |
| `"pre"` | from the previous x up to its own |
| `"mid"` | around its x, the steps half way between the xs |

The steps are made on screen, so on a log x axis a `"mid"` step is half
way on screen too. A stepped polyline cannot be `closed`.

## Polygons

```py
grid.polygon(surface, points, *, color=None, width=None, opacity=None,
              fill=None, fill_color=None, fill_opacity=None, fill_rule=None,
              style=None)
```

```python
grid.polygon(screen, [(-3, -1), (-1, -1), (-2, 1)])                 # translucent triangle
grid.polygon(screen, [(0, 0), (2, 0), (2, 1), (0, 1)], fill_opacity=0.6, width=0)
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

To shade the area under a curve, or between two curves, use
[`fill_between`](#areas-between-graphs).

## Arrows

```py
grid.arrow(surface, start, end, *, color=None, width=None, opacity=None,
            head_size=None, style=None)
```

```python
grid.arrow(screen, (0, 0), (2, 1))                                  # one arrow
grid.arrow(screen, (0, 0), [(3, 0), (0, 3)], color="crimson")       # two, from the origin
grid.arrow(screen, (1, -2), (4, -1), width=4, opacity=0.5)
```

An arrow runs straight from `start` to its tip at `end`, both points in
world units, so it moves and grows with the zoom; its width and head are
in pixels, so it looks the same at every zoom. Draw a vector `v` at a
point `p` with `grid.arrow(screen, p, np.add(p, v))`.

Each of `start` and `end` is one point `(x, y)` or many (a sequence of
points, or an `(N, 2)` array): many arrows in one call are much faster than
one call each. A single start is shared by all the ends, and a single end
by all the starts. Arrows with a NaN coordinate, or whose ends are in the
same place, are left out.

An arrow is filled as one shape, shaft and head together, so translucent
arrows have no darker patch where the head joins the shaft, and arrows
that overlap merge rather than darken each other. Its tail is square.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Arrow colour. |
| `width` | `2.5` | Shaft width in pixels. |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `head_size` | in proportion | Length of the head in pixels: by default about `3 × width + 4`; `0` for none (a plain segment with square ends). An arrow shorter than its head is drawn as a smaller head alone. |
| `style` | `None` | A reusable `ArrowStyle`. |

## Bars

```py
grid.bars(surface, data, start=0, *, width=None, baseline=0, horizontal=False,
           color=None, opacity=None, edge_width=None, style=None)
```

```python
grid.bars(screen, [3, 1, 4, 1, 5])                          # at x = 0, 1, ... 4
rolls = np.random.default_rng(4).integers(1, 7, 600)
grid.bars(screen, np.bincount(rolls, minlength=7)[1:] / 100, start=1)   # dice
grid.bars(screen, [(0.5, -1), (1.5, 2)], width=0.6, baseline=0.5)
```

`data` is the bars' heights, standing at `start`, `start + 1`, ..., or
their `(x, top)` pairs (a sequence of pairs or an `(N, 2)` array); a NaN
leaves a bar out. A bar runs from `baseline` to its top, so a top below the
baseline hangs down. Bars are in world units and grow with the zoom; by
default each is 0.8 of the smallest gap between neighbouring bars, so they
never touch (bars as wide as their spacing do). They are a translucent fill
with an opaque outline in one colour.

For counts of whole-number outcomes (dice, a binomial distribution), bars
of `np.bincount` stand on the outcomes; a [histogram](#histograms) is for
continuous values. Bars side by side for several series are separate calls
with shifted xs and a narrower width:

```python
years = np.arange(2020, 2024)
grid.bars(screen, np.column_stack((years - 0.2, [3, 4, 5, 4])), width=0.4, label="north")
grid.bars(screen, np.column_stack((years + 0.2, [2, 5, 3, 6])), width=0.4, label="south")
```

`baseline` may also be one value per bar. Stacked bars stand each series
on the tops of the one before (pass the running totals, standing on the
previous ones); floating bars run from a start to an end (a schedule, a
waterfall, a range):

```python
a, b, c = np.array([2, 3, 1]), np.array([1, 2, 2]), np.array([1, 1, 2])
grid.bars(screen, a, label="a")
grid.bars(screen, a + b, baseline=a, label="b")              # on a
grid.bars(screen, a + b + c, baseline=a + b, label="c")      # on a + b
grid.bars(screen, [(5, 7.5), (6, 4)], baseline=[3, 1])       # from 3 to 7.5, 1 to 4
```

`horizontal=True` lays the bars along x, standing on a baseline that is an
x value; `data` is then their ends (x values) at `start`, `start + 1`, ...
on the y axis, or `(y, end)` pairs, and `width` is along y:

```python
grid.bars(screen, [6, 3, 7.5], horizontal=True, width=0.5)
grid.draw(screen, y_labels=["low", "medium", "high"])   # names on the y axis
```

Name categories on the axis with `grid.draw(x_labels=[...])` (see
[categories on an axis](the-plane.md#categories-on-an-axis)), write values
on the bars with [`text`](#text) (`anchor="bottom"` above a bar), and draw
error bars on them with `points(tops, y_error=..., radius=0)`. On a log axis
across the bars a baseline of 0 is the edge; on a log axis along them,
`width` is in decades, so bars look alike.

| Parameter | Default | Meaning |
|---|---|---|
| `data` | | Heights, or `(x, top)` pairs. |
| `start` | `0` | The position of the first bar, for heights. |
| `width` | 0.8 of the gap | Width of the bars in x units (y units when horizontal). |
| `baseline` | `0` | Where the bars stand: a y value (an x when horizontal), or one per bar. |
| `horizontal` | `False` | Lay the bars along x. |
| `color` | next palette colour | Colour of the fill and the outline. |
| `opacity` | `0.5` | Opacity of the fill; the outline is opaque. |
| `edge_width` | `1.5` | Outline width in pixels; `0` for none. |
| `style` | `None` | A reusable `BarStyle`. |

## Histograms

```py
grid.histogram(surface, samples, *, width=None, edges=None, density=False,
                cumulative=False, color=None, opacity=None, edge_width=None,
                style=None)
```

```python
samples = np.random.default_rng(5).normal(size=1000)
grid.histogram(screen, samples)                         # automatic bins
grid.histogram(screen, samples, width=0.25)             # [0, 0.25), [0.25, 0.5), ...
grid.histogram(screen, samples, edges=[-3, -1, 0, 1, 3])
```

`samples` are raw values; the plane counts how many fall in each bin and
draws a bar as high as the count. The bins are **anchored**: a bin of
width `w` is always `[k w, (k + 1) w)` for a whole number `k`, wherever the
samples are. So a histogram stays put while samples are added frame by
frame (a simulation filling it in), and panning and zooming never change
it. Without `width`, it is chosen from the samples (the Freedman-Diaconis
rule: twice the interquartile range over the cube root of the number of
samples) and rounded to 1, 2 or 5 times a power of ten, so bin edges fall
on round numbers and the width changes only now and then as samples
arrive. `edges` gives the bins' edges instead, as in NumPy: `[a, b)` each,
the last including its right edge, and samples outside left out.

With `density=True` each bar is as high as the density the samples
estimate there (its count over the number of samples and the bin's width),
so the bars have area 1 and a probability density drawn over them matches:

```python
samples = np.random.default_rng(6).normal(size=2000)
grid.histogram(screen, samples, density=True, label="2000 samples")
grid.plot(screen, lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), label="N(0, 1)")
grid.legend(screen)
```

With `cumulative=True` each bar is as high as all the samples up to its
right edge, rising to their number, or to 1 with `density=True`: a stepped
estimate of the distribution function. Stretches without samples keep the
height reached, so it is one rising shape.

```python
grid.histogram(screen, samples, width=0.25, cumulative=True, density=True)
```

Neighbouring bars are filled as one shape with one outline, with lines
between them that fade out as the bins narrow below a few pixels, so a
histogram zoomed far out shows its outline rather than a comb of lines.
Only bins with samples are kept, so a sample far from the others costs a
bin of its own, nothing more. NaN and infinite samples are left out. On a
log x axis the bins are equal on screen, and `width` is in decades (`0.1`
is a tenth of a decade); the density is still per x unit.

| Parameter | Default | Meaning |
|---|---|---|
| `samples` | | The values: a sequence or array of numbers. |
| `width` | automatic | The bins' width in x units. |
| `edges` | `None` | The bins' edges instead (at least two, increasing). |
| `density` | `False` | Draw densities (area 1) instead of counts; with `cumulative`, fractions. |
| `cumulative` | `False` | Draw the counts (or fractions) added up along x. |
| `color` | next palette colour | Colour of the fill and the outline. |
| `opacity` | `0.5` | Opacity of the fill; the outline is opaque. |
| `edge_width` | `1.5` | Outline width in pixels; `0` for none. |
| `style` | `None` | A reusable `BarStyle`. |

## Box and violin plots

```py
grid.box_plot(surface, samples, start=0, *, positions=None, width=None,
               whisker=1.5, horizontal=False, color=None, opacity=None,
               edge_width=None, style=None)
grid.violin_plot(surface, samples, start=0, *, positions=None, width=None,
                  bandwidth=None, horizontal=False, color=None, opacity=None,
                  edge_width=None, style=None)
```

```python
rng = np.random.default_rng(8)
groups = [rng.normal(0, 1, 300), rng.gamma(2, 1.2, 300), rng.standard_t(3, 300)]
grid.box_plot(screen, groups)                                 # at x = 0, 1, 2
grid.violin_plot(screen, groups, start=4)                     # at x = 4, 5, 6
grid.violin_plot(screen, groups, start=8)
grid.box_plot(screen, groups, start=8, width=0.1, opacity=1)  # a narrow box in each
grid.draw(screen, x_labels={1: "box plots", 5: "violins"})
```

Both summarise groups of samples side by side. `samples` is one group (a
sequence or 1-D array of numbers) or several (a list of groups, which may
differ in size, or a 2-D array, one group per row), standing at `start`,
`start + 1`, ... or at `positions`.

A **box plot** shows each group's quartiles as a box with the median across
it, whiskers out to the furthest samples within `whisker` interquartile
ranges of the box (Tukey's 1.5; `whisker=math.inf` reaches every sample),
and the samples beyond them as small points: the outliers. A **violin
plot** shows each group's density, estimated with a Gaussian kernel
(Silverman's bandwidth unless `bandwidth=` is given), mirrored either side
of its position and `width` wide where it is greatest, from the smallest
sample to the largest, with the median across. A narrow box inside a
violin shows both. A group needs two different values to have a violin.

Boxes default to half the smallest gap between groups and violins to 0.8
of it. `horizontal=True` lays them along x, the positions on the y axis.
On a log axis they summarise the logarithms of the samples, as the axis
shows them. Both take the `BarStyle` fields; lines are at least a pixel
wide, and medians a pixel wider than the outlines.

| Parameter | Default | Meaning |
|---|---|---|
| `samples` | | One group of samples, or a list of groups. |
| `start` | `0` | The position of the first group, without `positions`. |
| `positions` | `None` | One position per group. |
| `width` | from the gaps | Width of the boxes (half the smallest gap) or violins (0.8 of it). |
| `whisker` | `1.5` | Box plots: how far whiskers reach, in interquartile ranges. |
| `bandwidth` | Silverman's rule | Violins: the kernel's standard deviation, in value units. |
| `horizontal` | `False` | Lay them along x. |
| `color` | next palette colour | Colour of the fill and the lines. |
| `opacity` | `0.5` | Opacity of the fill. |
| `edge_width` | `1.5` | Outline width in pixels. |
| `style` | `None` | A reusable `BarStyle`. |

## Colour by value

```py
pgp.color_map(values, colors="viridis", value_range=None, *, log=False)
grid.color_bar(surface, colors, value_range, *, log=False, title=None, corner="top-right")
```

```python
rng = np.random.default_rng(9)
positions = rng.normal(size=(300, 2))
speed = np.hypot(positions[:, 0], positions[:, 1])
grid.points(screen, positions, color=pgp.color_map(speed, "plasma", (0, 3)))
grid.color_bar(screen, "plasma", (0, 3), title="speed")
```

`pgp.color_map` turns values into colours along a colour map, ready for a
[colour per point](#a-colour-and-a-size-per-point), and `grid.color_bar`
draws the key: the map as an upright strip, low at the bottom, with ticks
at round values (1, 2 or 5 times a power of ten, or powers of ten on a log
scale) and a title. Give both the same map, range and scale.

Values are mapped linearly from `range[0]` (the map's first colour) to
`range[1]` (its last), or on a log scale with `log=True`, for values
spanning decades. Values beyond the range take the colour of its end; NaN
(and with `log`, values not positive) are transparent, so their points are
not drawn. Without a range `color_map` uses the values' own smallest and
largest: right for fixed data, but give a range for data that changes from
frame to frame, so a colour keeps its meaning.

| Map | Runs |
|---|---|
| `"viridis"` (default) | dark blue, green, yellow |
| `"plasma"` | dark blue, magenta, yellow |
| `"inferno"` | black, red, pale yellow |
| `"magma"` | black, purple, pale pink |
| `"coolwarm"` | blue, grey, red: diverging, for values either side of a middle |
| `"gray"` | black to white |

The first four are matplotlib's perceptually even maps: equal steps of
value look like equal steps of colour, they read the same in grey, and
colour-blind readers can read them. Any sequence of two or more colours is
a map too, evenly spaced from low to high, given as names, hex strings or
`(r, g, b)` tuples:

```python
signed = np.linspace(-1, 1, 9)
grid.points(screen, np.column_stack((signed * 3, signed)),
             color=pgp.color_map(signed, ["#2166ac", "white", "#b2182b"], (-1, 1)))
grid.color_bar(screen, ["#2166ac", "white", "#b2182b"], (-1, 1), corner="bottom-right")
```

The colour bar is a panel in a corner like the [legend](#legends): it stays
put while you pan and zoom, presses on it never pan the plane, and in a
corner with sliders or the legend it goes beyond them.

| Parameter | Default | Meaning |
|---|---|---|
| `values` | | Numbers of any shape (`color_map` returns one RGBA row each). |
| `colors` | `"viridis"` | A built-in map's name, or a sequence of colours. |
| `value_range` | the values' own | `(low, high)`: the values at the map's ends (required for `color_bar`). |
| `log` | `False` | Map on a log scale. |
| `title` | `None` | `color_bar`: a title above the strip. |
| `corner` | `"top-right"` | `color_bar`: which corner of the plane. |

## Text

```py
grid.text(surface, text, position, *, anchor="center", gap=0, color=None,
           size=None, opacity=None, halo=None, style=None)
```

```python
grid.text(screen, "A", (1, 2))                                   # centred on (1, 2)
grid.points(screen, (np.pi / 2, 1))
grid.text(screen, "maximum", (np.pi / 2, 1), anchor="bottom", gap=8)
grid.text(screen, "y = sin x", (4, -1), anchor="left", color="crimson")
grid.text(screen, "two lines\nof text", (-3, 2), anchor="top-left", size=16)
```

Text is written at a world point and moves with it while you pan and zoom,
keeping its size in pixels, like a marker. `anchor` is the point of the
text's box that is placed at `position`: `"bottom"` puts the text above the
point, `"left"` to its right (the point is at the text's left), `"top-left"`
below and to its right, and so on round the box:

| Anchor | The text is | Anchor | The text is |
|---|---|---|---|
| `"center"` | centred on the point | | |
| `"left"` | to the right | `"right"` | to the left |
| `"top"` | below | `"bottom"` | above |
| `"top-left"` | below right | `"top-right"` | below left |
| `"bottom-left"` | above right | `"bottom-right"` | above left |

`gap` moves the text that many pixels further from the point, on the
anchored sides, so that a label sits beside a marker instead of on it:
`gap=8` clears a default point. The box runs from the top of the letters
to the lowest reach of the font's descenders, so text with and without a
`g` or `y` sits alike. `"\n"` starts a new line; lines are aligned to the
anchor's side (left-aligned for the left anchors, right-aligned for the
right ones, centred otherwise).

To label something that moves, write the text at its position every frame.
A draggable point's label, or a value read from a slider, is one line:

```python
p = grid.draggable((1, 1))                                       # before the loop

grid.draw_controls(screen, p)
grid.text(screen, f"({p.x:.1f}, {p.y:.1f})", p.position, anchor="bottom-left", gap=10)
```

To place text beside a point *away* from a shape (a vertex name outside
its triangle, say), choose the anchor from the direction to the point;
`examples/annotations.py` does this.

Text is drawn with the theme's label font and, unless given a colour, its
label colour: text never takes [palette](styles-and-themes.md#the-palette)
colours, so labelling a plot does not change the colours of the plots
drawn after it (pass the plot's colour to match it). The label halo, an
outline in the background colour, keeps it readable over grid lines and
curves; `halo=False` turns it off. A position that is NaN (or not positive
on a log axis) draws nothing. Text is not rotated, and has no
typesetting beyond lines: use Unicode for symbols (`π`, `²`, `≤`).

| Parameter | Default | Meaning |
|---|---|---|
| `text` | | The text; `"\n"` starts a new line. |
| `position` | | The world point `(x, y)` the text is anchored to. |
| `anchor` | `"center"` | Which point of the text's box is at `position` (above). |
| `gap` | `0` | Pixels between the point and the text, on the anchored sides. |
| `color` | label colour | Text colour. |
| `size` | label size | Font size in points (a whole number). |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `halo` | `True` | Outline the glyphs in the background colour. |
| `style` | `None` | A reusable `TextStyle`. |

## Legends

```py
grid.legend(surface, *, corner="top-right")
```

```python
grid.plot(screen, np.sin, label="sin x")
grid.plot(screen, np.cos, label="cos x")
grid.fill_between(screen, np.sin, np.cos, label="between")
grid.points(screen, [(0, 0), (np.pi, 0)], label="roots of sin")
grid.legend(screen)
```

Every plotting method takes a `label=`. The legend lists the labelled calls
made since the frame began (at `grid.draw`), in order, each beside a small
sample of the plot in its own style and colour: a line for curves,
polylines and slope fields, a filled box for areas and regions, a box with
its outline for polygons, a marker for points, a stem with its marker for
sequences, and an arrow for arrows and vector fields. Thick lines and big
markers are drawn smaller there, so they fit their rows. Calls without a
label are left out, so a legend can name just the plots that matter.

The legend is redrawn every frame from that frame's calls, like everything
else, so it always shows what is on the plane: a plot that comes and goes
comes and goes from the legend too. A call repeated exactly, with the same
label, style and colour, is listed once, so a plot drawn in several pieces
can label every piece. With no labelled calls, nothing is drawn.

Call `legend` after the plots, so it is on top of them. It is drawn in a
panel in a corner of the plane, the same as the panels of
[sliders](draggable-points.md#sliders), and like them it stays put while
you pan and zoom; presses on it never pan the plane, and draggable points
under it cannot be grabbed. A legend in the same corner as sliders sits
beyond their panel, so the sliders never move. The labels use the theme's
label font and colour.

| Parameter | Default | Meaning |
|---|---|---|
| `corner` | `"top-right"` | `"top-right"`, `"top-left"`, `"bottom-left"` or `"bottom-right"`. |

| Plotting call's argument | Default | Meaning |
|---|---|---|
| `label` | `None` | The plot's name in the legend: one line of text, or `None` (or `""`) for none. |

## Drawing order

Plots are drawn in the order you call the methods, each on top of the ones
before. Draw fills first and points last so that nothing important is
covered, and draw text, the legend and your own overlays (captions) after
the plots.
Opacity blends properly with what is underneath, including on surfaces
with per-pixel alpha, so translucent layers can be composited later.

## Reusing unchanged plots

Most frames draw the same plots in the same view as the frame before, or
change only some of them. The plane notices and reuses those plots instead
of sampling and rasterising them again, which makes such frames several
times faster (an idle frame with six functions: about 10 ms without, under
2 ms with). There is nothing to do to get this, and nothing to change in
your code: plots are still drawn every frame, with new lambdas and all.

Reuse is exact, never a guess. A plotting call is reused when it is the same
kind of call in the same place among the frame's calls, with the same view,
style, colour, ranges and data as in the previous frame, and its functions
still give exactly the same values at every point where they were evaluated
last time. Sampling is deterministic, so it would then retrace its steps and
draw exactly the same thing. This is checked by calling each function once
on all those points, which costs a small fraction of drawing it.

So a function that reads a changing variable is drawn again as soon as the
variable changes, and only then:

```python
a = grid.draggable((1, 1))                                   # before the loop

grid.plot(screen, np.sin)                            # reused every frame
grid.plot(screen, lambda x: a.y * np.sin(x - a.x))   # drawn again while a moves
```

Animated plots (whose values change every frame) are simply drawn again;
telling them apart costs a quick check of a few dozen points.

`grid.cache_stats` counts the reused calls (`hits`) and the rest (`misses`),
which helps to see what a frame costs. `Grid(cache=False)` (or
`grid.cache = False`) turns reuse off; the drawing is the same, only
slower. That is useful to measure how long drawing takes, and needed only
for functions that are not *pointwise*, whose value at `x` depends on the
other points they are evaluated with (normalising by the largest value of
their input, say), which are not well-defined plots anyway.
[Antiderivatives](#integrals-and-antiderivatives) are pointwise only to the
last few digits, so they are checked through the function they integrate
instead: a plotted antiderivative is reused when its lower limit is the
same and its integrand gives the same values everywhere it was evaluated.

What is reused:

- function plots, parametric, polar and implicit curves, derivatives,
  antiderivatives, Taylor polynomials and Fourier series, areas between
  graphs, inequalities, vector and slope fields, sequences, and points
  (with their colours, sizes and error bars), polylines, polygons,
  arrows, bars, histograms, box and violin plots (their data is compared,
  and may be changed in place);
- the rasterised pixels of each reused stroke and fill, so only blending
  remains;
- the grid, while the view, the theme and the axis titles stay the same,
  and an axis' arrowhead while it does not move;
- the legend's panel and each of its samples, while they stay the same
  (a label that changes is only written again), and the colour bar.

Panning and zooming change the view, so during them everything is drawn
again (and as fast as before).

## Log axes

Everything above works on [logarithmic axes](log-axes.md) too. Functions
and curves are sampled in log space, so a power law is an exactly straight
line on log-log axes, and implicit curves are found on a grid that is even
on screen. Points, polylines, polygons and arrows are transformed vertex
by vertex and joined by straight lines on screen. Vector and slope fields
are drawn on a lattice that is even on screen, with arrows and segments
tangent to the flow lines and solution curves. Sequences keep their terms
at the integers, thinned where a log x axis crowds them together. Bars and
histogram bins are equal on screen on a log x axis (their widths are in
decades), and stand on the bottom edge on a log y axis.
