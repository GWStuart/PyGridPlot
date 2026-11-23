# Plotting

Every plotting method draws onto a surface, on top of whatever is already
there, and is called every frame after `grid.draw`. Everything is drawn
from the current view each time, so plots follow panning and zooming
exactly, and changing what you draw from one frame to the next animates it.
Plots that have not changed are [reused](#reusing-unchanged-plots)
automatically, so frames where little changes stay cheap.

The methods are described on five pages, by topic:

- [Functions and calculus](functions.md): `plot`, with derivatives,
  integrals, antiderivatives, Taylor polynomials and Fourier series, and
  `sequence`.
- [Curves](curves.md): `parametric`, `polar` and `implicit`, and contour
  maps.
- [Regions and fields](regions-and-fields.md): `fill_between`,
  `inequality`, `vector_field` and `slope_field`.
- [Data](data.md): `points` (scatter plots, error bars), `polyline` (lines
  and steps), `bars`, `histogram`, `box_plot`, `violin_plot`, and colour by
  value with `color_bar`.
- [Shapes, text and legends](shapes-and-text.md): `polygon`, `arrow`,
  `text` and `legend`.

This page lists them all, then covers what they have in common: the
[drawing order](#drawing-order), [reusing unchanged
plots](#reusing-unchanged-plots) and [log axes](#log-axes).

## All the methods

| Method | Draws |
|---|---|
| [`plot(surface, f)`](functions.md#functions) | The graph of `y = f(x)`, or of a [derivative](functions.md#derivatives), an [antiderivative](functions.md#integrals-and-antiderivatives), a [Taylor polynomial or a Fourier series](functions.md#taylor-polynomials-and-fourier-series). |
| [`parametric(surface, x, y, t_range)`](curves.md#parametric-curves) | The curve `(x(t), y(t))`. |
| [`polar(surface, r)`](curves.md#polar-curves) | The curve `r = r(θ)`. |
| [`implicit(surface, F)`](curves.md#implicit-curves) | The curve `F(x, y) = 0` (or `= c`). |
| [`fill_between(surface, f, g)`](regions-and-fields.md#areas-between-graphs) | The area between two graphs (or a graph and the x axis). |
| [`inequality(surface, condition)`](regions-and-fields.md#inequalities-and-regions) | The region where a condition such as `x² + y² < 4` holds. |
| [`vector_field(surface, field)`](regions-and-fields.md#vector-fields) | Arrows showing a vector field `(u, v) = F(x, y)`. |
| [`slope_field(surface, slope)`](regions-and-fields.md#slope-fields) | Segments showing the slopes of `dy/dx = f(x, y)`. |
| [`sequence(surface, terms)`](functions.md#sequences) | The terms `aₙ` of a sequence: a list or array, or a function of `n`. |
| [`points(surface, points)`](data.md#points) | Markers that keep their size in pixels: a scatter plot, with a colour and size per point and error bars. |
| [`polyline(surface, points)`](data.md#polylines) | Vertices joined by straight lines, or by steps. |
| [`polygon(surface, points)`](shapes-and-text.md#polygons) | A filled shape with an outline. |
| [`bars(surface, heights)`](data.md#bars) | Bars standing on the x axis: a bar chart. |
| [`histogram(surface, samples)`](data.md#histograms) | How many samples fall in each bin, or the density they estimate. |
| [`box_plot(surface, samples)`](data.md#box-and-violin-plots) | The quartiles, median and spread of groups of samples. |
| [`violin_plot(surface, samples)`](data.md#box-and-violin-plots) | The density of groups of samples, side by side. |
| [`color_bar(surface, colors, value_range)`](data.md#colour-by-value) | Which colour is which value, for points coloured by a value. |
| [`arrow(surface, start, end)`](shapes-and-text.md#arrows) | Arrows between points. |
| [`text(surface, text, position)`](shapes-and-text.md#text) | Text at a point, moving with it. |
| [`legend(surface)`](shapes-and-text.md#legends) | A legend naming the plots drawn with a `label=`. |

All of them accept `color`, `opacity` and a reusable `style`; see
[Styles and themes](styles-and-themes.md). Objects drawn without a colour
take the theme's [palette](styles-and-themes.md#the-palette) colours in
order. Every plotting method also takes a `label=`, the plot's name in the
[legend](shapes-and-text.md#legends). Everything is clipped to the plane's
rectangle. Axis titles, arrowheads on the axes and names along an axis are
part of the grid; see [The plane](the-plane.md#axis-titles-and-arrowheads).

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
[Antiderivatives](functions.md#integrals-and-antiderivatives) are pointwise only to the
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
