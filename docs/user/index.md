# User guide

How to use PyGridPlot, from a first program to every setting. Each page
stands on its own; the [API reference](reference.md) lists every parameter
and default.

1. [Getting started](getting-started.md): installing, the first program, the
   frame loop and quick mode.
2. [The plane](the-plane.md): creating a plane, setting up what it shows,
   converting coordinates, embedding a plane and using several.
3. [Interaction](interaction.md): the built-in pan and zoom controls,
   configuring them, and moving the view from your own code.
4. [Plotting](plotting.md): functions `y = f(x)` and their derivatives,
   parametric, polar and implicit curves, contour lines, points, polylines
   and polygons.
5. [Draggable points](draggable-points.md): points the user drags, freely,
   along one axis or along a curve.
6. [Graphs](graphs.md): lists of plots, curves, points and parameters
   written as maths (`"y = a sin(x - P.x)"`), kept between frames, edited
   while the program runs and saved as JSON.
7. [The expression language](expressions.md): the maths that graphs and
   `pgp.expression` read: operators, functions, derivatives, conditions.
8. [Logarithmic axes](log-axes.md): log-log and semi-log planes.
9. [Styles and themes](styles-and-themes.md): colours, reusable styles, the
   palette, themes and how numbers are written.
10. [Recipes](recipes.md): worked examples (mouse readouts, animation,
    legends, screenshots, ...), performance tips and troubleshooting.
11. [API reference](reference.md): everything `import pygridplot` provides.

The `examples/` folder of the repository has a runnable program for every
topic; [Recipes](recipes.md#examples) lists them.
