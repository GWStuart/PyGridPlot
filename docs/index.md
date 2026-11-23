# PyGridPlot documentation

PyGridPlot is an interactive Cartesian plane for
[pygame-ce](https://pyga.me/): an adaptive grid with clean tick labels,
smooth panning and zooming, linear or logarithmic axes, and robust,
antialiased plots of functions, parametric, polar and implicit curves,
areas between graphs and inequalities, vector and slope fields,
sequences, points, polylines, polygons and arrows, data (scatter plots
coloured by a value, error bars, step plots, bar charts, histograms, box
and violin plots), with axis titles, category names, legends, colour bars
and text, and points and sliders the user can drag. Your program
owns the main loop; the plane is an object you call every frame.

The [philosophy](philosophy.md) explains the principles behind it: one way
to do things, done well; Python first; a library, not a framework; and
immediate mode.

## For users

Start with [Getting started](user/getting-started.md), then read whichever
topic you need:

| Page | What it covers |
|---|---|
| [Getting started](user/getting-started.md) | Installing, the first program, the frame loop, quick mode. |
| [The plane](user/the-plane.md) | Creating a plane, setting up the view, coordinates, embedding, several planes. |
| [Interaction](user/interaction.md) | The built-in controls, configuring them, moving the view from code. |
| [Plotting](user/plotting.md) | Functions, their derivatives and integrals, Taylor polynomials and Fourier series, parametric, polar and implicit curves, areas and inequalities, vector and slope fields, sequences, points, polylines, polygons and arrows; data plots (error bars, steps, bars, histograms, box and violin plots, colour by value); text and legends. |
| [Draggable points and sliders](user/draggable-points.md) | Points the user drags: free, locked to an axis, or on a curve; sliders for values. |
| [Logarithmic axes](user/log-axes.md) | Log and semi-log planes, how their grid and labels behave. |
| [Styles and themes](user/styles-and-themes.md) | Colours, reusable styles, the palette, themes and number formats. |
| [Recipes](user/recipes.md) | Worked examples of common tasks, performance tips and troubleshooting. |
| [API reference](user/reference.md) | Every public class, function and setting, generated from the code. |

## For developers

The [developer guide](dev/index.md) explains how the code is organised and
how to extend it:

| Page | What it covers |
|---|---|
| [Architecture](dev/architecture.md) | Design principles, layers, the package map, what happens in a frame. |
| [The grid](dev/grid.md) | Spacing, lines, labels, layout and log axes, step by step. |
| [Plotting](dev/plotting.md) | Robust evaluation, the samplers, exact derivatives (Taylor arithmetic), adaptive quadrature, building primitives. |
| [Draggable points and sliders](dev/dragging.md) | Events, constraints, following curves, labels, slider panels and the legend's. |
| [Rendering](dev/rendering.md) | The renderer protocol, rasterising, compositing, text and performance. |
| [Testing and tools](dev/testing.md) | How the code is tested, and the visual and benchmark tools. |
| [Extending PyGridPlot](dev/extending.md) | Step-by-step recipes for new features, and the coding conventions. |
