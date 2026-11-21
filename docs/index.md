# PyGridPlot documentation

PyGridPlot is an interactive Cartesian plane for
[pygame-ce](https://pyga.me/): an adaptive grid with clean tick labels,
smooth panning and zooming, linear or logarithmic axes, and robust,
antialiased plots of functions, parametric, polar and implicit curves,
points, polylines and polygons, and points the user can drag. Your program
owns the main loop; the plane is an object you call every frame, drawing
in immediate mode or keeping objects (`pgp.Function`, `pgp.Point`, ...)
whose attributes you change. Graphs hold such objects written as maths
(`"y = a sin(x - P.x)"`), edited while the program runs and saved as JSON.

## For users

Start with [Getting started](user/getting-started.md), then read whichever
topic you need:

| Page | What it covers |
|---|---|
| [Getting started](user/getting-started.md) | Installing, the first program, the frame loop, quick mode. |
| [The plane](user/the-plane.md) | Creating a plane, setting up the view, coordinates, embedding, several planes. |
| [Interaction](user/interaction.md) | The built-in controls, configuring them, moving the view from code. |
| [Plotting](user/plotting.md) | Functions and their derivatives, parametric, polar and implicit curves, points, polylines and polygons. |
| [Draggable points](user/draggable-points.md) | Points the user drags: free, locked to an axis, or on a curve. |
| [Objects](user/objects.md) | Functions, curves and points made once and changed through their attributes. |
| [Graphs](user/graphs.md) | Lists of plots, curves, points and parameters written as maths, edited live and saved as JSON. |
| [The expression language](user/expressions.md) | Operators, functions, derivatives and conditions in the maths that graphs read. |
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
| [Plotting](dev/plotting.md) | Robust evaluation, the samplers, exact derivatives (Taylor arithmetic), building primitives. |
| [Draggable points](dev/dragging.md) | Events, constraints, following curves, labels. |
| [Objects, graphs and expressions](dev/graphs.md) | The persistent objects and their two-phase drawing, graphs, parsing and compiling maths, the JSON format. |
| [Rendering](dev/rendering.md) | The renderer protocol, rasterising, compositing, text and performance. |
| [Testing and tools](dev/testing.md) | How the code is tested, and the visual and benchmark tools. |
| [Extending PyGridPlot](dev/extending.md) | Step-by-step recipes for new features, and the coding conventions. |
