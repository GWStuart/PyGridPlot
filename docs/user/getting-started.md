# Getting started

## Installing

PyGridPlot needs Python 3.10 or newer, [pygame-ce](https://pyga.me/) 2.5 or
newer, NumPy, and [contourpy](https://contourpy.readthedocs.io/) (for
implicit curves); pip installs them all. From a checkout of the repository:

```bash
pip install -e .            # the library
pip install -e ".[dev]"     # plus the tools for working on it
```

Use pygame **community edition** (`pygame-ce`), not the original `pygame`
package: they install under the same import name, so uninstall `pygame`
first if both are present.

## A first program

```python
import numpy as np
import pygame
import pygridplot as pgp

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
pygame.display.set_caption("My first plane")
plane = pgp.Plane()                       # fills the window, follows resizes
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)         # drag to pan, scroll to zoom

    plane.draw_grid(screen)               # background, grid, axes, labels
    plane.plot(screen, np.sin)            # then anything on top of the grid
    plane.polar(screen, lambda a: 1 + np.cos(a))
    plane.points(screen, [(0, 0), (np.pi, 0)])
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
```

Run it, then drag with the left mouse button to pan, scroll to zoom about
the cursor, and press `Home` to return to the start. The grid adapts as you
zoom: lines stay about 100 pixels apart, labels never overlap, and new lines
fade in smoothly.

## The frame loop

The drawing methods are **immediate mode**: each frame you draw what you
want to see, and the plane recomputes it from the current view. There is
nothing to create, update or delete. Animation is just drawing something
different each frame, and plots follow panning and zooming automatically.
(If you would rather make each plot once and change it later, see
[Objects](#objects) below.)

Each frame does three things, in this order:

1. **Pass every event** to `plane.handle_event(event)`, including mouse
   motion (the plane zooms about the cursor, so it tracks the mouse). It
   returns `True` when the plane used the event, so your own code can skip
   it; see [Interaction](interaction.md#handling-your-own-input).
2. **Draw the grid once** with `plane.draw_grid(surface)`. It fills the
   background, draws the grid, axes and labels, advances eased pan and zoom,
   and restarts the colour palette for the new frame.
3. **Draw plots** with `plot`, `parametric`, `polar`, `points`,
   `polyline` and `polygon`, in the order they should be stacked. Each takes
   the target surface first, like `pygame.draw`.

Then flip the display as usual. Anything drawn after the plots (captions,
buttons, a heads-up display) appears on top of the plane.

## Objects

The same plots can be made once, as objects, and kept by the plane; you
change their attributes when you want something different, and one call
draws the whole frame:

```python
f = plane.add(pgp.Function(np.sin, color="crimson"))
p = plane.add(pgp.Point((1, 1), draggable=True))
g = plane.add(pgp.Function(lambda x: p.y * np.cos(x - p.x)))   # follows p

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)
    plane.draw(screen)        # the grid, f, g and p
    pygame.display.flip()
```

Both styles mix on one plane; see [Objects](objects.md).

## Quick mode

For a quick look at some functions, without writing a loop:

```python
pgp.show(np.sin, lambda x: x**2 / 4)
pgp.show(np.exp, lambda x: x**3, log_y=True, x_range=(0, 10))
```

`show` opens a resizable window, plots the functions in palette colours and
runs until the window is closed or Escape is pressed. It also takes lines
of maths (`pgp.show("y = x^2", "x^2 + y^2 = 9")`), objects and graphs. It takes the same
view options as `Plane` (`x_range`, `y_range`, `log_x`, `log_y`, `center`,
`scale`), plus `size`, `title`, `theme`, `width` and `fps`; see the
[reference](reference.md#show).

## Where next

- To control what the plane shows at first (for example x from 0 to 10),
  see [Setting up the view](the-plane.md#setting-up-the-view).
- To plot your own functions and data, see [Plotting](plotting.md), or
  [Objects](objects.md) to keep them as objects.
- For maths typed as text, with names and saving, see [Graphs](graphs.md).
- For log scales, see [Logarithmic axes](log-axes.md).
- To change colours, line widths or the look of the grid, see
  [Styles and themes](styles-and-themes.md).
- For worked examples of common tasks, see [Recipes](recipes.md).
