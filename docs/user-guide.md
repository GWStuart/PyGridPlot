# PyGridPlot user guide

PyGridPlot draws an interactive Cartesian plane into a pygame window: an
adaptive grid with clean tick labels, smooth panning and zooming, and plots of
functions, points, polylines and polygons. It is a *library, not a framework*:
your program owns the main loop, and the plane is an object you call every
frame.

- [Getting started](#getting-started)
- [The plane](#the-plane)
- [Controls](#controls)
- [Plotting functions](#plotting-functions)
- [Points](#points)
- [Polylines and polygons](#polylines-and-polygons)
- [Colours and styles](#colours-and-styles)
- [Themes](#themes)
- [The view](#the-view)
- [Quick mode](#quick-mode)
- [Tips](#tips)

## Getting started

Install the package (Python 3.10+, with pygame-ce and NumPy):

```bash
pip install -e .            # from the repository root
```

A complete program:

```python
import numpy as np
import pygame
import pygridplot as pgp

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
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
    plane.points(screen, [(0, 0), (np.pi, 0)])
    pygame.display.flip()
    clock.tick(60)
```

Everything is **immediate mode**: each frame you draw what you want to see,
and the plane recomputes it from the current view. Animation is therefore just
a matter of drawing different things each frame, and plots follow panning and
zooming automatically.

Each frame, in this order:

1. Pass **every** event to `plane.handle_event(event)`. Mouse motion
   matters too, because the plane zooms about the cursor. It returns `True`
   when the plane used the event, so you can skip your own handling of it.
2. Call `plane.draw_grid(surface)` once. It also advances eased movement and
   restarts the colour palette (see [Colours](#colours-and-styles)).
3. Draw plots with `plot`, `points`, `polyline` and `polygon`.

All drawing methods take the target surface first, like `pygame.draw`.

## The plane

```python
pgp.Plane(rect=None, *, center=(0, 0), scale=50.0, equal_aspect=True,
          theme=None, interactive=True, interaction=None, limits=None,
          renderer=None, sampling=None)
```

| Parameter | Meaning |
|---|---|
| `rect` | Screen rectangle to draw into (a `pygame.Rect` or `(x, y, w, h)`). `None` fills the surface given to `draw_grid` and follows window resizes. |
| `center` | World point shown at the middle of the plane. |
| `scale` | Pixels per world unit. One number, or `(x_scale, y_scale)` when `equal_aspect=False`. |
| `equal_aspect` | Keep both axes at the same scale, so circles look round. |
| `theme` | A [`GridTheme`](#themes); default `pgp.LIGHT`. |
| `interactive` | Whether `handle_event` reacts at all. Also an attribute you can toggle. |
| `interaction` | [`InteractionConfig`](#controls) for the built-in controls. |
| `limits` | [`ViewLimits`](#the-view) on zooming and panning. |
| `renderer` | A `PygameRenderer` to share between planes (they then share font and label caches). |
| `sampling` | Default [`SamplingConfig`](#plotting-functions) for `plot`. |

Useful members:

| Member | Meaning |
|---|---|
| `plane.rect` | The plane's rectangle. Assign a rect to move or resize it, or `None` to fill the surface again. Resizing keeps the centre and scale. |
| `plane.view` | The [`View`](#the-view): centre, scale, bounds, transforms. |
| `plane.theme`, `plane.interaction`, `plane.sampling` | Read or assign at any time. |
| `plane.to_world(px, py)` | Screen pixels (for example the mouse) to world coordinates. Accepts arrays. |
| `plane.to_screen(x, y)` | World coordinates to screen pixels. Accepts arrays. |
| `plane.set_view(x_range=None, y_range=None)` | Fit the view to world ranges, such as `x_range=(-1, 1)`. |
| `plane.reset_view()` | Return to the home view (the starting view, unless changed). |
| `plane.set_home(state=None)` | Make the current view (or a saved `ViewState`) the home view. |
| `plane.begin_frame()` | Restart the colour palette (`draw_grid` does this). |
| `plane.update()` | Advance eased movement (`draw_grid` does this). |
| `plane.last_grid` | The last grid drawn, including the chosen spacings (useful for debug overlays). |

**Embedding.** Give the plane a `rect` to draw it into part of the window.
Its drawing is clipped to that rectangle, and it only reacts to mouse input
inside it. See `examples/embedded_rect.py`.

## Controls

| Input | Action |
|---|---|
| Left-drag | Pan; the grabbed point stays exactly under the cursor |
| Mouse wheel / trackpad | Zoom about the cursor (eased) |
| Shift / Ctrl + wheel | Stretch only x / only y (needs `equal_aspect=False`) |
| Arrow keys | Pan by 10% of the plane (eased) |
| `+` / `-` | Zoom about the middle (eased) |
| `Home` / `0` | Reset the view |

Keys are ignored while Ctrl, Alt or Meta is held, so your own shortcuts are
safe. Customise the controls with `InteractionConfig` (all fields optional):

| Field | Default | Meaning |
|---|---|---|
| `pan_button` | `1` | Mouse button that drags (1 = left). |
| `wheel_zoom` | `True` | Whether the wheel zooms. |
| `zoom_per_notch` | `1.15` | Zoom factor per wheel notch (trackpads zoom proportionally). |
| `invert_wheel` | `False` | Reverse the wheel direction. |
| `keyboard` | `True` | Whether the keyboard controls are active. |
| `key_pan_fraction` | `0.1` | Arrow-key pan as a fraction of the plane's size. |
| `key_zoom_factor` | `1.25` | Zoom factor per `+`/`-` press. |
| `smoothing` | `0.05` | Easing time constant in seconds (90% of a move takes about 4×). `0` moves instantly. Use `0` if you only redraw when events arrive. |
| `bindings` | `KeyBindings()` | Key codes per action: `pan_left`, `pan_right`, `pan_up`, `pan_down`, `zoom_in`, `zoom_out`, `reset` (tuples of `pygame.K_*`; empty disables). |

```python
plane = pgp.Plane(interaction=pgp.InteractionConfig(zoom_per_notch=1.3, keyboard=False))
```

## Plotting functions

```python
plane.plot(surface, func, *, color=None, width=None, opacity=None,
           domain=None, style=None, sampling=None)
```

`func` maps x to y. It can be:

- **vectorised** (written with NumPy, such as `np.sin` or `lambda x: x**2`).
  This is the fastest kind, called once per batch of points.
- **scalar-only** (written with `math`, or with `if` statements). It is
  detected automatically and called point by point.

It doesn't need to be well behaved. Points where it raises an error, returns
NaN or infinity, or returns a complex number with a real imaginary part are
left out, and the frame never crashes. Jumps (`floor`) and poles (`1/x`,
`tan`) are not joined by vertical lines. Domain edges are found exactly
(`sqrt` starts at the origin), and the branches of a pole run to the edge of
the plane.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Line colour. |
| `width` | `2.5` | Line width in pixels (fractional widths are fine). |
| `opacity` | `1.0` | Line opacity, 0 to 1. |
| `domain` | `None` | Only plot where `domain[0] <= x <= domain[1]`. |
| `style` | `None` | A reusable `LineStyle`; the keyword arguments override it. |
| `sampling` | plane default | A `SamplingConfig` for this call. |

Sampling is adaptive and automatic. You'll rarely need `SamplingConfig`, but
these are its fields:

| Field | Default | Meaning |
|---|---|---|
| `samples_per_px` | `1.0` | Initial samples per pixel column. |
| `tolerance_px` | `0.25` | Largest allowed distance between the curve and the drawn line. |
| `max_rounds` | `8` | Refinement depth (jumps are located to 1/256 px). |
| `max_samples` | `16384` | Budget of evaluations per function per frame. |
| `jump_px` | `1.0` | Smallest unresolved rise treated as a discontinuity. |

> **Late binding.** When creating functions in a loop, bind the loop variable
> as a default argument: `[lambda x, k=k: x**k for k in range(3)]`. Without
> the `k=k`, all three functions use the last `k`. The same applies to
> animation variables captured by a lambda inside a loop.

## Points

```python
plane.points(surface, points, *, color=None, radius=None, marker=None,
             edge_color=None, edge_width=None, opacity=None, style=None)
```

`points` can be a single point `(x, y)`, a sequence of points, or an `(N, 2)`
array (use `np.column_stack((xs, ys))` for separate arrays). Points with a
NaN coordinate are skipped. Markers keep the same size in pixels at every zoom
level, and are positioned to a quarter of a pixel so they glide smoothly.

| Parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Marker colour. |
| `radius` | `4.5` | Size in pixels (non-circular shapes have the same area as the circle). |
| `marker` | `"circle"` | `"circle"`, `"square"`, `"diamond"`, `"cross"` or `"plus"`. |
| `edge_color` | `None` | Outline colour, drawn *outside* the marker. A background-coloured outline (such as `"white"`) separates points from the curves they sit on. |
| `edge_width` | `1.5` | Outline thickness in pixels (only used with `edge_color`). |
| `opacity` | `1.0` | Opacity, 0 to 1. |
| `style` | `None` | A reusable `PointStyle`. |

## Polylines and polygons

```python
plane.polyline(surface, points, *, color=None, width=None, opacity=None,
               closed=False, style=None)
plane.polygon(surface, points, *, color=None, width=None, opacity=None,
              fill=None, fill_color=None, fill_opacity=None, fill_rule=None,
              style=None)
```

Vertices are in world units, so these shapes grow and shrink with the zoom;
only line widths are in pixels. Lines have round joins and ends, and are
antialiased at any width.

- **`polyline`** connects the vertices in order. A vertex with a NaN
  coordinate breaks the line (like lifting the pen). Set `closed=True` to
  join the last vertex back to the first. Takes `LineStyle` fields.
- **`polygon`** draws a translucent fill with an outline. It closes itself
  automatically.

| Polygon parameter | Default | Meaning |
|---|---|---|
| `color` | next palette colour | Outline colour, also used for the fill. |
| `width` | `2.5` | Outline width in pixels; `0` for no outline. |
| `opacity` | `1.0` | Outline opacity. |
| `fill` | `True` | Whether to fill. |
| `fill_color` | same as `color` | Fill colour. |
| `fill_opacity` | `0.25` | Fill opacity. |
| `fill_rule` | `"nonzero"` | For self-crossing outlines: `"nonzero"` fills everything enclosed, `"evenodd"` leaves doubly enclosed areas (such as a star's centre) empty. |
| `style` | `None` | A reusable `PolygonStyle`. |

## Colours and styles

Colours can be anything pygame accepts: names (`"crimson"`), hex strings
(`"#2d70b3"`), RGB or RGBA tuples, or `pygame.Color`. `opacity` multiplies a
colour's own alpha.

**Palette.** Objects drawn without a colour take the theme's palette
colours in order: the first plot of a frame gets the first colour, and so on.
`draw_grid` restarts the sequence each frame, so every object keeps its
colour. If you draw plots without calling `draw_grid`, call
`plane.begin_frame()` at the start of each frame instead. Explicit colours
don't use up palette slots.

**Styles.** Keyword arguments are handy for one-off calls. For reuse, build
a style object and override fields per call:

```python
thin = pgp.LineStyle(width=1.5, opacity=0.6)
plane.plot(screen, np.sin, style=thin)
plane.plot(screen, np.cos, style=thin, color="crimson")   # override one field

big = pgp.PointStyle(radius=7, marker="diamond", edge_color="white")
plane.points(screen, data, style=big)
```

Styles are validated when they are created, so a mistake such as a negative
width raises a clear `ValueError` straight away.

## Themes

`pgp.LIGHT` (the default) and `pgp.DARK` are ready-made. Themes are frozen
dataclasses: derive variations with `dataclasses.replace` and assign them at
any time.

```python
from dataclasses import replace
plane.theme = replace(pgp.DARK, axis_color=(120, 190, 255), target_major_px=120)
```

| Field | Default (`LIGHT`) | Meaning |
|---|---|---|
| `background` | white | Fill colour, or `None` to draw over whatever is on the surface. |
| `minor_color`, `major_color`, `axis_color` | light grey, grey, near black | Grid line and axis colours. |
| `label_color` | near black | Tick label colour. |
| `label_halo_color` | `None` | Outline around label glyphs; `None` uses `background`. |
| `minor_width`, `major_width`, `axis_width` | `1`, `1`, `2` | Line thickness in pixels. |
| `target_major_px` | `100` | Preferred distance between major lines. |
| `spacing_rule` | `"nearest"` | Round spacings to the nearest 1-2-5 step, or `"ceil"` to round up. |
| `transition` | `"fade"` | `"fade"` cross-fades between spacings while zooming; `"snap"` switches instantly. |
| `fade_band` | `0.05` | Width of each cross-fade, in decades of zoom (about ±12%). |
| `hysteresis` | `0.04` | Extra zoom needed before a `"snap"` switch reverses. |
| `show_minor`, `show_major`, `show_axes`, `show_labels` | `True` | Turn parts of the grid on or off. |
| `label_font` | `None` | `None` for the bundled DejaVu Sans, a path to a font file, or a system font name. |
| `label_size` | `12` | Label font size in points. |
| `label_halo_width` | `2` | Halo thickness in pixels (`0` disables it). |
| `label_gap` | `4` | Pixels between an axis and its labels. |
| `label_edge_margin` | `4` | Margin between pinned labels and the plane's edge. |
| `label_min_separation` | `14` | Comfortable gap between labels. Closer labels fade out and are thinned. |
| `hang_minus` | `True` | Centre x-labels on their digits and let the minus sign hang. |
| `number_format` | `NumberFormat()` | How numbers are written (below). |
| `palette` | six colours | Default colours for plotted objects. |

`NumberFormat` fields: `sci_large_exponent` (default `6`) and
`sci_small_exponent` (default `-4`) set where scientific notation may start.
Each axis then picks whichever notation is shorter, so labels never mix
`500000` and `1×10⁶`. `minus` and `times` set the characters used
(`"−"` and `"×"`).

## The view

`plane.view` is the camera: it holds the centre and scale and does every
coordinate conversion. Use it for programmatic control:

| Member | Meaning |
|---|---|
| `center`, `center_x`, `center_y` | World point at the middle of the plane (`center` is assignable). |
| `scale`, `scale_x`, `scale_y` | Pixels per world unit (`scale` requires equal axes). |
| `set_scale(sx, sy=None)` | Set the scale, keeping the centre. |
| `bounds` | Visible region: `x_min`, `x_max`, `y_min`, `y_max`, `width`, `height`. |
| `to_screen(x, y)`, `to_world(px, py)` | Conversions (also `to_screen_x` and friends); arrays welcome. |
| `pan_pixels(dx, dy)`, `pan_world(dx, dy)` | Move the view. |
| `zoom(factor)`, `zoom_at(px, py, factor)` | Zoom about the middle, or about a pixel. |
| `fit(x_range, y_range)` | Same as `plane.set_view`. |
| `state()`, `restore(state)` | Save and restore a `ViewState`. |

Zooming is limited to what `float64` can render accurately. Customise the
limits with `ViewLimits` (`min_scale=1e-12`, `max_scale=1e14`,
`min_relative_pixel=1e-12`, `max_abs_center=1e15`) passed as
`Plane(limits=...)`.

## Quick mode

For a quick look at some functions, without writing a loop:

```python
pgp.show(np.sin, lambda x: x**2 / 4)
```

`show(*functions, size=(960, 640), title="PyGridPlot", theme=None,
center=(0, 0), scale=50.0, width=None, fps=60)` opens a resizable window and
runs until it is closed (or Escape is pressed).

## Tips

- **Examples.** `examples/` has runnable programs for everything above:
  `functions.py`, `shapes.py`, `animation.py`, `many_functions.py`,
  `embedded_rect.py`, `dark_theme.py`, and `grid_playground.py` (live theme
  toggles, timings, `P` cycles plots).
- **Performance.** A typical frame (grid plus a few functions) takes a few
  milliseconds. Vectorised functions are much faster than scalar-only ones.
  Functions that oscillate infinitely fast (such as `sin(1/x)` near 0) are
  the most expensive, at 10–20 ms when zoomed in, because their curve covers
  a lot of the screen.
- **Several planes** can share one renderer (`Plane(renderer=...)`), and
  with it their font and label caches. `renderer.clear_caches()` frees them.
