# The plane

A `Plane` is the whole interactive plane: what part of the world it shows
(its [view](#the-view)), how it looks (its [theme](styles-and-themes.md)),
its [controls](interaction.md) and the renderer that draws it.

```py
pgp.Plane(rect=None, *, x_range=None, y_range=None, log_x=False, log_y=False,
          center=None, scale=None, equal_aspect=None, theme=None,
          interactive=True, interaction=None, limits=None, renderer=None,
          sampling=None)
```

| Parameter | Meaning |
|---|---|
| `rect` | Screen rectangle to draw into (a `pygame.Rect` or `(x, y, w, h)`). `None` fills the surface given to `draw_grid` and follows window resizes. |
| `x_range`, `y_range` | `(min, max)` to show across the plane's width or height at first, like a graphing calculator's window settings. They override `center` and `scale` for that axis. |
| `log_x`, `log_y` | Make an axis [logarithmic](log-axes.md) (base 10). |
| `center` | World point shown at the middle of the plane. Default: 0 on a linear axis, 1 on a log axis. |
| `scale` | Pixels per world unit (per decade on a log axis). One number, or `(x_scale, y_scale)` for unequal axes. Default: 50 per unit, 150 per decade. |
| `equal_aspect` | Keep both axes at the same scale, so circles look round. By default (`None`) it is on, unless both ranges are given or only one axis is logarithmic. |
| `theme` | A [`GridTheme`](styles-and-themes.md#themes); default `pgp.LIGHT`. |
| `interactive` | Whether `handle_event` reacts at all. Also an attribute you can toggle. |
| `interaction` | [`InteractionConfig`](interaction.md#configuring-the-controls) for the built-in controls. |
| `limits` | [`ViewLimits`](#zoom-limits) on zooming and panning. |
| `renderer` | A `PygameRenderer` to share between planes (they then share font and label caches). |
| `sampling` | Default [`SamplingConfig`](plotting.md#sampling-settings) for `plot`. |

## Setting up the view

There are two ways to say what a plane shows at first. **Ranges** say which
values should span the plane, and are usually the most convenient:

```python
plane = pgp.Plane(x_range=(-2, 8), y_range=(-1, 3))     # each axis its own scale
plane = pgp.Plane(x_range=(0, 10))                      # equal aspect, y centred on 0
```

With both ranges given, the axes get different scales so that each range
fills its axis exactly, and zooming afterwards keeps the ratio between them.
With only one range, the other axis uses the same scale (circles stay
round) and is centred on `center`.

A **centre and scale** say where to look and how far to zoom in:

```python
plane = pgp.Plane(center=(3, 1), scale=80)              # 80 pixels per unit
plane = pgp.Plane(scale=(40, 400), equal_aspect=False)  # y stretched ten times
```

You can mix the two: `Plane(x_range=(0, 10), center=(0, 5))` fits x and
centres y on 5. Whatever you choose becomes the **home view**, which
`plane.reset_view()` and the `Home` key return to.

Ranges are fitted once the plane knows its real size in pixels. A plane
that fills the window (`rect=None`) learns that when it is first drawn, so
it is fine to create it before the window has its final size; if you move
the view yourself before the first frame, your change is kept.

To change the view later, use `set_view` (it takes the same ranges) or the
[view](#the-view) directly:

```python
plane.set_view(x_range=(-1, 1), y_range=(-1, 1))
plane.view.center = (2.0, 3.0)
plane.view.zoom(2.0)          # twice as close, about the middle
plane.reset_view()            # back to the home view
plane.set_home()              # make the current view the home view
```

With equal aspect on, `set_view` fits both ranges using the tighter of the
two, so the other axis shows extra space.

### Equal aspect

With equal aspect on, one unit is the same number of pixels on both axes,
so circles are round and angles are true. Turn it off for data whose axes
have different units or ranges (time against temperature, say). The
default is on, except when both ranges are given (each range then fills its
axis exactly) or one axis is logarithmic (a pixel per unit and a pixel per
decade cannot sensibly be equal). With it off, `Shift` and `Ctrl` + wheel
stretch one axis at a time.

## Coordinates

**World coordinates** are the plane's own: ordinary Cartesian coordinates
with y pointing up. **Screen coordinates** are pygame pixels, with y
pointing down; integer values are pixel centres. Convert between them with:

```python
x, y = plane.to_world(*pygame.mouse.get_pos())   # where the mouse is
px, py = plane.to_screen(1.0, 2.0)               # where (1, 2) is drawn
xs, ys = plane.to_screen(np.linspace(0, 1, 5), np.zeros(5))   # arrays work too
```

The visible region is `plane.view.bounds`, with `x_min`, `x_max`, `y_min`,
`y_max`, `width` and `height`:

```python
b = plane.view.bounds
print(f"showing x from {b.x_min:.3g} to {b.x_max:.3g}")
```

## Embedding a plane

Give the plane a `rect` to draw it into part of the window, for example
beside a side panel:

```python
plane = pgp.Plane(pygame.Rect(200, 0, 700, 600))
```

Everything it draws is clipped to its rectangle, and it only reacts to mouse
input inside it (a drag that starts inside continues outside). Move or
resize it by assigning `plane.rect`; assign `None` to make it fill the
surface again. Resizing keeps the centre and scale, so more or less of the
plane becomes visible. See `examples/embedded_rect.py`.

## Several planes

Planes are independent: give each its own `rect` and pass every event to
each of them (each only reacts to the mouse inside its own rectangle). They
can share one renderer, and with it their font and label caches:

```python
renderer = pgp.Plane().renderer
left = pgp.Plane(pygame.Rect(0, 0, 450, 600), renderer=renderer)
right = pgp.Plane(pygame.Rect(450, 0, 450, 600), renderer=renderer, log_y=True)
```

Keyboard controls act on every plane that receives the event, so for
several planes you may want `InteractionConfig(keyboard=False)` on all but
one; see [Interaction](interaction.md).

## The view

`plane.view` is the camera. It holds the centre and scale and does every
coordinate conversion. The most useful members:

| Member | Meaning |
|---|---|
| `center`, `center_x`, `center_y` | World point at the middle of the plane (`center` is assignable). |
| `scale`, `scale_x`, `scale_y` | Pixels per world unit, or per decade on a log axis (`scale` requires equal axes). |
| `set_scale(sx, sy=None)` | Set the scale, keeping the centre. |
| `bounds` | Visible region: `x_min`, `x_max`, `y_min`, `y_max`, `width`, `height`. |
| `to_screen(x, y)`, `to_world(px, py)` | Conversions (also `to_screen_x` and friends); arrays welcome. |
| `pan_pixels(dx, dy)`, `pan_world(dx, dy)` | Move the view. |
| `zoom(factor)`, `zoom_at(px, py, factor)` | Zoom about the middle, or about a pixel. |
| `fit(x_range, y_range)` | Same as `plane.set_view`. |
| `log_x`, `log_y`, `set_log(axis, log)` | Whether an axis is logarithmic; switch one, keeping its visible range. |
| `equal_aspect` | Whether both axes share one scale (assignable). |
| `state()`, `restore(state)` | Save and restore a `ViewState` (centre and scales). |
| `transformed_bounds`, `screen_to_transformed`, `transformed_to_screen` | The same in *transformed* coordinates, in which the screen is linear: the world value on a linear axis, `log10` of it on a log axis. |

The [reference](reference.md#view) lists every member.

### Zoom limits

Zooming is limited to what 64-bit floats can draw accurately: a pixel can be
no smaller than about a trillionth of the coordinates on screen, so the
grid never jitters. Customise the limits with `ViewLimits`:

| Field | Default | Meaning |
|---|---|---|
| `min_scale` | `1e-12` | Smallest scale (most zoomed out), in pixels per unit. |
| `max_scale` | `1e14` | Largest scale (most zoomed in), even around the origin. |
| `min_relative_pixel` | `1e-12` | Smallest pixel size relative to the size of the centre's coordinates. |
| `max_abs_center` | `1e15` | Largest magnitude of either centre coordinate. |

```python
plane = pgp.Plane(limits=pgp.ViewLimits(min_scale=1.0, max_scale=1e4))
```

On a log axis the limits apply to the decades: at least one pixel per
decade, and a centre between `10⁻³⁰⁰` and `10³⁰⁰`.

## Other members

| Member | Meaning |
|---|---|
| `plane.theme`, `plane.interaction`, `plane.sampling` | Read or assign at any time. |
| `plane.log_x`, `plane.log_y` | Whether an axis is logarithmic; assign to switch it (see [Logarithmic axes](log-axes.md#switching-at-run-time)). |
| `plane.interactive` | Whether the built-in controls react to events. |
| `plane.begin_frame()` | Restart the colour palette (`draw_grid` does this). |
| `plane.update()` | Advance eased movement (`draw_grid` does this). |
| `plane.last_grid` | The last grid drawn, with the spacings it chose (handy for debug overlays). |
| `plane.renderer` | The renderer; `plane.renderer.clear_caches()` frees its font and label caches. |
