# Draggable points

Draggable points are points the user moves with the mouse: handles that
reshape a curve, sliders that set a value, a point that runs round a circle.
They are the one thing on a plane that keeps state between frames, so they
work a little differently from the plotting methods. You **create** each
point once, before the main loop; the plane **drags** it in `handle_event`;
your code **reads** its position every frame and uses it like any number;
and you **draw** the points, usually last, with `draw_draggables`.

```python
a = plane.draggable((1, 2))                             # free
s = plane.draggable((-3, 0.5), lock="x", y_range=(0, 1))   # a vertical slider
c = plane.draggable(curve=(np.cos, np.sin, (0, 2 * np.pi)))  # on the unit circle

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)        # drags a point under the cursor, else pans

    plane.draw_grid(screen)
    plane.plot(screen, lambda x: s.y * np.sin(x - a.x) + a.y)
    plane.polyline(screen, [(0, 0), c.position])
    plane.draw_draggables(screen)
    pygame.display.flip()
```

A press on a point drags the point; anywhere else it pans the plane as
usual, and zooming works in the middle of a drag (the point stays under the
cursor). The point keeps the offset at which you grabbed it, so it never
jumps, and it cannot be dragged out of the plane.

## Four kinds of point

| Kind | Created with | Moves |
|---|---|---|
| Free | `plane.draggable((x, y))` | Anywhere. |
| x locked | `plane.draggable((x, y), lock="x")` | Up and down only: x stays fixed. |
| y locked | `plane.draggable((x, y), lock="y")` | Left and right only: y stays fixed. |
| On a curve | `plane.draggable(curve=(x, y, t_range))` | Along the parametric curve `(x(t), y(t))`. |

A point on a curve takes the curve's components exactly as
[`parametric`](plotting.md#parametric-curves) does: functions of `t`
(NumPy or plain `math`) or numbers. It starts at `t` (default: the start of
`t_range`), or at the point of the curve nearest a `position` you give:

```python
on_circle = plane.draggable(curve=(np.cos, np.sin, (0, 2 * np.pi)), t=np.pi / 4)
on_parabola = plane.draggable(curve=(lambda t: t, lambda t: t**2, (-3, 3)),
                              position=(1, 1))
on_spiral = plane.draggable(curve=(lambda t: 0.2 * t * np.cos(t),
                                   lambda t: 0.2 * t * np.sin(t), (0, 6 * np.pi)))
```

While dragged, the point follows the curve from where it is to the point
nearest the cursor, so it passes through crossings (a figure-eight, a
spirograph) and between the turns of a spiral without jumping. It jumps to
another part of the curve only when that part is clearly closer to the
cursor (by about 10 pixels), so it can still be pulled straight across to
another branch. On a closed curve such as a circle, dragging past the end
continues from the start. The point's `t` tells you where on the curve it
is: on the circle above, it is the angle.

## Reading and setting the position

| Attribute | Meaning |
|---|---|
| `x`, `y`, `position` | The world position. |
| `t` | The curve parameter of a point on a curve (`None` for other points). |
| `dragging`, `hovered` | Whether the point is being dragged, or under the cursor. |
| `moved` | Whether the point moved since the previous frame (stable while you draw). |

Assigning `position` moves a point from code: a free point goes there, a
locked point too (locks restrict only dragging), and a point on a curve goes
to the nearest point of its curve. Assigning `t` moves a point along its
curve.

```python
a.position = (0.5, 1.5)
on_circle.t = np.pi           # to (-1, 0)
if a.moved:
    print(f"a is now at ({a.x:.3g}, {a.y:.3g})")
```

Use `moved` to recompute something expensive only when a point has
changed. A frame starts at `draw_grid`, so `moved` keeps its value while
you draw.

## Bounds

`x_range` and `y_range` keep free and locked points within limits, while
dragging and when set from code. A point on a curve is kept within its
`t_range` instead.

```python
amplitude = plane.draggable((-6, 1), lock="x", y_range=(0.1, 3))
plane.polyline(screen, [(-6, 0.1), (-6, 3)], width=4, opacity=0.3)  # a track
```

On a [log axis](log-axes.md) a point can never be dragged to zero or below.

## How points look

Draggable points are larger than plain [points](plotting.md#points), with
an outline in the theme's background colour that separates them from the
curves they sit on. Each takes its own palette colour, in the order the
points were created (so colours do not change with what else you draw).
Change any of it when creating the point, or later through its attributes:

```python
handle = plane.draggable((2, 1), color="crimson",
                         style=pgp.PointStyle(radius=8, marker="square"))
handle.color = (40, 120, 200)
```

While the cursor is over a point (or the point is dragged) it gets a
translucent ring, the cursor becomes a hand, and the point's coordinates
appear next to it, rounded to the size of a pixel: `(1.23, -0.5)` at the
default zoom, `(1.2346, -0.5)` zoomed in a hundred times. `coordinates=`
chooses when they show: `"hover"` (the default), `"always"` or `"never"`.
The label uses the theme's number format and label font.

`draw_draggables(screen)` draws every point of the plane; pass some points
to draw only those (`plane.draw_draggables(screen, a, c)`). Draw them after
the plots so they sit on top.

## Turning points off and removing them

`point.enabled = False` keeps a point visible but stops it being dragged.
`point.remove()` takes it off the plane for good. `plane.draggables` lists
the plane's points, and `plane.dragged` is the one being dragged, if any.

`plane.interactive = False` switches off only panning and zooming:
draggable points still work, which suits a fixed view with handles on it.

## Settings

Three fields of [`InteractionConfig`](interaction.md#configuring-the-controls)
control dragging:

| Field | Default | Meaning |
|---|---|---|
| `drag_button` | `1` | Mouse button that drags points. |
| `grab_margin_px` | `6.0` | A point can be grabbed this many pixels beyond its marker. |
| `hover_cursor` | `True` | Show the hand cursor over points. |

## Animated curves

A point on a curve can follow a curve that changes from frame to frame:
assign the new curve to `point.curve` each frame, and the point keeps its
parameter `t`:

```python
t_now = pygame.time.get_ticks() / 1000
radius = 2 + np.sin(t_now)
on_circle.curve = (lambda t: radius * np.cos(t), lambda t: radius * np.sin(t),
                   (0, 2 * np.pi))
```

## Limits

- Points are dragged with the mouse only; there is no keyboard nudging or
  snapping to a step yet.
- A point on a curve follows the curve as it is drawn, sampled finely on
  screen; on a part of the curve that is far off screen it follows a coarser
  version until that part comes into view.

See `examples/draggable.py` for a sine wave shaped by a free point and two
sliders, and points running round a circle and a spirograph.

`plane.draggable` makes the plane's own points. A draggable
[`pgp.Point`](objects.md#points) object is the same with the conveniences
of objects: it can be made before the plane, follow a function or curve
*object* (`pgp.Point(curve=f, draggable=True)`), be switched between fixed
and draggable, and live on several planes at once.
