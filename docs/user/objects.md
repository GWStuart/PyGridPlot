# Objects

Objects are functions, curves and points that you **make once** and
**change through their attributes**; the plane draws them every frame.
They are the persistent counterpart of the plane's drawing methods:
`pgp.Function(f)` draws like `plane.plot(screen, f)`, takes the same
arguments, and keeps them as attributes you can change at any time.

```python
a = pgp.Point((1, 2), draggable=True)
f = pgp.Function(lambda x: a.y * np.sin(x - a.x), color="crimson")
circle = pgp.ParametricCurve(np.cos, np.sin)
plane.add(a, f, circle)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)       # drags a; f follows it

    f.width = 4 if a.dragging else 2.5  # attributes are read every frame
    plane.draw(screen)                  # grid, objects, draggable points
    pygame.display.flip()
```

There are three ways to draw, and they mix freely on one plane:

| Style | Looks like | Best for |
|---|---|---|
| [Immediate](plotting.md) | `plane.plot(screen, f)` every frame | drawing that your code decides anew each frame |
| Objects (this page) | `f = pgp.Function(...)`, `plane.add(f)`, `f.color = ...` | pictures made of parts that change now and then |
| [Graphs](graphs.md) | `graph.add("y = a sin(x)")` | maths typed as text, names, saving to a file |

Objects are drawn with the immediate methods underneath, so they look the
same, are as fast, and get everything the plane does (log axes, themes,
[reuse of unchanged plots](plotting.md#reusing-unchanged-plots)).

## The objects

| Object | Draws like | Its attributes |
|---|---|---|
| `pgp.Function(func, *, domain, ...)` | `plane.plot` | `func`, `domain`, `sampling` |
| `pgp.ParametricCurve(x, y, t_range=(0, 2π), ...)` | `plane.parametric` | `x`, `y`, `t_range`, `sampling` |
| `pgp.PolarCurve(r, theta_range=(0, 2π), ...)` | `plane.polar` | `r`, `theta_range`, `sampling` |
| `pgp.ImplicitCurve(func, level=0, ...)` | `plane.implicit` | `func`, `level`, `sampling` |
| `pgp.Point(position, *, draggable=False, ...)` | `plane.points`, or a draggable point | `position`, `x`, `y`, `draggable`, ... |
| `pgp.Points(points, ...)` | `plane.points` | `points` |
| `pgp.Polyline(points, *, closed=False, ...)` | `plane.polyline` | `points`, `closed` |
| `pgp.Polygon(points, ...)` | `plane.polygon` | `points`, `fill`, `fill_color`, ... |

Every object also has `visible`, `name` (optional, a label), `error`
(see [Errors](#errors)) and its style: `color`, `opacity` and, depending on
the kind, `width`, `radius`, `marker`, `edge_color`, `edge_width`, or a
whole `style` (a `LineStyle`, `PointStyle` or `PolygonStyle`, as for the
plane).

## Changing objects

Assign to an attribute whenever you like, in the main loop or in an event
handler; the next frame shows the change:

```python
wave = plane.add(pgp.Function(np.sin))
wave.func = np.cos
wave.color = "purple"
wave.width = 4
wave.domain = (0, 2 * math.pi)
wave.visible = False            # hidden; wave.visible = True shows it again
```

Data can be replaced every frame, as an `(N, 2)` array (drawn as it is,
without copying point by point):

```python
trace = plane.add(pgp.Polyline(np.zeros((0, 2))))
xs = np.linspace(-5, 5, 400)
for frame in range(3):
    trace.points = np.column_stack((xs, np.sin(xs + frame / 10)))
    plane.draw(screen)
```

## Objects that use each other

Objects are plain Python values, so one object can use another in a
function: read `a.x` and `a.y` of a point, or call a function object. The
functions are called every time they are drawn, so the picture follows
along:

```python
a = pgp.Point((1, 1), draggable=True)
f = pgp.Function(lambda x: x**3 / 3 - x)
tangent = pgp.Function(lambda x: f.derivative()(a.x) * (x - a.x) + f(a.x))
on_f = pgp.Point(lambda: (a.x, f(a.x)))          # a point that follows
plane.add(f, tangent, on_f, a)
```

- **Functions are functions.** `f(2.0)` evaluates a `Function` (numbers or
  arrays), `f.derivative(n)` is its exact derivative (as
  [`pgp.derivative`](plotting.md#derivatives), which also takes `f`), and a
  `Function` can be used inside another: `pgp.Function(lambda x: f(x)**2)`.
  `c(t)` of a `ParametricCurve` is the point `(x, y)`, `r(θ)` of a
  `PolarCurve` the radius, and `F(x, y)` of an `ImplicitCurve` the value of
  `F`.
- **The order does not matter for dragging.** The plane updates every
  object before drawing any, so a function drawn before the point it reads
  still uses where the point was dragged this frame.

## Points

A `pgp.Point` is one marker, in one of three ways:

```python
fixed = pgp.Point((1, 2))                         # numbers: set p.x, p.position
follows = pgp.Point(lambda: (fixed.x, fixed.y + 1))  # a function giving (x, y)
half = pgp.Point((lambda: fixed.x / 2, 0))        # or one coordinate at a time
dragged = pgp.Point((0, 0), draggable=True)       # moved by the user
```

Read the position as `p.x`, `p.y` or `p.position` (numbers, for every
kind of point); set it with `p.x = 3` or `p.position = (3, 4)`. A point
whose function fails is undefined (NaN), and not drawn.

**Draggable points** take the options of
[`plane.draggable`](draggable-points.md): `lock="x"` or `"y"`, bounds
(`x_range`, `y_range`), `coordinates` ("hover", "always", "never") and
`enabled`, and they can follow a curve:

```python
f = pgp.Function("x^2 / 4")
on_graph = pgp.Point(curve=f, t=1, draggable=True)       # on y = f(x)
circle = pgp.ParametricCurve(np.cos, np.sin)
on_circle = pgp.Point(curve=circle, draggable=True)      # on the circle
spiral = pgp.Point(curve=("0.2t cos(t)", "0.2t sin(t)", (0, 20)), draggable=True)
slider = pgp.Point((-4, 1), lock="x", y_range=(0, 3), draggable=True)
plane.add(f, circle, on_graph, on_circle, spiral, slider)
```

`curve` is a `Function`, `ParametricCurve` or `PolarCurve` object (the
point follows it even when its function changes; on a function's graph,
`t` is `x`), or `(x(t), y(t), t_range)`. `p.t` is the parameter on the
curve, and setting it moves the point along. `p.dragging` says whether it
is being dragged, and `p.draggable` can be switched off and on while the
program runs (switching it on needs a position of numbers). A point
created draggable takes the plane's style for draggable points (larger,
with an outline); other points take the marker style.

A draggable point added to several planes keeps one handle on each, all
in step. The handles are the plane's
[`DraggablePoint`s](draggable-points.md): `plane.draggable(...)` is the
lower-level way to make one, still there for immediate-mode programs.

## Drawing

- `plane.add(obj, ...)` keeps objects on the plane (returning the first,
  so `f = plane.add(pgp.Function(np.sin))` works); `plane.remove(obj)`,
  `plane.clear_objects()` and `plane.objects` manage them. A graph can be
  added too: `plane.add(graph)`.
- `plane.draw(screen)` draws a whole frame: the grid, the objects in the
  order they were added, then every draggable point on top.
- To mix with immediate drawing, draw the parts yourself: `plane.draw_grid`,
  your own calls, then `plane.draw_objects(screen)` for the objects (with
  their draggable points on top):

```python
plane.add(pgp.Function(np.cos))
plane.draw_grid(screen)
plane.polygon(screen, [(-2, -1), (2, -1), (0, 1)], fill_opacity=0.2)  # below
plane.draw_objects(screen)
```

- `obj.draw(plane, screen)` draws one object now, for full control over
  the order; an object drawn this way needs no `plane.add`.

**Colours.** An object without a colour gets the next *palette slot* when
it is added to a plane (or a graph), so it keeps its colour when others are
hidden or removed, and follows the theme. Set `color`, or `palette` (the
slot), to choose.

## Text

Wherever an object takes a function, it also takes text in the
[expression language](expressions.md): `pgp.Function("x^2 sin(x)")`,
`pgp.ParametricCurve("cos(3t)", "sin(2t)")`, `pgp.Point(("1/2", "sqrt(3)/2"))`.
On their own, objects know only the built-in names (`sin`, `pi`, ...); in a
[graph](graphs.md), text can also use the names of the graph's other
objects and parameters.

## Objects in graphs

A graph is a named collection of the same objects, with lines of maths and
saving: `graph.add("y = x^2")` returns a `pgp.Function`, and an object
made on its own can be put in a graph:

```python
graph = pgp.Graph()
f = graph.append(pgp.Function("a x^2", name="f"))
graph.add("a = 0.5")
graph.append(pgp.Point(curve=f, t=1, draggable=True, name="P"))
graph.add("y = f'(P.x)(x - P.x) + P.y")
plane.add(graph)
```

An object belongs to at most one graph; add the graph to the plane rather
than the object. See [Graphs](graphs.md) for names, errors and saving.

## Errors

An object that cannot be drawn (text with an unknown name, a curve that
has an error itself, an invalid setting) is skipped, and everything else
still draws. Its `error` attribute says why, and drawing issues a
`pgp.PlotWarning` the first time each error appears, pointing at your
`plane.draw` line. Calling a function object that has an error raises a
`ValueError` with the same message.

```python
broken = pgp.Function("q x")
print(broken.error)    # unknown name 'q'
```

Errors *inside* your Python functions are not errors of the object: as
for the plane's methods, a function that raises or returns NaN just leaves
a gap.

## See also

- `examples/three_ways.py`: the same picture in immediate mode, with
  objects and with a graph, side by side, with comments comparing them.
- `examples/objects.py`: a picture made of objects (a function shaped by
  points and sliders, a tangent, a point on a circle).
- [Draggable points](draggable-points.md): how dragging works.
- [Graphs](graphs.md): objects with names, text and saving.
- [API reference](reference.md#objects): every object and attribute.
