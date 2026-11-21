# Objects

`pygridplot.objects` holds the persistent objects: `Function`,
`ParametricCurve`, `PolarCurve`, `ImplicitCurve`, `Point`, `Points`,
`Polyline` and `Polygon`. Each is made once, keeps the arguments of the
plane method it mirrors as attributes, and draws itself **through the
public plane API** every frame. The plane keeps a list of them
(`plane.add`) and draws it (`plane.draw`), knowing them only through a
small protocol, so `plane.py` never imports `objects/`:

```text
objects/  ──>  plane.py (public API)  ──>  core/, render/
```

`tests/test_architecture.py` enforces this layering: nothing below the
objects imports them.

## Values

Objects take plain Python values: functions where the plane's methods take
functions, numbers where they take numbers, and, for coordinates, either a
number or a *function of nothing* (`lambda: p.x`), called whenever the
object is drawn or read. `objects/values.py` validates them. There is
nothing to compile and no state to keep in step: every attribute is read
when the object is drawn, so changing one (every frame, if need be) costs
nothing, and dependencies between objects are ordinary Python closures.

An object implements two hooks (`objects/base.py`): `_render(plane,
surface)`, one call of a plane method with the object's attributes, and
`_describe()`, a short description for `repr` and warnings. The base
classes do the rest: visibility, errors and warnings (`PlotObject`), the
style and its per-field properties, and the palette slot (`StyledObject`,
`LineObject`, `MarkerObject`).

## Drawing in two phases

`plane.draw(surface)` draws the grid, then the objects in two phases
(`Plane._draw_objects`): every object's `update(plane)`, then every
object's `render(plane, surface)`, then all their draggable handles on top
(`handles(plane)`). The split matters for points: `update` brings a
draggable point up to date with this frame's drag, so a function drawn
*before* the point, but reading `a.y`, already sees the new position.
`obj.draw(plane, surface)` does the same three steps for one object.

The protocol (`Drawable` in `plane.py`: `update`, `render`, `handles`,
`detach`) is all the plane knows, so a program can add classes of its own.

`render` never breaks a frame: an exception from a plane call (a
`ValueError` for an invalid setting) is caught and kept as the object's
`error`. `_report` issues a `PlotWarning` when that error appears or
changes, through `errors.warn`, which walks the stack to the first frame
outside the package so that the warning points at the program's
`plane.draw` line. Errors inside the program's own functions are not
object errors: the plane's samplers already turn them into gaps.

**Colours.** An object's colour is its own, or its **palette slot**'s from
the plane's theme. `plane.add` gives slots in order and never reuses them,
so colours stay with objects when others are hidden or removed; an object
drawn with `obj.draw` and no slot takes the frame's next colour, like an
immediate call.

Since objects draw with stable functions, the plane's
[plot cache](plotting.md#reusing-plots-between-frames-memopy) reuses
unchanged plots between frames as for any program; it checks the
functions' values, so a moved point is redrawn and everything else is not.

## Points and their handles

A `Point` is fixed (numbers), derived (a function of nothing returning
`(x, y)`, or one per coordinate) or draggable. Its coordinates are read
when needed (`_coordinate`), so `p.x` is always current.

A draggable point is the model: its numbers are the state that dragging
changes. On each plane it is drawn on it keeps a **handle**, a
`DraggablePoint` made with `plane.draggable(...)`, in `Point._handles` (a
weak dictionary by plane, so planes are freed when the program drops
them). `points._sync` runs in `update`:

1. A handle whose point's **constraint** (`_settings()`: lock, curve,
   bounds) changed is made again; a point turned non-draggable or removed
   from the plane loses its handle.
2. A handle on a curve gets its curve again every frame (a function's graph
   spans the visible x range, which changes; a curve's function may have
   changed). The handle keeps its parameter.
3. If the point's `(x, y, t)` differs from what was last synchronised, the
   point was **changed by code** (or on another plane): the handle moves
   to it, unless it is being dragged, since a drag wins.
4. The point then takes the handle's state (the drag, or the projection of
   what code set).

With several planes, a drag on one shows up on the others in step 3, as a
change "by code".

**Curves.** `Function`, `ParametricCurve` and `PolarCurve` are
`CurveObject`s: `_curve_spec(view, t)` gives the curve a point follows. A
function's graph is the curve `(t, f(t))` over the visible x range,
widened by the range's width on both sides (a decade each way on a log
axis) and always including `t ± 1`, so the point can be dragged anywhere
on screen. A point follows a curve object, or a curve `(x(t), y(t),
t_range)` of its own.

## Tests

`tests/objects/` checks that every object draws pixel for pixel like the
plane call it mirrors, on a plane and on its own; attributes changing
every frame; functions called, composed and differentiated; palette slots;
errors and where their warnings point; a program's own drawable; and
fixed, derived and draggable points (dragging seen in the same frame,
curve objects, toggling, several planes, no planes kept alive).

## Retired: graphs and the expression language

Two features were built on top of the objects and then deliberately
removed, because they do not fit how the library is meant to be used, and
supporting them would have constrained every new feature:

- **Graphs**: a retained, Desmos-like "expression list" (`pgp.Graph`)
  holding named objects written as lines of maths
  (`graph.add("y = a sin(x - P.x)")`), with names usable in any order,
  errors per item, and saving to and loading from JSON.
- **The expression language**: a small parser and compiler from text
  (`"x^2 sin(x)"`) to NumPy functions, with exact derivatives (`f'(x)`),
  behind `pgp.expression` and text arguments to objects.

PyGridPlot is a Python library: pictures are written in Python, and
lambdas, closures and objects already say everything text and names did,
without a second language to document, test and extend. Both features are
**not planned to come back** in this library. The last version that has
them is commit `c609cfb` ("persistent plot objects"); the first version of
the graph layer, before objects, is `45cd36c`. If they are ever wanted,
they belong in a separate package built on the public objects API (the
`Drawable` protocol lets any class draw itself on a plane), not in
PyGridPlot itself.
