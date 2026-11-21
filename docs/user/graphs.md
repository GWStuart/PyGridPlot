# Graphs

A **graph** is a list of named [objects](objects.md) (functions, curves,
points), plus parameters and definitions, kept between frames, drawn with
one call and saved to a file. It is what the expression list of a graphing
calculator holds: maths written as text, with names that the other lines
can use. Underneath it is the objects, drawn with the plane's ordinary
methods, so drawing a graph makes exactly the calls you would make by
hand.

```python
graph = pgp.Graph()
graph.add("a = 2")
P = graph.draggable((1, 1), name="P")
graph.add("y = a sin(x - P.x) + P.y")
graph.add("x^2 + y^2 = 9")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)    # drags P; the wave follows

    plane.draw_grid(screen)
    graph.draw(plane, screen)        # or plane.add(graph) once, then plane.draw(screen)
    pygame.display.flip()

graph.save("wave.json")              # and Graph.load("wave.json") later
```

Use a graph when what is drawn is *data*: typed in by a user, loaded from
a file, edited while the program runs, or shared between programs. For a
picture written in Python, [objects](objects.md) on their own are simpler,
and for drawing that your code decides every frame, so are the plane's
methods (see [Plotting](plotting.md)); all three mix freely on one plane.

## Adding items

### One line of maths

`graph.add(text)` reads a line the way a graphing calculator does and adds
the item it describes:

| Line | Item |
|---|---|
| `y = x^2`, or just `x^2` | a plot of `x^2` |
| `f(x) = x^2` | a plot named `f`, also usable as a function: `f(2)`, `f'(x)` |
| `r = 1 + cos(θ)` | a polar curve |
| `(cos(t), sin(t))` | a parametric curve (it uses `t`) |
| `C = (2cos(t), sin(t))` | a parametric curve named `C` |
| `(1, 2)`, or `P = (a, 2a)` | a point (named `P`); its coordinates may be expressions |
| `a = 2`, or `b = 2a + 1` | a parameter: a named value |
| `g(u, v) = u v` | a definition: a function for other items, not drawn |
| `x^2 + y^2 = 9`, `x = 3`, `y^2 = x^3 - x` | an implicit curve (any other equation in `x` and `y`) |

The expressions are written in the [expression language](expressions.md):
`2x`, `a sin(x)`, `x^2`, `sqrt`, `ln`, `if`, `pi`, `f'(x)` and so on.
Keyword arguments are passed on to the item, as for the method that makes
it (below):

```python
graph = pgp.Graph()
graph.add("y = sin(x)", color="red", width=4, domain=(0, 2 * math.pi))
graph.add("(cos(t), sin(t))", t_range=(0, math.pi))
graph.add("P = (1, 2)", draggable=True, lock="x")   # a draggable point
```

Inequalities (`y < x^2`, shaded regions) are not supported yet.

### One method per kind

Each kind of item also has a method, with the same arguments as the
plane's method that draws it, except that expressions may be text, and
that a parametric curve's `t_range` may be left out (it is `(0, 2π)`, as
for a line of maths). The items are the [objects](objects.md), so each
method returns one:

| Graph method | Draws like | Returns |
|---|---|---|
| `graph.plot(expression, *, name, domain, ...)` | `plane.plot` | `pgp.Function` |
| `graph.parametric(x, y, t_range, ...)` | `plane.parametric` | `pgp.ParametricCurve` |
| `graph.polar(r, theta_range, ...)` | `plane.polar` | `pgp.PolarCurve` |
| `graph.implicit(expression, level, ...)` | `plane.implicit` | `pgp.ImplicitCurve` |
| `graph.point((x, y), ...)` | `plane.points` | `pgp.Point` |
| `graph.points(points, ...)` | `plane.points` | `pgp.Points` |
| `graph.polyline(points, *, closed, ...)` | `plane.polyline` | `pgp.Polyline` |
| `graph.polygon(points, ...)` | `plane.polygon` | `pgp.Polygon` |
| `graph.draggable(position, *, lock, curve, ...)` | `plane.draggable` | `pgp.Point` (draggable) |
| `graph.parameter(name, value)` | (not drawn) | `Parameter` |
| `graph.define(signature, expression)` | (not drawn) | `Definition` |

An object made on its own joins a graph with `graph.append(obj)` (at the
end, or at `index=`); an object belongs to at most one graph.

```python
graph = pgp.Graph()
graph.parameter("k", 3)
graph.plot("sin(k x) / x", name="wave", width=3)
graph.parametric("cos(k t)", "sin(2t)", (0, 2 * math.pi))
graph.implicit("x^2 - y^2", [-1, 1], color="gray")      # contour lines
graph.polygon([(0, 0), (2, 0), ("k", "k")], fill_opacity=0.4)
graph.plot(np.sinc, name="s")                           # Python functions work too
```

Points of `points`, `polyline` and `polygon` items are numbers (a NumPy
array of data is fine, and fast) or expressions without variables, which
are evaluated every frame. A NaN coordinate leaves a gap, as on the plane.

Each method returns its item; `graph["name"]` finds a named item,
`graph.items` lists them all in drawing order, and `graph.remove(item)`,
`graph.move(item, index)` and `graph.clear()` edit the list.

## Names

Naming an item lets other items use it:

- a **parameter** `a` is a number: `"a x"`;
- a **plot** `f`, a **polar curve** `R` and a **definition** `g` are
  functions: `"f(x - 1)"`, `"f'(x)"`, `"R(θ) / 2"`, `"g(x, 2)"`; a named
  **implicit curve** `F` is the function `F(x, y)` of its expression;
- a **point** or **draggable point** `P` is a point: `"P.x"`, `"P.y"`.

Names can be used before they are defined, in any order: a graph works
out the order itself. A name is letters, digits and `_` (not first a
digit), and it cannot be a built-in name (`sin`, `pi`, `e`, ...) or a
variable (`x`, `y`, `t`, `θ`, `theta`). Each name is used once in a graph.

```python
graph = pgp.Graph()
graph.add("y = f(x) + b")        # uses f and b, defined below
graph.add("f(x) = a x^2")
graph.add("b = 2a")
graph.add("a = 0.5")
print(graph.evaluate("f(2) + b"))   # 3.0
```

Renaming an item (`item.name = "h"`) does not rewrite the expressions
that used the old name: they then report an unknown name.

## Changing items

Every argument an item was made with is an attribute that can be changed
at any time; the next `draw` shows the change:

```python
graph = pgp.Graph()
a = graph.add("a = 1")
wave = graph.add("y = a sin(x)")
wave.color = "purple"
wave.width = 4
wave.func = "a cos(x)"
a.value = 2.5                    # everything using a follows
wave.visible = False             # hidden, but its name still works
```

Changing a parameter's number costs nothing (it is read when drawn), so a
parameter can be animated every frame:

```python
graph = pgp.Graph()
phase = graph.parameter("phase", 0)
graph.add("y = sin(x - phase)")
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        plane.handle_event(event)
    phase.value = pygame.time.get_ticks() / 1000
    plane.draw_grid(screen)
    graph.draw(plane, screen)
    pygame.display.flip()
    clock.tick(60)
```

Changing an expression, a name or the list itself makes the graph compile
again when it is next drawn, which takes well under a millisecond for a
typical graph, so it can also happen every frame if needed.

**Colours.** An item without a colour takes one from the plane's theme
palette. Each item gets its own *palette slot* when it is added (the next
one), so hiding or removing an item does not change the colours of the
others, and the colours follow the theme (light or dark). Set `color` to
choose one yourself, or `palette` to choose the slot.

## Draggable points

A graph's draggable points are draggable [`pgp.Point`s](objects.md#points)
(see also [Draggable points](draggable-points.md)): `graph.draggable(...)`
is `graph.point(..., draggable=True)`, and `graph.add("P = (1, 2)",
draggable=True)` makes one from a line. Their position is part of the
graph, changed by dragging, by setting it, and saved with the graph.

```python
graph = pgp.Graph()
graph.add("f(x) = x^2 / 4")
P = graph.draggable((1, 1), name="P")                     # free
S = graph.draggable((-4, 1), name="S", lock="x", y_range=(0, 3))  # a slider
D = graph.draggable(curve="f", t=2, name="D")             # on the plot of f
graph.add("y = f'(D.x)(x - D.x) + D.y")                   # the tangent at D
graph.add("y = S.y sin(x - P.x) + P.y")                   # driven by P and S
```

- `curve` is the name of a plot, parametric or polar curve of the graph,
  which the point follows even when its expression changes (on a plot,
  `t` is `x`), the curve object itself (`curve=graph["f"]`), or
  `(x(t), y(t), t_range)` for a curve of its own:
  `curve=("cos(t)", "sin(t)", (0, math.pi))`.
- Read the position as `P.position`, `P.x`, `P.y` (and `D.t` on a curve);
  set it from code with `P.position = (2, 1)` or `D.t = 0.5`.
- A point on a curve that is given neither `position` nor `t` starts at
  `x = 0` on a plot and at the start of the range on other curves; until
  first drawn its position is `None`.
- A hidden point (`visible = False`) is neither drawn nor draggable; a
  disabled one (`enabled = False`) is drawn but cannot be dragged.

Each point keeps a handle (the plane's `DraggablePoint`) on every plane
it is drawn on, made the first time, and keeps the two in step: dragging
moves the point; changing the point moves the handle. A graph drawn on several planes keeps a point on each, all in
step. `graph.remove(P)` takes the point off every plane;
`graph.detach(plane)` takes all of a graph's points off one plane that
will no longer draw it.

## Errors

An item that cannot be drawn (an unknown name, a circular definition, a
syntax error in a changed expression) is skipped, and the rest of the
graph still draws. `item.error` says what is wrong, `graph.errors` lists
every item with an error, and drawing issues a `pgp.PlotWarning` the
first time each error appears:

```python
graph = pgp.Graph()
f = graph.add("f(x) = g(x) + 1")
g = graph.add("g(x) = f(x)")
uses = graph.add("y = f(x) + b")
print(f.error)       # circular definition: f → g → f
print(uses.error)    # depends on 'f', which has an error
graph.remove(g)
graph.add("g(x) = x^2")
print(uses.error)    # unknown name 'b'
```

Syntax errors are found as soon as an expression is set: `graph.add` and
assigning `item.func` raise a `pgp.ExpressionError`, so a line
typed by a user can be checked before it is added. Names are checked when
the graph is next drawn or asked for its errors, since items may be added
in any order.

## Evaluating

`graph.evaluate(text, **variables)` evaluates an expression with the
graph's names, and `graph.function(name)` returns a named function to call,
plot or differentiate:

```python
graph = pgp.Graph()
graph.add("f(x) = x^2")
graph.add("k = 3")
print(graph.evaluate("f(k) + f'(k)"))                # 15.0
print(graph.evaluate("f(x) + k", x=np.arange(3)))     # [3. 4. 7.]
f = graph.function("f")
print(pgp.derivative(f, 2)(1.0))                      # 2.0
```

## Python functions

Expressions can call Python functions that are registered with the graph,
and a Python function can be given wherever an expression's text can:

```python
def bump(x):
    return np.exp(-x**2)

graph = pgp.Graph(functions={"bump": bump})
graph.add("y = 2 bump(x - 1)")
graph.register("sinc", np.sinc)
graph.add("y = sinc(x)")
graph.plot(np.cos, name="c")       # draws, but cannot be saved: see below
```

The functions take NumPy arrays (or numbers) like any function given to a
plane. A saved graph records a registered function by name only, so the
program that loads it must pass the function again:
`Graph.load(path, functions={"bump": bump})`. A Python function that is
not registered (like `np.cos` above) cannot be saved: `save` raises a
`pgp.GraphError` that names the item, so register it first.

## Saving and loading

```python
graph = pgp.Graph()
graph.add("a = 2")
graph.add("y = a x^2", color="red")
graph.draggable((1, 2), name="P")
graph.save("parabola.json")

loaded = pgp.Graph.load("parabola.json")
text = loaded.to_json()               # the same as a string
data = loaded.to_dict()               # the same as plain data
again = pgp.Graph.from_json(text)
```

The file is plain JSON, meant to be readable and editable by hand. Each
item lists only what differs from the defaults:

```json
{
  "format": "pygridplot-graph",
  "version": 1,
  "items": [
    {"kind": "parameter", "name": "a", "value": 2},
    {"kind": "plot", "expression": "a x^2", "palette": 0, "style": {"color": "red"}},
    {"kind": "draggable", "name": "P", "position": [1, 2], "palette": 1}
  ]
}
```

Expressions are saved as their text, colours as given (a name, a hex
string or an RGB list), and draggable points where they are. Loading is
strict: an unknown key, a value of the wrong kind or an invalid setting is
a `pgp.GraphError` that says where it is
(`items[1].style.width: expected a number`), so mistakes in a hand-written
file are easy to find. A file written by a newer version of the format is
refused rather than misread.

### Views

A graph can also hold the region it is meant to be seen at, a
`pgp.GraphView` (ranges and the kinds of axes), saved with it:

```python
graph = pgp.Graph()
graph.add("y = 10^x")
plane.log_y = True
plane.set_view((0, 3), (1, 1000))
graph.capture_view(plane)             # remember what the plane shows
graph.save("growth.json")

other = pgp.Plane()
pgp.Graph.load("growth.json").apply_view(other)   # show it again
```

The view is kept as ranges, so it shows the same region in a window of
another size, and exactly the same view in a window of the same size.

## A quick look

`pgp.show` draws lines of maths and graphs, as well as functions, in a
window of its own:

```python
pgp.show("y = x^2", "x^2 + y^2 = 9", "r = 1 + cos(θ)")
pgp.show(pgp.Graph.load("parabola.json"))   # with its draggable points
```

A graph with a saved view opens at it, unless `show` is given view
options of its own.

## See also

- [The expression language](expressions.md): everything an expression
  can contain.
- [Draggable points](draggable-points.md): how dragging works.
- `examples/graph_editor.py`: a small expression list to type into, with
  saving and loading.
- [API reference](reference.md#graphs): every method and attribute.
