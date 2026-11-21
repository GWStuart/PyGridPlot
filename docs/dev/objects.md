# Objects, graphs and the expression language

Three packages sit on top of the immediate-mode plane:

- **`objects/`**, the persistent objects: `Function`, `ParametricCurve`,
  `PolarCurve`, `ImplicitCurve`, `Point`, `Points`, `Polyline`, `Polygon`.
  Each is made once, keeps its arguments as attributes, and draws itself
  **through the public plane API** every frame. The plane keeps a list of
  them (`plane.add`) and draws it (`plane.draw`), knowing them only through
  a small protocol, so `plane.py` never imports `objects/`.
- **`core/expr`**, the expression language: text such as
  `"a sin(x - P.x)"` compiled into a NumPy function. It is pure (standard
  library and NumPy, plus `core.plot` for derivatives), like the rest of
  `core`. Objects accept text anywhere they accept a function.
- **`graph/`**, named collections of objects: a `Graph` reads lines of
  maths, compiles its objects' text with each other's names, and saves to
  JSON. It is built on `objects/`, which never imports it.

```text
"y = a sin(x - P.x)"
   │  lines.interpret              (Graph.add: which kind of object?)
   ▼
Function(Formula("a sin(x - P.x)"))   parsed at once: syntax errors show here
   │  namespace.compile_items      (on the first draw after a change)
   ▼
CompiledExpression  ── bindings: a → its value (live), P → its coordinates
   │  Function.update / render     (called by Graph.draw or Plane.draw)
   ▼
plane.plot(surface, compiled, style=..., color=palette[slot])
```

The layering, enforced by `tests/test_architecture.py`:

```text
graph/  ──>  objects/  ──>  plane.py (public API)  ──>  core/, render/
   └─────────────────────────────>  core/expr
```

## The expression language

### Lexing and parsing

`lexer.tokenize` splits text into numbers, names (with the primes after
them counted, so `f''` is one token) and operators, reading Unicode
operators and superscript powers as their ASCII forms. An `e` ends a
number only when digits follow it, so `2e` is `2·e` and `1e-3` a number.

`parser.py` is a recursive-descent parser for the grammar in its module
docstring. Points worth knowing:

- **Implicit multiplication** is part of `term`: after a factor, a name or
  `(` starts another factor at the same precedence as `*`. A number does
  not (`x 2` is an error), nor does `-` (`2 -x` is a subtraction).
- **`name(...)` is always parsed as a call.** Whether it *is* one is only
  known when compiling: if the name turns out to be a number (`a(x + 1)`),
  the compiler multiplies instead.
- **Every node records its span** (`start`, `end`) in the text, and the span
  of a parenthesised expression includes its parentheses. `lines.py` slices
  the text by spans to split an equation into the expressions of an item,
  so `y = (x + 1)(x - 1)` keeps its parentheses. Errors use the spans for
  their caret.
- **Depth is limited** (`MAX_DEPTH`, 120) for parentheses, signs and
  powers, so neither parsing nor evaluating can exhaust Python's recursion
  limit, whatever the input.

### Compiling

`compiler.compile_expression(tree, variables, bindings)` turns a tree into
a `CompiledExpression`: a tree of small closures, one per operation, each
calling a NumPy ufunc on the results of its children. So a compiled
expression runs on arrays at NumPy's speed, and because it calls nothing
but ufuncs (and `np.where`), **Taylor jets pass through it untouched**:
`pgp.derivative` differentiates compiled expressions exactly, and `f'(x)`
in the text compiles to `derivative(f)`.

Each compiling step returns `(run, value)`, where `value` is the constant
value of the piece or a marker meaning "varies". Constant pieces are
evaluated once, when compiling (`2 pi / 3` costs nothing per call), and
chains of `+ -` and of `* /` become a loop rather than a deep tree.

Names are looked up in order: the variables, the built-in constants and
functions (`builtins.py`), then the caller's **bindings**:

| Binding | Stands for | Read |
|---|---|---|
| `ConstantBinding(value)` | a fixed number | when compiling (folded) |
| `ValueBinding(get)` | a number that may change | at every call |
| `FunctionBinding(func, arity)` | a function; `arity=None` if it varies | at every call |
| `PointBinding(get)` | a point, for `P.x` and `P.y` | at every call |

A graph binds parameters and points with getters, so changing a
parameter's number needs no recompiling. `FunctionBinding.of(func)` finds
the arity of a Python function (`ufunc.nin`, or its signature), so a call
with the wrong number of arguments is a compile error.

**Undefined values** are NaN: the whole evaluation runs under
`np.errstate(all="ignore")`. Comparisons and `and`/`or`/`not` give 1, 0 or
NaN (`builtins.truth`), and `if` gives NaN where its condition is NaN, so
a condition on an undefined value never counts as false. (A jet of an
undefined constant has undefined derivatives too: `Jet.constant` sets
them to NaN.)

**Errors** are `ExpressionError`s with a message, the text and a position.
Unknown names get a suggestion when one is likely: `_split` tries reading
a run-together name as a product or a call (`xy` → `x*y`, `sinx` →
`sin(x)`), and `difflib` finds near-misses of known names (`sqr` →
`sqrt`), with a high cutoff so that suggestions are rarely wrong.

### Why not `eval`, or SymPy?

`eval` runs arbitrary code from a file or a text box; a graph loaded from
someone else's JSON must never do that. SymPy would be a large dependency
for a small job, and slower to evaluate than closures over ufuncs. The
language is small enough that a hand-written parser stays short, gives
precise error messages, and makes the jets' exact derivatives work by
construction.

## The objects

### Formulas

Each object stores its functions and coordinates as `Formula`s
(`objects/values.py`): what the program gave (a Python function, a number,
or text) plus the parsed tree of a text. Parsing happens when the value is
set, so syntax errors are raised at once; names in text are resolved only
when the object is compiled, because in a graph an object may refer to
objects added later.

An object describes itself through a few hooks (`objects/base.py`):

- `_formulas()`: its formulas and the variables of each (`x` for a
  function, `t` for a parametric curve, none for a point's coordinates);
- `_runtime(make)`: compiles them with a `make` function into what drawing
  needs (a function, a pair of functions, getters of points);
- `_binding(runtime)`: what its name stands for in a graph's expressions,
  if anything (a function is a `FunctionBinding`, a point a
  `PointBinding`);
- `_render(plane, surface, runtime)`: draw, with one plane call;
- `references()`: the names its text uses (a draggable point adds its
  curve's).

**Who compiles.** `PlotObject._live()` returns the compiled runtime and
the error. An object on its own compiles itself with
`values.standalone` (built-in names only) and caches the result until
`_changed()`; an object in a graph asks its *owner* (`_owner`, typed by
the `Owner` protocol, which `Graph` implements), which compiles the whole
graph with names. Setters that change text or names call `_changed()`;
everything read at draw time (styles, visibility, ranges, levels, a
number, a Python function, an array of data) just changes the attribute,
so changing them every frame costs nothing.

### Drawing in two phases

`plane.draw(surface)` draws the grid, then the objects in two phases
(`Plane._draw_objects`): every object's `update(plane)`, then every
object's `render(plane, surface)`, then all their draggable handles on top
(`handles(plane)`). The split matters for points: `update` brings a
draggable point up to date with this frame's drag, so a function drawn
*before* the point, but reading `a.y`, already sees the new position. A
graph is itself a `Drawable` (`update` compiles and updates its objects,
`render` renders them), so `plane.add(graph)` works, and `graph.draw` is
the same three steps for one graph. `obj.draw(plane, surface)` does them
for one object.

`render` never breaks a frame: an object with an error is skipped; an
exception from a plane call (a `ValueError` for an invalid setting) is
caught and kept as the object's draw error. `_report` issues a
`PlotWarning` when an object's error appears or changes, through
`errors.warn`, which walks the stack to the first frame outside the
package so that the warning points at the program's `plane.draw` line.

**Colours.** An object's colour is its own, or its **palette slot**'s from
the plane's theme. `plane.add` and `graph.append` give slots in order and
never reuse them, so colours stay with objects when others are hidden or
removed; an object drawn with `obj.draw` and no slot takes the frame's
next colour, like an immediate call.

Since objects draw with stable functions, the plane's
[plot cache](plotting.md#reusing-plots-between-frames-memopy) reuses
unchanged plots between frames as for any program; it checks the
functions' values, so a moved point or a changed parameter is redrawn and
everything else is not.

### Points and their handles

A `Point` is fixed (numbers), derived (a function of nothing, or one per
coordinate, or text) or draggable. Its coordinates are read through
`_coordinate`, which only goes through the compiled runtime for text;
numbers and functions are read directly, so `p.x` is cheap and setting a
number never recompiles.

A draggable point is the model: its numbers are the state that dragging
changes and that a graph saves. On each plane it is drawn on it keeps a
**handle**, a `DraggablePoint` made with `plane.draggable(...)`, in
`Point._handles` (a weak dictionary by plane, so planes are freed when the
program drops them). `points._sync` runs in `update`:

1. A handle whose point's **constraint** (`_settings()`: lock, curve,
   bounds) changed is made again; a point turned non-draggable, removed or
   in error loses its handles.
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
on screen. A point follows a curve object directly, a graph's curve by
name (resolved by the graph's namespace, `Graph._curve_named`), or a curve
`(x(t), y(t), t_range)` of its own.

## The graph

A `Graph` owns its objects (`_owner`) and adds what a graphing calculator
has: names, text compiled with them, errors per object, and saving.
`Parameter` and `Definition` (`graph/definitions.py`) are the items only a
graph has: they draw nothing and define names.

### Compiling a graph

`namespace.compile_items` builds a `Namespace` (the compiled runtime of
each object, the bindings by name, the errors, the curves of draggable
points that follow one by name) in four steps:

1. Each object's **dependencies** are the names it references that name
   objects of the graph (built-ins and registered functions are not
   objects).
2. **Kahn's algorithm** orders the objects after their dependencies,
   keeping the graph's order where there is a choice. Objects left over are
   in a cycle or depend on one; `_cycle_message` follows the dependencies to
   say which (`circular definition: f → g → f`, or `depends on 'f', which
   has an error`).
3. Objects compile in that order, each with the bindings of the objects
   before it. An object whose dependency failed fails with "depends on
   ...", so one mistake gives one root error and clear messages downstream.
4. A name used in an expression that names an object without a binding (a
   parametric curve, a polygon) is reported as such, rather than as an
   unknown name.

Compiling a typical graph takes well under a millisecond, so it is done
whenever anything changed, rather than tracking finer dependencies.

### Reading lines

`lines.interpret` decides what `Graph.add` makes from a line, in this
order: `y = ...` (or `... = y`) without `y` on the other side is a plot;
`r = ...` a polar curve; `f(x) = ...` a named plot and `g(u, v) = ...` a
definition; `Name = (a, b)` a named point or parametric curve (it uses
`t`); `Name = ...` without variables a parameter; a lone pair a point or a
curve; a lone expression a plot; any other equation an implicit curve
(`lhs - (rhs)`, or `lhs` at the level `rhs` when that is a number). It
returns the name of the `Graph` method and its arguments, so `add` and the
methods share all validation.

### Saving

`serialize.py` writes plain data: a format marker, a version, the names of
the registered functions the graph needs, the view, and the items, each
with only the arguments that differ from their defaults (a style lists
only its changed fields). `dumps` writes it as JSON that is easy to edit by
hand: objects spread over lines, short lists of values on one line, whole
numbers without `.0`, Unicode as itself. NaN coordinates (gaps) and the
infinite ends of bounds are written as `null`.

A `Point` is written as kind `"point"` or, when draggable, `"draggable"`
(its numeric state, constraint and curve), so version 1 files from before
the objects existed read the same. Python functions, wherever they are
(a function, a curve's component, a point's coordinate), are written as
`{"function": name}` when registered and refused otherwise; a point
following a curve *object* is written with that object's name, which it
must have, in the same graph.

Reading is strict. `_Reader.take(key, kind)` checks each value's type, and
`done()` rejects unknown keys, both with the path of the value in the
message (`items[2].style.width`); item constructors then validate as
usual, and their errors get the item's path too. A `GraphError` is the
only exception loading can raise.

**Versions.** `VERSION` is the format written and the newest read. A new
optional key with a default needs no new version (older files lack it and
read as before). A change that would make an older reader misread a file
(a key whose meaning changes, a new kind of item) increments the version,
and reading keeps supporting every older version.

**Views.** A `GraphView` stores ranges, so a graph shows the same region in
a window of another size. They come from `View.fitted_ranges()`, the ranges
that `View.fit` maps back to the same centre and scale, rather than from
`bounds` (which `fit` would widen by up to a pixel each time), so saving
and loading a view is exact.

## Tests

- `tests/core/expr/`: the lexer and parser (precedence, spans, every error,
  a property test that no text makes the parser raise anything but an
  `ExpressionError`), the compiler (values against NumPy, every built-in,
  bindings, primes, NaN logic, suggestions), and **property tests** that
  compile random expression trees and compare them, and their exact
  derivatives, with the same computation written in NumPy.
- `tests/objects/`: every object drawing pixel for pixel like the plane
  call it mirrors, alone and on a plane; attributes changing every frame;
  functions called, composed and differentiated; palette slots; errors and
  where their warnings point; fixed, derived and draggable points
  (dragging seen in the same frame, curve objects, toggling, several
  planes, no planes kept alive).
- `tests/graph/`: reading lines, names in any order, cycles and errors,
  palette slots, drawing **pixel for pixel identical** to the equivalent
  plane calls, draggable points (dragging, setting, curves by name,
  several planes, hiding, removal), saving (exact round trips in JSON and
  in pixels, every validation error, a property test that random graphs
  read back the same), exact views, and `show`.
