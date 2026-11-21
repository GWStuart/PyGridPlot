# Graphs and the expression language

Two packages turn PyGridPlot into something a user can type maths into:

- **`core/expr`**, the expression language: text such as
  `"a sin(x - P.x)"` compiled into a NumPy function. It is pure (standard
  library and NumPy, plus `core.plot` for derivatives), like the rest of
  `core`.
- **`graph/`**, the retained layer: a `Graph` holds an ordered list of
  items, compiles their expressions with each other's names, and draws
  them each frame **through the public plane API**. It sits above
  `plane.py`, and nothing below it imports it.

```text
"y = a sin(x - P.x)"
   │  lines.interpret              (Graph.add: which kind of item?)
   ▼
PlotItem(Formula("a sin(x - P.x)"))   parsed at once: syntax errors show here
   │  namespace.compile_items      (on the first draw after a change)
   ▼
CompiledExpression  ── bindings: a → its value (live), P → its coordinates
   │  Graph.draw
   ▼
plane.plot(surface, compiled, style=..., color=palette[slot])
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

## The graph

### Items and formulas

Each item (`graph/items/`) stores its expressions as `Formula`s: what the
user gave (text, a Python function or a number) plus the parsed tree of a
text. Parsing happens when the value is set, so syntax errors are raised
at once; names are resolved only when the graph compiles, because items may
refer to items added later.

An item tells the graph what it needs through a few hooks:

- `_formulas()`: its formulas and the variables of each (`x` for a plot,
  `t` for a parametric curve, none for a point's coordinates);
- `_runtime(make)`: compiles them with the graph's `make` into what drawing
  needs (a function, a pair of functions, a getter of points);
- `_binding(runtime)`: what its name stands for in other expressions, if
  anything (a plot is a function, a point a `PointBinding`);
- `references()`: the names it uses (a draggable point adds its curve's).

Setters that change names or expressions call `_changed()`, which drops the
graph's compiled namespace; the next draw compiles again. Everything read
at draw time (styles, visibility, ranges, levels, a parameter's number)
just changes the attribute.

### Compiling a graph

`namespace.compile_items` builds a `Namespace` (the compiled runtime of
each item, the bindings by name, the errors, the curves of draggable
points) in four steps:

1. Each item's **dependencies** are the names it references that name items
   of the graph (built-ins and registered functions are not items).
2. **Kahn's algorithm** orders the items after their dependencies, keeping
   the graph's order where there is a choice. Items left over are in a
   cycle or depend on one; `_cycle_message` follows the dependencies to say
   which (`circular definition: f → g → f`, or `depends on 'f', which has an
   error`).
3. Items compile in that order, each with the bindings of the items before
   it. An item whose dependency failed fails with "depends on ...", so one
   mistake gives one root error and clear messages downstream.
4. A name used in an expression that names an item without a binding (a
   parametric curve, a polygon) is reported as such, rather than as an
   unknown name.

Compiling a typical graph takes well under a millisecond, so it is done
whenever anything changed, rather than tracking finer dependencies.

### Drawing

`Graph.draw` makes one plane call per visible item (`_draw_item`), with the
item's style and its colour: its own, or the theme palette's colour for its
**palette slot**. Slots are given when items are added and never reused
until `clear()`, so colours stay with items when others are hidden or
removed, and follow the plane's theme.

Errors never break a frame: compile errors skip the item; an exception
from a plane call (a `ValueError` for an invalid setting) is caught and
recorded for that frame. `_warn` issues a `GraphWarning` for an error the
first time it appears (or changes), and only from `draw`: reading
`graph.errors` or `item.error` is silent.

Since the graph draws with stable compiled functions, the plane's
[plot cache](plotting.md#reusing-plots-between-frames-memopy) reuses unchanged plots between frames as for
any program; it checks the functions' values, so a changed parameter is
redrawn and an unchanged one is not.

### Draggable points

A `DraggableItem` owns the point's state (position, `t`), which is what is
saved; the plane owns a `DraggablePoint` per item, created by the graph the
first time it draws on that plane. `Graph._bound` maps each plane (weakly)
to `_Bound(point, settings, synced)` per item, and `_sync` runs before
drawing:

1. A point whose item's **constraint** (`_settings()`: lock, curve, bounds)
   changed is made again; an item removed or in error loses its point.
2. A point on a curve gets its curve again every frame (a function's graph
   spans the visible x range, which changes; a curve's expression may
   have changed). The point keeps its parameter.
3. If the item's `(x, y, t)` differs from `synced`, the item was **changed
   by code** (or by another plane) since the last frame: the point is
   moved to it, unless it is being dragged, since a drag wins.
4. The item then takes the point's state (the drag, or the projection of
   what code set), and `synced` records it.

With several planes, a drag on one shows up on the others in step 3, as a
change "by code". `remove` and `detach` take points off planes at once.

A point following a curve by name gets a `CurveSource`: a plot `f`
becomes the curve `(t, f(t))` over the visible x range, widened by the
range's width on both sides (a decade each way on a log axis) and always
including `t ± 1`, so the point can be dragged anywhere on screen.

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
- `tests/graph/`: reading lines, names in any order, cycles and errors,
  palette slots, drawing **pixel for pixel identical** to the equivalent
  plane calls, draggable points (dragging, setting, curves by name,
  several planes, hiding, removal), saving (exact round trips in JSON and
  in pixels, every validation error, a property test that random graphs
  read back the same), exact views, and `show`.
