# The expression language

Graphs (see [Graphs](graphs.md)) and `pgp.expression` take maths written
as text, the way you would type it into a graphing calculator:

```python
f = pgp.expression("x^2 sin(x) / 4")         # a function of x
plane.plot(screen, f)
plane.plot(screen, pgp.derivative(f))        # derivatives are exact

wave = pgp.expression("a sin(k x - 1)", a=2, k=3)
height = pgp.expression("x^2 + y^2")         # a function of x and y
plane.implicit(screen, height, level=4)
```

Text is never run as Python (there is no `eval`): it is parsed and
compiled into a small NumPy function, so it runs on whole arrays at NumPy's
speed, gives NaN where it is undefined instead of raising, and can be
differentiated exactly. This page describes the language.

## Numbers, variables and names

- **Numbers**: `2`, `0.5`, `.5`, `1e-3`, `2.5E4`.
- **Variables** are the arguments of the function being written: `x` for a
  plot, `x` and `y` for an implicit curve, `t` for a parametric curve, `θ`
  (or `theta`) for a polar curve.
- **Constants**: `pi` (or `π`), `tau` (or `τ`, 2π) and `e`.
- **Other names** stand for what you defined: a parameter `a`, a function
  `f`, a point `P` (whose coordinates are `P.x` and `P.y`). Names are
  letters, digits and `_`, starting with a letter or `_`: `a`, `k2`,
  `speed_max`, `α`.

## Operators

From the loosest to the tightest binding:

| Operators | Meaning | Notes |
|---|---|---|
| `<` `<=` `>` `>=` `==` `!=` | comparisons | 1 if true, 0 if false (see [Conditions](#conditions)); not chained |
| `+` `-` | addition, subtraction | left to right |
| `*` `/`, or nothing | multiplication, division | left to right: `a/b*c` is `(a/b)*c` |
| `-x`, `+x` | sign | |
| `^` (or `**`) | power | right to left: `2^3^2` is `2^9` |
| `f(x)`, `f'(x)`, `P.x`, `( )` | calls, coordinates, grouping | |

So `-x^2` is `-(x^2)` and `2^-x` is `2^(-x)`, as in maths.

**Implicit multiplication.** Writing two things side by side multiplies
them, as on paper: `2x`, `3(x + 1)`, `(x + 1)(x - 1)`, `a sin(x)`,
`2 pi r`, `x y`. It binds like `*`, so `x/2y` is `(x/2)·y`: write
`x/(2y)` to divide by both. A number after a term is not multiplied
(`x 2` is an error), and `2 -x` is a subtraction. Names written together
are one name: `xy` is the name `xy`, not `x*y`, and the error message says
so (`did you mean x*y?`).

**Unicode.** `−`, `×`, `·`, `÷`, `≤`, `≥` and `≠` work as `-`, `*`, `/`,
`<=`, `>=` and `!=`; superscript numbers are powers: `x²`, `x¹⁰`, `x⁻¹`.

## Functions

| Function | Notes |
|---|---|
| `sin(x)`, `cos(x)`, `tan(x)`, `sec(x)`, `csc(x)`, `cot(x)` | in radians |
| `arcsin(x)`, `arccos(x)`, `arctan(x)` | also `asin`, `acos`, `atan` |
| `arctan(y, x)` | the angle of the point `(x, y)`, in `(-π, π]` (`atan2`) |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | |
| `arcsinh(x)`, `arccosh(x)`, `arctanh(x)` | also `asinh`, `acosh`, `atanh` |
| `exp(x)`, `ln(x)` | `e^x` and the natural logarithm |
| `log(x)`, `log(x, b)` | base 10, or base `b` |
| `log2(x)`, `log10(x)` | |
| `sqrt(x)`, `cbrt(x)` | the cube root is defined for negative `x` |
| `hypot(x, y)` | `sqrt(x^2 + y^2)`, without overflow |
| `abs(x)`, `sign(x)` | |
| `floor(x)`, `ceil(x)`, `round(x)` | `round` rounds halves up: `round(2.5)` is 3 |
| `mod(x, m)` | the remainder, with the sign of `m`: `mod(-1, 3)` is 2 |
| `min(a, b, ...)`, `max(a, b, ...)` | any number of arguments |
| `clamp(x, low, high)` | `x` kept between `low` and `high` |
| `if(c, a)`, `if(c, a, b)` | see [Conditions](#conditions) |
| `and(a, b, ...)`, `or(a, b, ...)`, `not(a)` | see [Conditions](#conditions) |
| `gamma(x)`, `factorial(x)` | `factorial(x)` is `gamma(x + 1)`, for any real `x` |

A function's name must be followed by its arguments in parentheses:
`sin x` is an error (`sin(x)` is meant).

## Derivatives

Primes differentiate a function of one variable exactly, to any order up
to 10:

```python
graph = pgp.Graph()
graph.add("f(x) = x^3 - 2x")
graph.add("y = f'(x)")            # 3x^2 - 2
graph.add("y = f''(x)")           # 6x
print(graph.evaluate("f'(2)"))    # 10.0
print(graph.evaluate("sin'(0)"))  # 1.0: built-ins too
```

The derivative is exact (no step size, no rounding beyond the arithmetic
itself), also for functions written with `if`, `abs` or `min`, which have
a derivative on each piece. Python functions given to a graph are
differentiated as [`pgp.derivative`](plotting.md#derivatives) does it.

## Conditions

A comparison is a number: 1 where it holds, 0 where it does not. So
conditions combine with arithmetic, and `if` chooses between values:

```python
step = pgp.expression("x > 0")                      # 0, then 1
ramp = pgp.expression("(x > 0) x")                  # 0, then x
absolute = pgp.expression("if(x >= 0, x, -x)")
band = pgp.expression("if(and(x > -1, x < 1), 1)")  # undefined outside the band
```

- `if(c, a, b)` is `a` where `c` is true (not 0) and `b` where it is
  false; `if(c, a)` is undefined where `c` is false, which cuts a plot
  off: `if(x > 0, sqrt(x))`.
- `and(a, b, ...)`, `or(a, b, ...)` and `not(a)` work on truth values.
  Chained comparisons are not supported: write `and(-1 < x, x < 1)`.
- Where a condition is **undefined**, so is the result: `sqrt(x) > 1` is
  undefined for negative `x`, and so is `if(sqrt(x) > 1, 1, 0)`. A
  condition on an undefined value never silently counts as false.

## Points

In a graph, a point named `P` (a point item or a draggable point) is used
through its coordinates `P.x` and `P.y`:

```python
graph = pgp.Graph()
graph.draggable((1, 2), name="P")
graph.add("y = (x - P.x)^2 + P.y")          # a parabola with its vertex at P
graph.add("Q = (P.x + 1, 2P.y)")            # a point that follows P
print(graph.evaluate("hypot(Q.x - P.x, Q.y - P.y)"))  # 2.236...
```

Pairs `(a, b)` are only for writing points and parametric curves, as a
whole line: `(cos(t), sin(t))`. They cannot be used inside an expression.

## Undefined values

Outside a function's domain, an expression is NaN (undefined), without
errors or warnings: `sqrt(-1)`, `ln(0)` (minus infinity), `1/0`
(infinity), `arcsin(2)`. Plots leave a gap there, as for any function given
to the plane.

## Errors

Mistakes are reported with the place they were found and, when one is
likely, a suggestion:

```py
>>> pgp.expression("2 + * x")
ExpressionError: expected a number, a name or '(', not '*'
  2 + * x
      ^
>>> pgp.expression("sinx")
ExpressionError: unknown name 'sinx' (did you mean sin(x)?)
  sinx
  ^
```

`pgp.ExpressionError` is a `ValueError`; its `message`, `text` and
`position` attributes describe the problem for programs that show it
themselves. In a graph, an item with an error is not drawn, and its
`error` attribute says why (see [Graphs](graphs.md#errors)).

## `pgp.expression`

`pgp.expression(text, *variables, **names)` compiles text outside a
graph. The variables are the function's arguments, in order; by default
they are those of `x`, `y`, `t`, `θ` and `theta` that the text uses (in
that order), or `x` alone if it uses none. The other names are given as
keywords: numbers, functions or `(x, y)` points.

```python
f = pgp.expression("a h(x - P.x)", a=2, h=np.sinc, P=(1, 0))
print(f(np.array([0.0, 1.0, 2.0])))     # a NumPy function of x
g = pgp.expression("u v + w", "u", "v", w=1)
print(g(2, 3))                           # 7.0
print(f.text, f.variables)
```

Names given as keywords are fixed when compiling. For values that change
while the program runs, use a [graph](graphs.md): its parameters and
points are read every time the function is called.

## Limits

- Expressions may be nested about 120 deep (parentheses, powers, signs):
  far more than any formula, and a guarantee that evaluating one can never
  exhaust Python's recursion limit.
- Derivatives with primes go up to the 10th.
- `gamma` and `factorial` are computed point by point, so they are slower
  than the other functions on large arrays (still well under a millisecond
  per plot).
