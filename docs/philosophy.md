# Philosophy

PyGridPlot is an interactive Cartesian plane for pygame, driven entirely
from Python code. This page explains the principles behind it: why it
works the way it does, and how new features are judged. They matter to
anyone using the library (they explain what to expect from it) and to
anyone changing it (they decide what belongs in it).

## The principles

### 1. One way to do things, done well

There is one way to draw: call the plane's methods. Each kind of drawing
is one method (`plot`, `parametric`, `polar`, `implicit`, `fill_between`,
`inequality`, `vector_field`, `slope_field`, `points`, `polyline`,
`polygon`, `arrow`), and every improvement goes into that method, so
every program benefits from it. There is no second, parallel API doing
the same things another way, which would have to be learned, documented,
tested and kept in step with the first.

A small library that does one thing very well is worth more than a large
one that does many things adequately.

### 2. Python first

Pictures are written in Python. A function to plot is a Python function,
written with NumPy or `math`, a lambda or a `def`. Parts of a picture
refer to each other through ordinary Python: variables, closures, your own
classes. A wave that follows a draggable point is just

```py
plane.plot(screen, lambda x: p.y * np.sin(x - p.x))
```

There is no second language to learn (no expressions as text, no parser),
no naming system of its own, and no object model to fit your program into.
Python is already good at all of that.

### 3. A library, not a framework

Your program owns the main loop. The plane is an object you call
(`handle_event`, `draw_grid`, `plot`, ...); it never takes control, never
sleeps, never reads events by itself, and never draws outside its
rectangle. So it fits next to the rest of a pygame program (menus,
sidebars, games, several planes in one window), and it never decides how
your program is structured.

### 4. Immediate mode

Each frame, you draw what you want to see, and the plane draws it from
the current view. Nothing is created, updated or deleted: to change the
picture, draw something different next frame. Animation is free, every
plot follows panning and zooming, and the state of the picture is the
state of your program, in your own variables, where you can see it.

The plane keeps only what it must between frames, and keeps it explicit:
the view (which the user pans and zooms), the draggable points (which the
user moves), and caches that make frames fast without changing what they
show.

### 5. It just looks right

A plot should look the way a careful person would draw it, at every zoom
level, without settings:

- grid spacings are exact 1-2-5 steps and labels are exact decimals
  (never `0.30000000000000004`, never `-0`);
- labels never overlap, and fade rather than pop;
- panning and zooming are smooth, and nothing jumps;
- asymptotes are not joined, domain edges are found exactly, and curves
  stay smooth however far you zoom in;
- log axes are first-class, not an afterthought.

### 6. Robust by construction

A frame never breaks because of the program's functions. A function may
raise, return NaN, infinity or complex values, or only accept single
numbers; the plot simply has a gap where it is undefined. Invalid
*settings*, on the other hand, are rejected at once with a clear error,
when they are made, not later while drawing.

### 7. Fast without asking

Speed should not need the program's attention. Unchanged plots are
reused between frames automatically and exactly; the samplers spend
effort where a curve needs it; the hot paths are vectorised. A program
written plainly is a fast program.

### 8. Small, honest and well tested

PyGridPlot depends only on pygame-ce, NumPy and contourpy. All the
mathematics lives in a pure core that never imports pygame, so it is
tested headless, including with property tests that check the guarantees
above (no overlapping labels, no jumps, exact reuse) over many random
views. The runnable examples in the documentation run as tests, and the
API reference is generated from the code, so the documentation cannot
drift from what the library does.

## Deciding what belongs

The principles are also how features are judged. A new feature should:

- be a better plane method, or a new one, rather than a second way to
  draw something that can already be drawn;
- take Python values (functions, numbers, arrays), not a description in
  another language;
- keep state only if the user changes it;
- hold the guarantees above: exact, smooth, robust, fast, tested.

### Explored and retired

Three features were built, tried in real programs, and then deliberately
removed, because they failed these tests:

- **Retained plot objects** (`pgp.Function`, `pgp.Point`, ... kept by the
  plane and changed through their attributes). They duplicated every
  drawing method as a class with the same arguments, so every feature
  would have had to be written twice, and every program would have had to
  choose between two ways of doing the same thing (principle 1). Your own
  Python classes and variables already give a picture lasting parts.
- **Graphs**: a Desmos-like list of expressions written as text
  (`"y = a sin(x - P.x)"`), with names and saving to JSON. A second way to
  build pictures (principle 1), in text rather than Python (principle 2).
- **An expression language** compiling text to NumPy functions. A second
  language next to Python (principle 2).

They are not planned to come back. Programs that want them can build them
on top of the plane's methods, which is exactly what those features did;
see [the developer guide](dev/architecture.md#retired-experiments) for
where their code can still be found.
