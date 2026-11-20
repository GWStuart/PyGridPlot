# Project Plan: Interactive Cartesian Plotting Library for Python

> **Status:** Early concept. Treat this document as a template, not a contract. Sections marked *Open question* are deliberately undecided.

---

## 1. Overview

This is a small Python library for drawing mathematical objects (functions, points, polygons and, later, parametric curves, implicit curves and inequalities) onto an interactive Cartesian plane that can be panned and zoomed, similar to the Desmos graph view.

Unlike Desmos, the library has **no UI apart from the plane itself**. There is no expression list, no sliders and no input boxes. Everything is defined in Python code. The library's job is to:

1. Render a good-looking, adaptive Cartesian plane (grid, axes, labels).
2. Handle pan and zoom interaction.
3. Draw mathematical objects onto that plane accurately and robustly.
4. Stay out of the programmer's way so they control the main loop, timing, events and redraws.

The main design value is **flexibility**. Animation, interactivity and experiments should be easy because the user owns the program, not the library.

---

## 2. Goals and Non-Goals

### Goals (v1)

- An interactive plane with smooth panning (mouse drag, keyboard) and zooming (scroll wheel, zooming about the cursor).
- An **adaptive grid** with major and minor lines, axes and tick labels that look good at every zoom level. *This is the main technical challenge of v1 (see Section 8).*
- Plotting of **explicit functions** $y = f(x)$ given as ordinary Python callables.
- Plotting of **points** (single points and collections).
- Plotting of **polygons and polylines** from a list of vertices.
- A library-style API: the user writes the main loop and the library never takes control of it.
- A convenience `show()` helper for quick, non-animated use.

### Non-goals (v1)

- No in-window UI for typing expressions, sliders, legends, menus and so on.
- No symbolic maths or expression parsing from strings.
- No 3D.
- No parametric curves, implicit curves or inequalities *yet*. The architecture must leave room for them (see Section 14).

---

## 3. Design Principles

1. **Library, not framework.** The user owns the loop. The library provides objects the user calls: `handle_event()`, `draw_*()`, `plot()`.
2. **Immediate mode first.** Drawing calls are issued every frame and nothing is remembered between frames, so animation needs no extra work. A retained-mode layer (persistent objects) may be built on top later.
3. **Separate the maths from rendering.** All coordinate maths, sampling, grid computation and geometry live in pure Python/NumPy modules with **no pygame imports**. A thin backend layer does the drawing.
4. **One internal drawing primitive.** Everything is reduced to world-space **polylines** (and later **filled regions**) plus a **style**. New features produce polylines; the renderer stays simple.
5. **Robust by default.** User functions that throw, return `NaN`, blow up to infinity or have asymptotes must never crash the program or draw garbage.
6. **Sensible defaults, everything overridable.** It should look good with zero configuration, and colours, widths, spacing targets and fonts should all be configurable.

---

## 4. Architecture

A suggested package layout is below. However feel free to change it if we need to. I want the project as a whole to be well organised and easy to work with an extend and so it is important to have a good architecture that accomodates for this.


```
PyGridPlot/
├── core/                  # Pure maths. No pygame imports.
│   ├── view.py            # View state, world <-> screen transforms, pan/zoom maths
│   ├── grid.py            # Grid spacing selection, line generation, label formatting
│   ├── sampling.py        # Sampling explicit functions into polylines
│   ├── geometry.py        # Polyline/polygon types, clipping helpers
│   └── style.py           # Style dataclasses (colours, widths, markers, grid theme)
├── backends/
│   └── pygame_backend.py  # Converts primitives to pygame draw calls; font cache
├── plane.py               # Public Plane class: ties view + grid + backend together
├── interaction.py         # Mapping pygame events to view operations
└── quick.py               # show() helper with a default main loop
tests/
examples/
```

### Data flow (one frame)

```
user code ──► Plane.plot(f) ──► sampling.py ──► world polylines
                                                     │
                     view.py (world → screen) ◄──────┘
                                                     │
                     backend draws screen polylines ◄┘
```

The grid follows the same path: `grid.py` produces lines and labels in world space, and the backend draws them.

---

## 5. Public API Sketch

*This is a starting point; names and signatures are open to change.*

### Typical user program

```python
import pygame
import numpy as np
import PyGridPlot

pygame.init()
screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
plane = PyGridPlot.Plane(rect=screen.get_rect())
clock = pygame.time.Clock()
t = 0.0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        consumed = plane.handle_event(event)   # True if the plane used the event
        # ... user's own event handling here ...

    t += clock.tick(60) / 1000

    screen.fill("white")
    plane.draw_grid(screen)
    plane.plot(screen, lambda x: np.sin(x - t), color="crimson", width=2)
    plane.points(screen, [(np.cos(t), np.sin(t))], color="navy", radius=5)
    plane.polygon(screen, [(0, 0), (1, 0), (1, 1)], color="green", fill=False)
    pygame.display.flip()
```

### Quick mode

Rough idea for quick mode (open to changes though)

```python
PyGridPlot.show(lambda x: x**2, lambda x: np.sin(x))
```

### Core public surface (v1)

| Method / attribute | Purpose |
|---|---|
| `Plane(rect, *, center=(0,0), scale=50, equal_aspect=True, theme=None)` | Create a plane drawn into `rect` |
| `plane.handle_event(event) -> bool` | Pan/zoom handling; returns whether the event was used |
| `plane.draw_grid(surface)` | Draw grid, axes and labels |
| `plane.plot(surface, f, **style)` | Plot an explicit function |
| `plane.points(surface, pts, **style)` | Draw one or more points |
| `plane.polyline(surface, pts, **style)` / `plane.polygon(...)` | Draw connected vertices |
| `plane.to_world(px, py)` / `plane.to_screen(x, y)` | Coordinate helpers for users |
| `plane.view` | Access/modify centre, scale and visible bounds directly |
| `plane.set_view(x_range=..., y_range=...)` | Fit the view to a region |
| `plane.reset_view()` | Return to the initial view |
| `plane.interactive = True/False` | Turn built-in pan/zoom on or off |

*Open question:* should drawing calls take `surface` every time, or should there be a `with plane.frame(screen):` context? Passing `surface` explicitly is simpler and more obvious, so start there. I think I like passing `surface` because it mimicks the way other pygame draw calls are.

---

## 6. Coordinate System and View

### View state

- `center = (c_x, c_y)` in world units
- `scale_x`, `scale_y` in pixels per world unit (equal by default)
- `rect` is the screen rectangle the plane occupies, with centre $(o_x, o_y)$

### Transforms

World → screen:

$$
p_x = o_x + s_x (x - c_x), \qquad p_y = o_y - s_y (y - c_y)
$$

Screen → world:

$$
x = c_x + \frac{p_x - o_x}{s_x}, \qquad y = c_y - \frac{p_y - o_y}{s_y}
$$

(The minus sign is there because screen $y$ points down.)

**Requirement:** every conversion goes through `view.py`. There should be no ad-hoc transforms anywhere else. All transforms should accept NumPy arrays so they can work on whole polylines at once.

### Visible bounds

$x_{\min} = c_x - \frac{w/2}{s_x}$, $x_{\max} = c_x + \frac{w/2}{s_x}$, and similarly for $y$. These drive both sampling and grid generation.

### Window resize

When `rect` changes, keep the **centre fixed** and the **scale fixed**, so more of the plane becomes visible rather than the content stretching.

---

## 7. Interaction

| Input | Action |
|---|---|
| Left-drag | Pan: $c_x \mathrel{-}= \Delta p_x / s_x$, $c_y \mathrel{+}= \Delta p_y / s_y$ |
| Scroll wheel | Zoom about the cursor by a factor per notch (e.g. 1.1) |
| Arrow keys | Pan by a fixed fraction of the view |
| `+` / `-` | Zoom about the centre |
| `Home` / `0` | Reset view |

### Zoom about the cursor (requirement)

The world point under the mouse must stay under the mouse while zooming:

1. $w = \text{to\_world}(m)$ before the zoom.
2. Update the scale: $s \leftarrow s \cdot k$.
3. Recompute the centre:

$$
c_x = w_x - \frac{m_x - o_x}{s_x}, \qquad c_y = w_y + \frac{m_y - o_y}{s_y}
$$

### Other details

- Use `MOUSEWHEEL` with `precise_y` (pygame 2) so trackpads zoom smoothly.
- Only begin a drag when the mouse goes down **inside** the plane's rect.
- `handle_event` returns `True` when it used the event, so user code can avoid double-handling.
- Clamp the scale to a supported range (see Section 8.8 on precision).
- *Optional:* smooth/eased zoom animation, and independent axis scaling (e.g. Shift + scroll).

---

## 8. The Adaptive Grid ⭐ (Main v1 Challenge)

The grid is the first thing a user sees and the thing that most decides whether the tool feels polished. It must look good and readable at **every** zoom level, from $10^{-6}$ to $10^{6}$ units across the screen, and it must behave sensibly during continuous zooming. **Expect to spend a large part of v1 here, and iterate on it visually.**

### 8.1 Requirements

- Major grid lines, minor grid lines, and emphasised axes ($x=0$, $y=0$).
- Spacing is always a "nice" number: $\{1, 2, 5\} \times 10^k$.
- Spacing on screen stays within a comfortable pixel range at every zoom.
- Tick labels on major lines, formatted cleanly (no `0.30000000000000004`, no `-0`).
- Labels stay visible when an axis is off-screen (they stick to the nearest screen edge, as in Desmos).
- No flicker, no "popping" that feels jarring, and no gaps or drift after long panning.
- Crisp 1 px lines (no antialiasing blur).
- Fast enough to redraw at 60 fps.
- Fully themeable (colours, widths, target spacings, font, show/hide parts).

### 8.2 Choosing the major spacing

Given a target pixel spacing $T$ (e.g. ~100 px, configurable) and scale $s$ (px/unit):

1. Raw spacing in world units: $r = T / s$.
2. Exponent: $k = \lfloor \log_{10} r \rfloor$, mantissa $m = r / 10^k \in [1, 10)$.
3. Choose the nice mantissa $n \in \{1, 2, 5, 10\}$ closest to $m$. (Compare in log space so that rounding is symmetric.)
4. Major spacing $\Delta = n \cdot 10^k$.

Compute this **independently for $x$ and $y$** so that unequal axis scaling works later.

*Open question:* nearest nice number vs "smallest nice number $\ge r$". Try both and see which feels better.

### 8.3 Minor subdivisions

The usual convention (close to Desmos):

| Major mantissa | Minor spacing | Subdivisions |
|---|---|---|
| 1 | $\Delta / 5$ | 5 |
| 2 | $\Delta / 4$ | 4 |
| 5 | $\Delta / 5$ | 5 |

### 8.4 Transitions between spacing levels

When the zoom crosses a threshold, the spacing jumps (e.g. 1 → 2). Options, from simplest to most polished:

1. **Hard snap:** just switch. Simple, and Desmos mostly does this.
2. **Hysteresis:** switch up at one threshold and down at a slightly different one, so the grid doesn't flicker when the zoom sits on a boundary.
3. **Alpha fading:** fade minor lines in or out based on their current on-screen spacing:
   $\alpha = \operatorname{clamp}\left(\frac{\text{px\_spacing} - a}{b - a}, 0, 1\right)$
   This gives continuous-looking zoom. It needs drawing with alpha (draw onto an `SRCALPHA` surface, or blend colours with the background manually).

Start with (1), add (2), and treat (3) as polish once everything else works.

### 8.5 Generating line positions

- Compute integer index ranges, **not** accumulated floats:
  $i_{\text{start}} = \lceil x_{\min} / \Delta \rceil$, $i_{\text{end}} = \lfloor x_{\max} / \Delta \rfloor$, and position $x_i = i \cdot \Delta$.
  Adding $\Delta$ repeatedly drifts; multiplying an integer does not.
- Decide whether a minor line is also a major one using **integer arithmetic on indices** (e.g. `i % subdivisions == 0`), never float modulo.
- Draw minor lines first, then major lines, then axes, then labels (painter's order).
- Cap the number of lines as a safety net (e.g. ≤ 1000 per axis).

### 8.6 Pixel snapping and crispness

- Round line positions to whole pixels before drawing. For 1 px lines with pygame's integer coordinates, `round()` is enough. Keep this in mind if a sub-pixel or GPU backend is added later.
- Grid lines are axis-aligned, so draw them as plain (non-antialiased) lines or 1 px-wide rects for crispness.
- Use consistent widths, e.g. minor 1 px, major 1 px in a darker colour, axes 2 px.

### 8.7 Tick labels

**Formatting**

- Decimal places from the spacing: `decimals = max(0, -floor(log10(Δ)))`. Spacing 0.2 gives 1 dp, spacing 0.05 gives 2 dp.
- Switch to scientific notation outside a configurable range (e.g. $|v| \ge 10^6$ or nonzero $|v| < 10^{-4}$). Render as `2×10⁶` or `2e6` (*open question:* which style. I don't mind too much but it should look clean visually).
- Get rid of `-0` and floating-point junk by formatting from the integer index where possible: value $= i \cdot n \cdot 10^k$. The digits can be built from $i \cdot n$ and $k$ instead of a lossy float.
- Label the origin once, as `0`, not on both axes.

**Placement**

- $x$ labels sit just below the $x$-axis and $y$ labels just left of the $y$-axis.
- If an axis is off-screen, pin its labels to the corresponding screen edge (still inside the rect).
- Skip labels that would overlap: if the label's pixel width exceeds the spacing, label every 2nd or 5th line, or shrink the target spacing.
- Don't draw labels that would sit on top of the other axis.
- Optional: a small background "halo" (a background-coloured rect or outline) so labels stay readable over grid lines and curves.

**Performance**

- `pygame.font.Font.render` is slow. **Cache rendered label surfaces** keyed by `(text, colour, size)`, with a bounded size such as an LRU cache.

### 8.8 Floating-point precision limits

`float64` has about 15–16 significant digits. Problems appear when:

- Zoomed very far in: the spacing becomes tiny compared with the coordinates, e.g. centre at $10^8$ with spacing $10^{-9}$.
- Computing $(x_i - c_x) \cdot s$: subtracting two close, large numbers loses precision, and lines jitter or merge.

v1 approach:

- **Clamp the zoom range** to a range that renders correctly (to be found experimentally, e.g. scale between $10^{-6}$ and $10^{10}$ px/unit) and document the limit.
- Where possible compute offsets relative to the view: $x_i - c_x = (i - i_c)\Delta - (c_x - i_c\Delta)$, where $i_c = \operatorname{round}(c_x/\Delta)$. This keeps the numbers small before they are multiplied by the scale.
- Stretch goal: arbitrary-precision view centres (`decimal`/`fractions`) for deep zoom. Probably not worth it in v1.

### 8.9 Theming

Not sure what the best way to support themeing would be but could do something like that. Have a `GridTheme` dataclass, with roughly:

```python
@dataclass
class GridTheme:
    background: Color = "white"
    minor_color: Color = (230, 230, 230)
    major_color: Color = (200, 200, 200)
    axis_color: Color = (40, 40, 40)
    minor_width: int = 1
    major_width: int = 1
    axis_width: int = 2
    target_major_px: float = 100
    show_minor: bool = True
    show_major: bool = True
    show_axes: bool = True
    show_labels: bool = True
    label_font: str | None = None
    label_size: int = 14
    label_color: Color = (60, 60, 60)
```

Also provide a dark theme preset early. It's a good test that nothing is hard-coded.

### 8.10 Testing the grid

- **Unit tests (pure maths):** nice-number selection at many scales; minor subdivision counts; index ranges covering exactly the visible region; label strings (`0.1`, `0.2`, `0.3`, never `0.30000000000000004`); no `-0`; scientific notation at the thresholds.
- **Property tests (e.g. Hypothesis):** for random centres and scales, every generated line lies within the visible bounds and spacing in pixels stays within $[T/2.5, 2.5T]$ (or whatever bounds the selection rule guarantees).
- **Visual test script:** an example that sweeps the zoom smoothly through many decades and pans across large offsets, for checking by eye.
- **Golden images (optional):** save frames at fixed views and compare them after refactors.

### 8.11 Suggested build order for the grid

1. Major lines only, hard snap, no labels.
2. Axes.
3. Minor lines.
4. Labels with basic formatting and placement.
5. Edge-pinned labels when axes are off-screen.
6. Label overlap avoidance and label caching.
7. Hysteresis.
8. Precision handling and zoom clamping.
9. Alpha fading (polish).

---

## 9. Plotting Explicit Functions

### Input

Any Python callable `f(x) -> y`. Users may write it with NumPy (vectorised) or with `math` (scalar only).

### Sampling pipeline

1. Get the visible $x$-range, plus a small margin.
2. Create samples, starting at about **one per pixel column**: `xs = np.linspace(x_min, x_max, width_px)`.
3. **Evaluate robustly:**
   - Try `f(xs)` on the whole array inside `np.errstate(all="ignore")`.
   - If it raises, or returns the wrong shape or a scalar, fall back to a scalar loop that wraps each call in `try/except` and turns failures into `NaN`.
   - Optionally remember per function which path worked, so the vectorised attempt isn't retried every frame.
   - Convert complex results to `NaN` (or take the real part only if the imaginary part is ~0; *open question*).
4. **Split into segments** at `NaN`/`inf` values.
5. **Detect discontinuities** so that asymptotes aren't joined by vertical lines:
   - Basic heuristic: break between neighbouring samples if the jump exceeds the screen height **and** the sign changes.
   - Better: evaluate at the midpoint. If $f(x_{\text{mid}})$ isn't roughly between $f(x_a)$ and $f(x_b)$, recurse or break.
6. **Adaptive refinement (later):** subdivide where the curve bends sharply or changes quickly in pixels, up to a depth limit. This fixes functions like $\sin(1/x)$ and steep curves.
7. **Clip** very large $y$-values to slightly outside the visible rect before converting to pixels, since pygame misbehaves with huge coordinates.
8. Hand the segments to the backend as polylines.

### Requirements

- A user function that throws must never crash the frame.
- `1/x`, `tan(x)`, `sqrt(x)`, `log(x)`, `floor(x)` and `x**2` should all look correct by v1 (a good shared test set).
- Performance: one function at one sample per pixel should take well under a millisecond with NumPy.

### Documentation note: lambda late binding

```python
fs = [lambda x: x**k for k in range(3)]   # all three are x**2!
fs = [lambda x, k=k: x**k for k in range(3)]  # correct
```

Users animating families of curves will hit this, so document it.

---

## 10. Points, Polylines and Polygons

- **Points:** accept a single `(x, y)`, a list of tuples, or an `(N, 2)` NumPy array. Style: colour, radius in pixels, marker shape (circle first; square/cross later), optional outline.
- **Polylines:** an ordered list of vertices drawn connected.
- **Polygons:** a closed polyline, with an option to fill. Filled polygons with transparency need an alpha surface.
- Points keep a **constant size in pixels** regardless of zoom (as in Desmos). Polygons are in world units, so they scale with zoom.
- Only draw what's on screen: check bounding boxes and skip anything entirely outside the rect.

---

## 11. Styling

- Keyword arguments on each call (`color=`, `width=`, `radius=`, `alpha=`, `dash=`), plus style dataclasses for reuse.
- Colours accept anything pygame accepts (names, RGB tuples, hex strings).
- A default colour cycle (like Desmos or matplotlib) for plots without a colour, *if* this fits with immediate mode. *Open question:* the cycle would need to reset each frame, e.g. in `draw_grid` or with a `plane.begin_frame()` call.
- Dashed lines: a stretch goal (pygame has no built-in support).

---

## 12. Rendering Backend Notes (pygame)

- **Prefer pygame-ce** (the actively maintained community fork). It is essentially a drop-in replacement and is being improved more actively.
- **Thick antialiased lines** are a known pygame weakness. Options:
  - `aalines` for 1 px curves.
  - For thicker curves, draw each segment as a thin filled quad (polygon), possibly with round joins drawn as circles.
  - Draw at 2× resolution and smoothscale down (expensive; last resort).
- **Clip drawing to the plane's rect** with `surface.set_clip(rect)` so the plane can be embedded in part of a window.
- **HiDPI:** check how pixel sizes behave on high-DPI displays. Possibly add a `pixel_ratio` setting.
- Keep the backend interface small, e.g. `draw_polyline(points_px, style)`, `draw_line`, `draw_circle`, `draw_polygon`, `draw_text`, `set_clip`. This makes a future OpenGL backend (pyglet/arcade/moderngl) realistic.

---

## 13. Testing and Quality

- **Unit tests** for everything in `core/`: transforms (round trip world→screen→world), zoom-about-cursor keeping the cursor point fixed, grid maths, sampling (segment splitting for `1/x`, NaN handling, scalar fallback).
- **Examples directory** doubling as manual visual tests: `basic.py`, `animation.py`, `many_functions.py`, `asymptotes.py`, `zoom_sweep.py`, `embedded_rect.py`, `dark_theme.py`.
- **Headless runs:** `SDL_VIDEODRIVER=dummy` lets render tests run in CI.
- Tooling suggestions: `pytest`, `hypothesis`, `ruff`, type hints with `mypy`/`pyright`, `pyproject.toml` packaging.

---

## 14. Roadmap

*Rough milestones. The order within each milestone is flexible.*

### Milestone 0: Skeleton
- Package layout, `pyproject.toml`, test setup.
- `View` class with transforms and unit tests.

### Milestone 1: Interactive plane
- `Plane` with `handle_event` (drag pan, wheel zoom about cursor, keys, reset).
- Simple placeholder grid (fixed spacing) so interaction can be tested.

### Milestone 2: Adaptive grid ⭐
- Everything in Section 8, following the build order in 8.11.
- Light and dark themes.
- The zoom-sweep visual test looks good.

### Milestone 3: Explicit functions
- Robust sampling, NaN handling, scalar fallback, discontinuity breaking, clipping.
- Shared function test set renders correctly.

### Milestone 4: Points and polygons
- Points, polylines, polygons (outline and fill), styling.

### Milestone 5: Convenience and polish
- `show()` helper, coordinate helpers, `set_view`, label halos, grid fading, docs and examples.
- First release.

### Later (architecture should allow these)
- **Parametric curves** $(x(t), y(t))$: same pipeline as explicit functions, sampled over $t$, with adaptive refinement on arc length in pixels.
- **Polar curves** $r(\theta)$: a special case of parametric.
- **Implicit curves** $F(x, y) = 0$: evaluate on a grid and extract contours with **marching squares**, with refinement near the curve. A GPU backend could do this per pixel in a shader.
- **Inequalities** $F(x, y) < 0$: filled regions, by per-pixel evaluation (NumPy into a surface via `surfarray`, or a shader) or contours plus filling.
- **Retained-mode layer**: persistent `Function`/`Point` objects with mutable properties.
- **Export**: save frames to PNG (and image sequences for video), and possibly SVG export using the backend-independent primitives.
- **Interactivity helpers**: hit-testing points under the mouse, draggable points.
- **OpenGL backend** for better lines and GPU implicit plotting.
- Polar grid option, log-scale axes.

---

## 15. Known Pitfalls (Quick Reference)

- Screen $y$ points down, so the sign in the transforms is easy to get wrong.
- Accumulating `x += step` drifts; use `i * step`.
- Float modulo for "is this a major line?" is unreliable; use integer indices.
- `0.1 + 0.2` label junk; format from integers and exponents.
- `math.sin` on NumPy arrays raises; you need the scalar fallback.
- Huge pixel coordinates break pygame drawing; clip first.
- Joining samples across asymptotes draws vertical lines.
- Font rendering every frame is slow; cache it.
- Lambda late binding in loops.
- Precision loss at extreme zoom; clamp and compute relative to the centre.

---

## 16. Open Questions

- Pass `surface` to every call, or use a frame context?
- Nearest vs ceiling rule for choosing nice grid spacing.
- How to style scientific-notation labels.
- How a default colour cycle fits with immediate mode.
- How to handle complex return values from user functions.
- Should `equal_aspect=True` be enforced or just the default?
- What zoom range to support in v1 (find experimentally).
