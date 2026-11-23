# Draggable points and sliders

Draggable points and [sliders](#sliders) are the only objects with state
that lives between frames: their positions and values, which the user
changes. Everything else in the
library is immediate mode, so the design keeps that state in one obvious
place, a `DraggablePoint` object created once with `Grid.draggable`, and
leaves the rest immediate: user code reads the point's position every frame
and draws the points explicitly with `draw_controls`.

| Module | Role |
|---|---|
| `core/handles/point.py` | `DraggablePoint`: position, constraint, style, flags (`dragging`, `hovered`, `moved`). Pure. |
| `core/handles/constraints.py` | `AxisConstraint` (free and locked, with bounds) and `CurveConstraint` (on a parametric curve, with its sampled-curve cache and the exact refinement). |
| `core/handles/tracking.py` | `nearest_on_curve`: following a sampled curve towards the cursor without jumping between branches. |
| `core/handles/labels.py` | The coordinate label: rounding to a pixel, formatting, placement. |
| `core/handles/drawing.py` | Markers (point, hover ring) and the label as ordinary primitives. |
| `dragging.py` | `DragController`: the pygame side (hit test, grab, motion, release, hover, cursor). |

`core.handles` builds on both `core.plot` (curves, styles, markers) and
`core.grid` (number formatting, the theme); neither imports it, and the
architecture test enforces this. The renderer needs nothing new: points are
`Markers` and their labels `TextItem`s.

## Events

`Grid.handle_event` offers every event to the `DragController` first.

1. **Button-down** inside the plane: `DragController.hit` finds the
   nearest enabled, visible point within `radius + edge_width +
   grab_margin_px` of the cursor (ties go to the most recently created,
   which is drawn on top). If there is one, the press is used, so the pan
   controller never sees it, and the offset between the cursor and the
   point's centre is recorded.
2. **Motion** while dragging: the target is the cursor less the grab offset,
   clamped to the plane's rectangle, and `DraggablePoint.drag_to` moves the
   point through its constraint. The event is also passed on to the pan
   controller, which only tracks the cursor (for zooming about it).
3. **Motion** otherwise updates the hovered point and the cursor: the hand
   cursor is set while over a point, and the cursor that was there before is
   restored afterwards (only if the plane changed it). Nothing is hovered
   while the plane itself is being panned.
4. **Button-up**, `WINDOWLEAVE` or `WINDOWFOCUSLOST` end the drag.

`Grid.update` (called by `grid.draw`) then calls `DragController.update`,
which drags the grabbed point to the last cursor position again. The view
may have eased (a wheel zoom in the middle of a drag) or the point's curve
may have changed, and the point must stay under the cursor regardless.

`interactive=False` disables the pan controller only; the drag controller
always runs.

## Constraints

- **Free and locked points** (`AxisConstraint.drag`) turn the target into
  *transformed* coordinates (`View.screen_to_transformed`), then world
  coordinates, replace the locked coordinate, and clamp to the bounds
  (`AxisConstraint.place`). A coordinate that is not finite, or that a log
  axis cannot show, keeps its current value, so on a log axis a point can
  never reach zero.
- **Points on a curve** (`CurveConstraint.drag`) work on screen, so they
  follow the curve as it is drawn on unequal and log axes alike:
  1. The curve is sampled on screen with `curves.sample_curve_pieces`
     (the adaptive curve sampler, keeping each sample's parameter) over the
     plane plus half its size on every side. The samples are cached until
     the view changes, the frame ends (`invalidate`, called from
     `begin_frame`, in case the curve depends on time) or the curve is
     replaced.
  2. `nearest_on_curve` picks the parameter (below).
  3. `_refine` samples the real curve across the segment found, twice
     narrowing 16 times, so the result is the true nearest point and no
     longer depends on the sampling's chord tolerance.
  4. The point is placed exactly at `(x(t), y(t))`.

### Following a curve

**Graphs follow the cursor's x.** If the sampled curve's screen x keeps
moving one way through all its pieces (gaps included), it is a graph over
x (`tracking.graph_axis`; over y likewise, if its y does; closed curves
never are), and a dragged point goes to the point of the curve at the
target's x (`along_axis`: interpolated on the segment that spans it, or
the nearest piece end along x in a gap or beyond the ends), refined on
the real curve by matching x rather than distance. This replaced the
nearest-point tracking below for graphs, which lagged and jumped on any
wavy graph: sweeping the cursor sideways at a constant height across the
derivatives example's `sin x + 0.3 sin 2.7x + 0.05x²`, the nearest point
stayed in a local minimum of the distance (the crest beside a trough the
cursor was over) up to 190 px behind the cursor, and then jumped up to
290 px in one 2 px step; following x, it is within 0.01 px of the cursor's
x everywhere the graph is on screen. Placing a point from code
(`position=`) still takes the nearest point.

**Other curves follow the nearest point, continuously.** The user chose
"nearest, but continuous". A plain nearest point jumps
wherever two parts of a curve are about equally close: at the crossing of a
figure-eight, between the turns of a spiral, between nearby branches.
`nearest_on_curve` instead:

1. projects the target onto every sampled segment at once (distance and
   interpolated parameter);
2. finds the *global* nearest segment;
3. starting from the segment that holds the point's current parameter,
   slides along the curve in both directions while the distance decreases
   (`_descend`, vectorised with `np.diff`), giving the *local* nearest
   point;
4. takes the global one only if it is closer by more than `SWITCH_PX`
   (10 px), otherwise the local one.

A curve is **closed** if it is defined at both ends of its range and the
ends are within `CLOSED_PX` (0.5 px) on screen. Sliding then wraps round:
a closed curve in one piece becomes a cyclic chain, and when gaps split it
into pieces, the last piece is joined to the first. Gaps in the curve
(undefined parameters) split it into pieces, so crossing a gap takes a
"clearly closer" jump.

Measured on a spirograph whose 18 branches all cross at one point, dragged
along with the cursor wandering smoothly up to 12 px off the curve, the
point never left its branch; it does so only when the cursor wanders much
further from its branch than from another.

## The coordinate label

Each coordinate is rounded to `10^floor(log10(p))`, where `p` is the world
size of a pixel at that coordinate, measured through the axis transform
(so on a log axis the precision is relative). The rounded `Decimal` is
formatted with the grid's `format_value` and the theme's `NumberFormat`;
leading signs inside the label are ordinary text. The label sits above and
to the right of the marker, flips left near the right edge and below near
the top, and is not drawn at all for a point off screen.

## Frames and `moved`

`DraggablePoint` counts changes of its position (`_version`). At
`begin_frame` (called by `grid.draw`), `moved` becomes whether the count
changed since the previous frame. A frame thus runs from one `grid.draw` to
the next, and `moved` stays the same while the user's code draws.

## Sliders

A `Slider` (`core/handles/slider.py`) is a value on a track, created once
with `Grid.slider` and read every frame, like a draggable point; but it
lives on the screen, in a panel in a corner of the plane, not in the
world, so panning and zooming leave it alone.

| Module | Role |
|---|---|
| `core/handles/slider.py` | `Slider`: the value, its range, step and log track, the mapping between the value and a point of the track, the text, flags (`dragging`, `hovered`, `moved`). Pure. |
| `core/handles/panel.py` | `layout_panels`: one panel per corner, rows top to bottom; `SliderRow`: where a slider is drawn and what a press there means; the primitives. |
| `dragging.py` | `SliderController`: presses, drags, the wheel, hover and the cursor. |

**Values.** With a step, values are `low + k · step` computed in `Decimal`
(`repr` of the floats), so a step of 0.1 gives 0.3 exactly; a step that is
an `int` makes an integer slider whose `value` is an `int`, typed
`Slider[int]` through overloads of `Grid.slider`. Without a step, a
dragged value is rounded to the size of one pixel of the track, with
`labels.rounded_value` (the coordinate labels' rounding), on a linear or a
log transform. The text shows as many decimals as the step has (at most
two beyond its first digit, for steps such as π/12), or one pixel resolves.

**Layout and events.** `Grid.draw_controls` lays out the sliders it
draws (`layout_panels`, which sets each slider's `track_px`) and keeps the
panels until the next frame: `begin_frame` clears them, so a slider not
drawn in a frame cannot be dragged, which is how a program hides one. The
`SliderController` sees events before the `DragController`: a press on a
row's band grabs the slider (moving the handle to the press, unless the
press is on the handle), a press or the wheel anywhere on a panel is used
up (the plane never pans or zooms from under it), and the wheel over a
row nudges the slider by a step. Its `owns_cursor` blocks points under a
panel from being hovered or grabbed; a pan or a point's drag blocks the
sliders in turn.

**Drawing fast.** A slider is a few strokes and markers, but rasterising
them every frame costs about 1 ms per panel. The plane keeps the
primitives of each panel's background and of each slider, with a key of
everything they depend on (`panel.row_key` and the theme), and draws the
same objects again when the key is unchanged, so the renderer reuses
their coverage: a panel at rest costs only blending, and a drag rebuilds
only the dragged slider.


### The legend and the colour bar share the corners

The legend (`core/handles/legend.py`, drawn by `Grid.legend`) is a panel
in a corner too, with the same look (`panel.background_primitives`), and it
lives in the handles layer for that reason, although nothing in it moves.
Its panel is added to what `SliderController` sees as panels (an empty
`Panel`, `Grid._overlays`), so presses and the wheel on it are used up and
points under it cannot be grabbed, like the sliders' panels. When the two
share a corner, the **sliders keep their place** and the legend goes
beyond their panel: a legend whose size changes (an entry appears) must
never move a slider from under the cursor. The legend goes by the sliders'
panels as drawn in this frame or the last (`Grid._slider_layout`), so the
order of `legend` and `draw_controls` in a frame does not matter; in the
very first frame, before any slider is drawn, it expects all of them. Like
the slider panels, its background and each swatch keep their primitives
while unchanged, so a legend at rest costs only blending (0.5 ms for eight
entries against 2.5 ms built afresh), and a label that changes is only
written again. The colour bar (`core/handles/color_bar.py`, drawn by
`Grid.color_bar`) is a third such panel: it goes beyond both the sliders
and the legend in its corner (`Grid._occupied(corner, legend=True)`, with
the legend as drawn in this frame or the last), and presses on it are used
up too.
