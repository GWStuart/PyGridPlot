# Draggable points

Draggable points are the only plotted objects with state that lives between
frames: their positions, which the user changes. Everything else in the
library is immediate mode, so the design keeps that state in one obvious
place, a `DraggablePoint` object created once with `Plane.draggable`, and
leaves the rest immediate: user code reads the point's position every frame
and draws the points explicitly with `draw_draggables`.

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

`Plane.handle_event` offers every event to the `DragController` first.

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

`Plane.update` (called by `draw_grid`) then calls `DragController.update`,
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

The user chose "nearest, but continuous". A plain nearest point jumps
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
`begin_frame` (called by `draw_grid`), `moved` becomes whether the count
changed since the previous frame. A frame thus runs from one `draw_grid` to
the next, and `moved` stays the same while the user's code draws.
