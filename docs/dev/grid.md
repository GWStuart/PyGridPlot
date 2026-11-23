# The grid

`core.grid.scene.build_grid(view, theme, measurer, state, x_title=,
y_title=)` turns the view into a `GridScene`: line batches in painter's
order, the axes' arrowheads, positioned labels and axis titles, and the
spacing chosen for each axis. It is a pure function of its arguments; the
small `GridState` only carries hysteresis for `"snap"` mode.

```text
                 for each axis                                        both axes
 ┌──────────────────────────────────────────────────────────────┐  ┌─────────────────────┐
 │ spacing.py ──> ticks.py ──> gridlines.linear_axis_lines ─┐   │  │ gridlines.line_     │
 │  (choose)      (classify)                                ├─> AxisLines ─>  batches   │
 │                logticks.py (three tiers) ────────────────┘   │  │ + axis_batches,     │
 │                                                              │  │   axis_arrows       │
 │                                                              │  │                     │
 │ ticklabels.py / loglabels.py ──> LabelCandidates ────────────┼─>│ layout.py: place    │
 │  (which labels, how opaque)                                  │  │ labels and titles,  │
 │                                                              │  │ resolve collisions  │
 └──────────────────────────────────────────────────────────────┘  └─────────────────────┘
```

`scene.axis_grid()` runs the left half for one axis (linear or log) and
returns an `AxisGrid` (its lines, its label candidates and how overlapping
labels are treated); `build_grid()` then paints and places both axes.

## Exact spacings: `nice.py`

A spacing is a `NiceStep`, identified by an integer **ladder level**:
`NiceStep(level)` is `(1, 2, 5)[level % 3] × 10^(level // 3)`, so level 0 is
1, level 1 is 2, level 2 is 5, level 3 is 10, level -1 is 0.5, and so on.
Steps are never floats:

- one step coarser or finer is `level ± 1`;
- the minor step of a major step is always `level - 2` (1 → 0.2, 2 → 0.5,
  5 → 1, giving 5, 4 and 5 subdivisions);
- divisibility (`multiple_of`, `is_multiple_of`) and greatest common
  divisors (`common_step`) are integer arithmetic.

`select_level(raw, rule)` rounds a raw spacing (in world units) onto the
ladder. `"nearest"` compares in log space, so major lines end up between
`target/√2.5 ≈ 63` and `target·√2.5 ≈ 158` pixels apart; `"ceil"` rounds up,
so they are at least `target`. `level_boundary(level, rule)` is the raw
spacing (in `log10`) where one level gives way to the next.

## Choosing a spacing: `spacing.py`

The ideal major spacing is `raw = target_major_px / scale`. Two transition
modes decide what happens as the zoom crosses a level boundary:

- **`"fade"`** (default, stateless). `choose_faded` returns a
  `SpacingChoice(fine, coarse, blend)`. Within `±fade_band` (in `log10` of
  zoom) of a boundary both levels are drawn, the coarser weighted
  `smoothstep(...)`; elsewhere only one. Bands never overlap, because
  consecutive boundaries are at least `log10(2)` apart and the band is
  capped at `MAX_FADE_BAND`.
- **`"snap"`**. `SnapTracker` remembers the last level and only switches
  once the zoom has passed a boundary by `hysteresis`, so holding the zoom
  on a boundary never flickers. This is the only grid state kept between
  frames (`GridState` holds one tracker per axis).

## Linear axes: `ticks.py`

Candidate lines are the multiples `n × base` of a **base step**: the
largest step dividing every major and minor step of the active levels (with
two levels mid-fade, both must be expressible). `compute_axis_ticks`
returns an `AxisTicks`: the integer indices `n`, their screen positions, and
for each level a `LineClass` per line (`NONE`, `MINOR` or `MAJOR`), decided
by integer divisibility of `n`.

Positions come from `View.multiples_to_screen`, which works relative to the
multiple nearest the centre (`(n - n_c) × base - (c - n_c × base)`). Both
terms stay small, so lines stay evenly spaced even when zoomed in far from
the origin. A cap (`MAX_LINES_PER_AXIS`) protects against absurd settings:
minor lines are dropped first, then the axis.

## Painting lines: `gridlines.py`

Every axis, linear or log, describes its lines as `AxisLines`: a position
per line, its overall opacity (`total`) and how much of that is in the major
style (`major`, between 0 and `total`).

- For linear axes, `linear_axis_lines` sums the weights of the levels that
  draw each line (`level_visibility`); a line that is minor in one level and
  major in the other is "mixed". The line at zero is left out when the axes
  are shown, since the axis is drawn on top.
- `line_batches` groups lines with equal opacities into `LineBatch`
  primitives. Their **paint stack** is the minor colour at some opacity with
  the major colour composited over it, chosen so the result is exactly
  `total` opaque with `major` of it in the major colour: a linear blend
  between the two looks. Minor-style batches are ranked before major ones
  (painter's order).
- `axis_batches` adds the axis lines, unless the perpendicular axis is
  logarithmic (it has no zero).
- With `GridTheme.show_arrows`, `axis_arrows` adds an arrowhead at each
  drawn axis' positive end: a triangle `ARROW_LENGTH_PER_WIDTH × width +
  ARROW_LENGTH_BASE` long, with the proportions of plotted arrows (their
  geometry is in `core.plot`, which the grid may not import), centred on
  the axis line's actual pixels (`line_start + (width − 1) / 2`), its tip
  `ARROW_INSET_PX` inside the edge. The axis line then stops
  `ARROW_OVERLAP_PX` under the head, where the head is still wider than
  the line, so the line never shows beside the tip and there is no seam.
  The heads are `Fill`s (`GridScene.arrows`), one per axis, so neither is
  rasterised over the whole plane; the plane hands the renderer last
  frame's head when one has not moved, so panning along an axis does not
  rasterise its head again.

The renderer flattens each stack against the background into one solid
colour, so grid lines stay crisp 1-pixel rectangles.

## Number formatting: `formatting.py`

Labels are built from exact `Decimal` values: grid line `n` of step
`m × 10^k` is exactly `n·m × 10^k`. `format_value` drops trailing zeros and
never writes `-0`, and returns a `LabelText` (defined in `primitives.py`): a
tuple of `TextRun`s with roles `"text"`, `"sign"` (a leading minus, kept
separate so layout can hang it) and `"superscript"` (exponents, drawn
smaller and raised by the renderer, so no Unicode superscript glyphs are
needed).

`choose_notation(largest, resolution, fmt)` picks **one notation per
linear axis**: scientific only when the largest magnitude is outside the
`NumberFormat` range *and* scientific is no longer than fixed at the axis'
resolution. So an axis never mixes `500000` and `1×10⁶`, and near `10⁶`
with a fine spacing `1000000.25` beats `1.00000025×10⁶`.
`format_log_value` instead chooses per label (see [Log axes](#log-axes)).

## Which labels show: `ticklabels.py`

Every level labels its major lines; `label_candidates` decides which labels
are shown and how opaque each is, returning `LabelCandidate`s (position,
text, size, opacity) in position order. Zero is never a candidate: it gets
its own label.

- **Thinning.** When labels would be closer than the widest label plus
  `MIN_LABEL_GAP_PX`, only every 2nd, 10th, 20th, ... label is kept.
  `next_label_step` gives a *nested* sequence (each step a multiple of the
  previous), so thinning only ever removes labels. `thinning_opacity` fades
  the labels that the next thinning step would remove as the spacing shrinks
  from comfortable (`label_min_separation`) to the minimum, so nothing pops.
  `label_length` weights labels outside the plane down with their distance,
  so a wide label sliding in from the edge raises the requirement gradually.
- **Cross-fades.** During a fade, labels shared by both levels stay opaque.
  The others fade *sequentially* (`label_fade`): the outgoing level's labels
  are gone at the midpoint, before the incoming ones appear, so the two sets
  never crowd each other. If the two levels write numbers differently, their
  labels fade as separate sets.

## Placing labels: `layout.py`

- x labels sit just **below** the x axis, centred on their tick (with
  `hang_minus`, centred on the digits); y labels sit just **left** of the y
  axis, right-aligned, centred vertically.
- Near an edge, labels are **pushed** along so they stay inside the plane,
  and once the axis is off screen they stay **pinned** to that edge. Pushing
  is continuous, so labels never jump while panning; their halo keeps them
  readable while an axis line passes through them.
- Zero is labelled once, at the bottom-left of the origin, and fades out
  (over `ORIGIN_FADE_PX`) only as the last axis leaves the plane.
- **Collisions** between groups fade by distance, weighted by the stronger
  label's opacity (`fade_collisions`), with priority x labels > `0` > y
  labels: in a corner where both axes are pinned, y labels fade out of the
  way rather than overlapping.
- Within one axis, labels of cross-faded levels can meet. On a linear axis
  the more opaque wins (`resolve_overlaps`). On a log axis neighbouring
  labels can belong to different levels side by side, and a winner could
  flip between frames, so both fade continuously and symmetrically
  (`soften_overlaps`): each keeps `Π(1 − closeness × other.opacity)` of its
  opacity.

### Axis titles

`build_grid(..., x_title=, y_title=)` places the titles with the labels, at
the axes' positive ends on the side away from the tick labels, the
textbook convention: `place_x_title` right-aligns the x title at the right
edge, just above the x axis; `place_y_title` puts the y title at the top
edge, just right of the y axis. Both are pushed and pinned into the plane
like the labels, and keep `arrow_overhang(theme)` further from the axis
with arrowheads. A title's box counts the font's descent
(`TextSize.descent`, `title_box`), as titles are words and their letters
reach below the baseline; digits never do, so tick labels ignore it.

Titles have the highest priority in the collision fades (x title, then y
title, then x labels, `0`, y labels), so tick labels fade out where a
title comes onto their row or column, and the y title fades where it meets
the x title. To keep labels from fading for a title that is merely
*near* them, the titles keep `COLLISION_PADDING_PX + COLLISION_FADE_PX`
(10 px) from the tick labels' row and column: on screen that is exactly
the gap, axis and gap between them, and with the x axis below the plane
the x title rides that far above the row of pinned labels. Only when a
title is pinned onto the labels' row (the x axis above the plane) or
column (the y axis off to the side) do labels near it fade.

### Category labels

`build_grid(..., x_labels=, y_labels=)` takes named positions
(`categories.check_categories` turns a sequence or a mapping into sorted
`(position, name)` pairs) and replaces that axis' numeric label
candidates with `categories.category_candidates`, which the rest of the
pipeline places like any labels. Thinning keeps every `stride`-th name,
`stride` the smallest power of two that leaves the largest name
(`label_length`, weighted by distance from the plane like numbers) and
`MIN_LABEL_GAP_PX` between kept names, the gap measured as the smallest
between neighbouring names near the plane; the names the next doubling
would remove fade with `thinning_opacity`, as numbers do, so no name pops.
Strides count from the first category, so panning never changes which
names are kept. A categorical axis has no shared `0` at the origin
(`shared_zero=False`); a numeric, linear other axis gets a `0` candidate of
its own (`_with_categories`). Where the axes' labels meet, names win over
numbers: with only the y axis named, y labels take priority over x labels
in the collision fades (`names_first="y"`).

All positions are snapped with `core/pixels.py`, so a label sits a fixed
number of pixels from its line and never wobbles while zooming. The fades
above, however, measure distances between the labels' *exact* positions
(`PlacedLabel.offset`, `separation(..., exact=True)`): measured between
snapped positions, a fade over `OVERLAP_FADE_PX = 8` pixels would change in
steps of 1/8 whenever a label crossed a pixel boundary, which showed as a
faint flicker while zooming.

## Log axes

On a log axis equal distances are equal ratios, so its grid is not one
evenly spaced set of lines. `logticks.compute_log_ticks` builds three
tiers, each fading in as zooming makes room, and returns them as
`LogTicks` (lines plus the tier data, which `loglabels` labels):

1. **Decades** (`…, 0.1, 1, 10, …`) are evenly spaced in the exponent, so
   the linear machinery is reused on the exponent: `decade_choice` limits
   the spacing to whole decades, `decade_levels` gives minor steps of at
   least one decade, and `ticks_for_levels` classifies the lines. Every 2nd,
   5th, 10th, ... decade when zoomed out, cross-faded like a linear axis.
2. **Log-paper lines** `2..9 × 10ⁿ` (`IntegerTier`), only while every
   decade is a major line. They fade in together as minor lines once the
   most crowded pair (9 and 10, `log10(10/9)` decades apart) is
   `PAPER_DENSITY × target` pixels apart, then each becomes a major line on
   its own once the gap to its nearest neighbour (a known constant in
   decades) passes `major_threshold_px`: 2 and 5 first, then 3, 4, 6, ...
3. **Subdivisions** (`FineTier`, one per decade on screen), gated by the
   log-paper lines. Over short distances a log axis is nearly linear, with
   `scale / (m ln 10)` pixels per unit of the mantissa `m`. Every
   subdivision takes the 1-2-5 level a linear axis with that *local* scale
   would choose (`_local_levels`, a vectorised `choose_faded`), cross-faded
   near switching points. Every such step divides 1, so subdivisions line up
   with the integers; and the local scale is continuous, so the spacing
   never jumps at an integer or a power of ten. Candidates are multiples of
   the finest step needed in the *visible* part of each decade, which keeps
   the work proportional to what is on screen.

Each line's opacities depend only on its value and the zoom, never on where
it is on screen, so panning is rigid; and they are continuous in the zoom
(smoothstep ramps over `±fade_band`), so nothing pops. In `"snap"` mode the
ramps become hard switches.

`loglabels.log_label_candidates` labels the tiers: decades with the linear
thinning on the exponents; log-paper lines once they are major and their
gap fits the widest label; subdivisions with nested thinning judged by the
scale *at each label's own position* (`_fine_label_opacity`), since the
density changes along the axis. Labels choose their notation one by one
(`format_log_value`, with `10ⁿ` for powers of ten), so a label's text
never changes while panning.

## Invariants and where they are tested

| Invariant | Tests |
|---|---|
| Lines stay in the plane; opacities in `(0, 1]` | `test_lines_stay_inside_the_plane`, `test_lines_stay_inside_and_opacities_are_valid` |
| Major spacing stays near the target | `test_major_spacing_stays_near_the_target` |
| Labels never overlap (overlapping opacities sum to at most 1) | `test_labels_never_overlap`, `test_log_labels_never_overlap`, `test_titles_never_overlap_labels_and_stay_inside` |
| Titles move rigidly with their axes | `test_titles_move_rigidly_with_their_axes` |
| Category names never overlap other labels or titles | `test_names_never_overlap` (`test_categories.py`) |
| Panning moves every line rigidly | `test_panning_by_whole_pixels_moves_every_line_rigidly`, `test_panning_a_log_view_moves_every_line_rigidly` |
| Zooming never pops lines or labels | `test_zooming_never_pops_labels`, `test_zooming_never_pops_log_lines`, `test_zooming_never_pops_log_labels` |
| One notation per linear axis | `test_labels_on_one_axis_never_mix_notations` |
| Grid pixels are exactly theme colours (no blur) | `test_grid_lines_use_only_theme_colours` |

They live in `tests/core/grid/test_scene.py`, `test_logticks.py` and
`tests/test_renderer.py`; see [Testing](testing.md#property-tests).
