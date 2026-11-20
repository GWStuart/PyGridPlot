# Rendering

The core describes what to draw as primitives in continuous screen pixels;
a **renderer** turns them into pixels. `render/` holds the pygame backend.
Everything except `pygame_renderer.py`, `composite.py`, `markers.py`,
`text.py` and `fonts.py` is pure NumPy and tested without a display.

## The renderer protocol: `base.py`

`Renderer` lists what any backend must provide:

| Method | Draws |
|---|---|
| `measure_text(text, font)` | Nothing: reports a label's `TextSize` (core needs it for layout). |
| `draw_grid(surface, scene)` | A `GridScene`: background, line batches, labels. |
| `draw_stroke(surface, stroke)` | Polylines with a round pen. |
| `draw_fill(surface, fill)` | Filled polygons with a fill rule. |
| `draw_markers(surface, markers)` | Point markers. |

`PygameRenderer` implements these, plus `clipped(surface, rect)` (a context
manager that clips drawing to the plane and restores the previous clip),
`draw_line_batch`, `draw_text` and `clear_caches`. `Plane` uses `clipped`
and pygame events, so `Plane` is pygame-specific even though everything it
draws goes through the protocol; see
[A new backend](extending.md#a-new-backend).

## Grid lines and labels

- **Line batches** are drawn as 1-pixel-wide `surface.fill` rectangles
  starting at `pixels.line_start(position, width)`, so lines are crisp at
  any width. The batch's paint stack is flattened against the theme
  background into one solid colour (`colors.ColorResolver.flatten`,
  cached); with no background, it is blended onto what is there.
- **Labels** are typeset by `text.py` run by run: normal runs on a shared
  baseline, superscripts smaller and raised. Measurements are relative to
  the label's **digit box** (top of the digits to the baseline), which is
  what layout aligns, so labels line up with their lines whatever
  characters they contain. Finished labels, halo included, are cached as
  single surfaces (an LRU), so drawing a label costs one blit; label text
  depends only on its value, so cache hits are the norm while panning.
- **Fonts** come from `fonts.FontLibrary`: the bundled DejaVu Sans by default
  (it has the true minus sign and the multiplication sign), a font file, or
  a system font by name, with a warning and a fallback if a font cannot be
  found.

## Rasterising: `raster.py`

pygame cannot draw thick antialiased curves with clean joins, or
antialiased filled polygons, so the backend rasterises them itself into
**coverage**: how much of each pixel a shape covers, from 0 to 1.

### Strokes

A stroke is the union of **capsules** (a segment with round ends) of radius
`width / 2 + 0.5`, combined with `max`, so joins and ends are round and a
curve split into thousands of tiny segments looks exactly like one smooth
line, with no dark spots where segments overlap. Lines thinner than a pixel
are drawn one pixel wide with lower coverage.

1. **Split long segments** (`_split_long`) into pieces with tight bounding
   boxes: a diagonal piece's box grows with the square of its length, so
   diagonal pieces are about two radii long, while nearly horizontal or
   vertical pieces can be up to 64 pixels.
2. **Distances** (`_segment_distances`). The candidates are the pixels of
   each piece's bounding box grown by the radius. They are generated in two
   levels: one entry per (piece, row) holding everything constant along the
   row, then one per pixel, where the projection parameter onto the segment
   changes by a constant step along the row. That avoids integer division
   and per-pixel gathers of segment data. Arithmetic is in `float32`
   (coordinates are clipped to the screen, so it is accurate to about
   1/10000 px).
3. **Merge duplicates** (`_unique_max`). Neighbouring pieces share many
   candidates (a dense tangle can list a pixel dozens of times), so each
   pixel keeps its largest coverage. Normally that is `np.maximum.at` into a
   buffer the size of the candidates' bounding box, which is linear in the
   number of candidates; when the box is much larger than the candidate
   count (`_BUFFER_FACTOR`, a long thin line across the screen), sorting the
   covered candidates is cheaper.

### Fills

`fill_coverage` samples each pixel row with `FILL_SUBSAMPLES` (5)
sub-scanlines. On each, the spans inside the polygon are found exactly
(edge crossings sorted along the line, with a running winding number for
`"nonzero"` or its parity for `"evenodd"`), and each pixel gets the
fraction of its width covered. Those fractions are accumulated with a
cumulative-sum trick (four point weights per span, one `cumsum` per row) in
`float32`, and round-off within `1e-5` of 0 or 1 is snapped, so holes are
exactly empty and interiors exactly solid.

### Markers

`marker_coverage` draws a marker as a signed-distance stamp (circle,
square, diamond, cross, plus; non-circular shapes have the circle's area)
with an optional outline ring *outside* the body. `markers.MarkerPainter`
caches a stamp per style for each quarter-pixel phase (`PHASES = 4`), so
markers glide smoothly, and blits them all with `fblits`.

## Compositing: `composite.py`

Coverage is painted with standard "over" blending:

- **Sparse coverage** (strokes) is blended pixel by pixel through
  `pygame.surfarray`. On the usual 32-bit surfaces (the display, default
  `pygame.Surface`s) whole packed pixels are read and written at once
  (`_over_packed`: one gather and one scatter instead of three strided ones,
  in `float32`), which is several times faster than per-channel access.
  Other formats use per-channel access, or for very large strokes a
  translucent patch blitted by pygame.
- **Dense coverage** (fills) becomes a translucent patch with the coverage
  in its alpha channel, blitted so pygame blends in C.
- **Surfaces with per-pixel alpha** get the full "over" formula, so
  antialiased edges never leave a dark fringe when a transparent layer is
  composited later.

Every function respects the surface's clip area.

## Performance

`scripts/benchmark.py` times the main cases. Typical numbers on a 1100×720
surface (median of 20 runs; your machine will differ):

| Case | Time |
|---|---|
| Linear grid | ~0.9 ms |
| Log-log grid | ~1.8–2.9 ms |
| Function test set (6 functions, sampling and drawing) | ~9 ms |
| Parametric circle | ~1 ms |
| Spirograph (~10,000 px of line) | ~4 ms |
| Big translucent polygon fill | ~4.5 ms |
| `sin(1/x)` zoomed in (the pathological case) | ~36 ms |

Drawing cost grows with the length of line on screen (roughly half a
millisecond per thousand pixels of a 2.5 px line), and fills with their
area. Two optimisations were tried and rejected, so they need not be tried
again:

- **Simplifying polylines before rasterising** (dropping vertices within
  0.02 px of a straight line) made strokes with long straight runs up to
  2.6× faster to rasterise, but its overhead made typical frames slower, and
  float noise near the threshold changed which vertices were dropped after a
  pan, breaking rigid panning.
- **A full-screen max buffer for every stroke** costs about a millisecond
  even for tiny strokes; the box-sized buffer with a sort fallback avoids
  that.
