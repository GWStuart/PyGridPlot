# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/) once 1.0 is released.

## [Unreleased]

### Added

- **Parametric and polar curves**: `Plane.parametric(surface, x, y,
  t_range)` and `Plane.polar(surface, r, theta_range)`, sampled adaptively
  on screen, with exact domain edges and broken at jumps and poles; tuned
  with `CurveSampling`.
- **Logarithmic axes**: `Plane(log_x=True, log_y=True)` and the assignable
  `plane.log_x` / `plane.log_y`. The grid shows thinned decades, log paper,
  then finer subdivisions as you zoom, all cross-faded; labels choose their
  notation per value.
- **View from ranges**: `Plane(x_range=..., y_range=...)`, fitted once the
  plane knows its size; equal aspect is decided automatically.
- `show()` accepts the same view options as `Plane`.
- Documentation split into a user guide and a developer guide with topic
  pages, recipes, and an API reference generated from the docstrings
  (`scripts/build_api_docs.py`), with tests that run every documented
  example and check every link.
- `scripts/benchmark.py` for per-frame timings; `examples/curves.py` and
  `examples/log_axes.py`; a headless smoke test for every example.
- `py.typed`, so type checkers use the package's annotations.

### Changed

- Strokes are rasterised 1.5 to 3 times faster (two-level candidate
  generation, duplicate merging with `np.maximum.at`), and blended up to
  six times faster on 32-bit surfaces; large fills are about 1.4 times
  faster.
- `core` is organised into foundation modules plus the `core.grid` and
  `core.plot` subpackages, with the layering enforced by tests.

## [0.1.0.dev0] - 2026-11-20

### Added

- The interactive plane: adaptive 1-2-5 grid with cross-fades, exact
  labels, push-and-pin label layout, anchored dragging and eased zooming.
- Plotting of functions, points, polylines and polygons, with robust
  evaluation, adaptive sampling and antialiased rendering.
- Light and dark themes, styles and the colour palette, `show()`.
