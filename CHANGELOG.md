# Changelog

Notable changes to PyGridPlot. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/): before 1.0, a minor
version (0.2, 0.3, ...) may change the API, and this file says how.

## [Unreleased]

The first release, to be 0.1.0: an interactive, zoomable Cartesian plane
for pygame-ce, drawn from your own main loop.

### Added

- **The plane**, `pgp.Grid`: an adaptive grid whose spacings are exact
  1-2-5 steps that cross-fade while you zoom, with exact decimal labels
  that never overlap and stay on screen when their axis leaves it; linear
  or logarithmic axes; views set up from a centre and scale or from ranges
  (`pgp.data_range` gives one that shows data); smooth, eased panning and
  zooming with the mouse, the wheel and the keyboard; a plane can fill the
  window or a rectangle of it, and several can share a window. Axis titles,
  arrowheads on the axes, and names in place of numbers along an axis.
  Light and dark themes, reusable styles and a colour palette.
- **Functions** with `plot`: NumPy or `math` functions that may raise,
  return NaN or infinity, or take only single numbers; adaptive sampling;
  poles and jumps never joined up, and curves that end exactly where they
  stop being defined.
- **Calculus**, as functions to plot or evaluate: exact derivatives of any
  order (`pgp.derivative`, `pgp.derivatives`, by Taylor-mode automatic
  differentiation), definite integrals and antiderivatives
  (`pgp.integral`, `pgp.antiderivative`, by adaptive Gauss-Kronrod
  quadrature, NaN rather than a wrong number when they diverge), Taylor
  polynomials (`pgp.taylor`) and Fourier series (`pgp.fourier`).
- **Curves**: parametric (`parametric`), polar (`polar`) and implicit
  (`implicit`, `F(x, y) = c`, with several levels for contour maps).
- **Regions and fields**: the area between two graphs (`fill_between`),
  regions where conditions hold (`inequality`, combined with `&`, `|` and
  `~`), and vector and slope fields (`vector_field`, `slope_field`) on
  lattices that pan rigidly and fade arrows in as you zoom.
- **Shapes and text**: points with five marker shapes (`points`),
  polylines (`polyline`), polygons with fill rules (`polygon`), arrows
  (`arrow`), the terms of sequences (`sequence`), text at points of the
  plane (`text`) and a legend of the labelled plots (`label=` on every
  plotting method, and `legend`).
- **Data**: scatter plots with a colour and a size per point, coloured by
  value with `pgp.color_map` and a `color_bar`; error bars
  (`points(x_error=, y_error=)`); step plots (`polyline(step=)`); bar
  charts, stacked, floating or horizontal (`bars`); histograms in anchored
  bins, as counts, densities or cumulative (`histogram`); and box and
  violin plots of groups of samples (`box_plot`, `violin_plot`).
- **Interaction**: points the user drags, free, locked to an axis or kept
  on a curve (`draggable`), and sliders, stepped, integer or geometric, in
  panels in the corners (`slider`), drawn with `draw_controls`.
- **Speed**: plots that have not changed since the previous frame are
  reused automatically and exactly, and so are their rasterised pixels.
- **Quick mode**: `pgp.show(f, g, ...)` plots functions in a window of its
  own, without a loop.
- **Documentation**: a user guide with a picture of every example, an API
  reference generated from the code, and a developer guide; 21 short
  example programs.
- Type annotations (`py.typed`), checked with strict mypy; an
  `ImportError` that says how to fix it when the original `pygame` is
  installed instead of pygame-ce.

[Unreleased]: https://github.com/GWStuart/PyGridPlot/commits/HEAD
