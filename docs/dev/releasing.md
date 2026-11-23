# Releasing

How PyGridPlot is checked on every change, how it is packaged, and the
steps to publish a version on [PyPI](https://pypi.org/).

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it checks |
|---|---|
| Tests | The whole suite on Linux with every supported Python (3.10 to 3.15), and on Windows and macOS with one. |
| Oldest dependencies | The suite with the lowest versions `pyproject.toml` allows (NumPy 1.26, contourpy 1.2, pygame-ce 2.5.7) on Python 3.10, so the lower bounds stay true (pygame-ce before 2.5.7 has no `Font.outline`, which draws the labels' halos). |
| Lint | `ruff check`, `ruff format --check` and strict `mypy` on `src/`. |
| Package | Builds the source archive and the wheel, checks their metadata and README with `twine check --strict`, installs the wheel alone and draws a frame with it, and runs the tests from the source archive (so it holds everything they need). |

The tests choose SDL's dummy drivers themselves, so no display is needed.

## Packaging

The project is built with [hatchling](https://hatch.pypa.io/):

- **The version** has one source, `__version__` in
  `src/pygridplot/__init__.py`; `pyproject.toml` reads it.
- **The wheel** holds the package, its font and the font's licence, and
  `py.typed`.
- **The source archive** holds what is needed to build, test and read the
  project (`src`, `tests`, `docs`, `examples`, `scripts`, the README,
  changelog and licence), and nothing else.
- **The README on PyPI**: its links are relative, so they work on GitHub
  and in a checkout (and the docs test checks them).
  [hatch-fancy-pypi-readme](https://github.com/hynek/hatch-fancy-pypi-readme)
  rewrites them to absolute links into the GitHub repository when the
  package is built. They point at the default branch (`HEAD`), so they
  always show the current docs.
- **The licence** is `LGPL-2.1-or-later` (an SPDX expression, PEP 639),
  with `LICENSE` and the font's licence listed as licence files.

The repository's address appears in `pyproject.toml` twice: in
`[project.urls]` and in the README link substitutions. Change both if it
moves.

## Publishing a version

Once, before the first release:

1. Push the repository to `https://github.com/GWStuart/PyGridPlot` (or
   change the address as above).
2. Check that the name is free: `https://pypi.org/project/pygridplot/`.
3. On PyPI, add a *pending trusted publisher* (Your account → Publishing):
   the GitHub owner and repository, workflow `release.yml`, environment
   `pypi`. No token is stored anywhere. On GitHub, create the environment
   `pypi` (Settings → Environments); it can require your approval before
   each upload.
4. Optionally, try the whole process on [TestPyPI](https://test.pypi.org/)
   first.

For each release:

1. Make sure CI passes on the default branch.
2. Set `__version__` in `src/pygridplot/__init__.py` (`"0.1.0"`), and in
   `CHANGELOG.md` turn *Unreleased* into the version and date
   (`## 0.1.0 (2026-12-01)`).
3. Build and look at it locally:

   ```bash
   pip install build twine
   python -m build                     # dist/*.tar.gz and dist/*.whl
   twine check --strict dist/*
   ```

4. Commit, tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
5. Publish a GitHub release from the tag. `.github/workflows/release.yml`
   checks that the tag matches the version, builds, checks and uploads to
   PyPI.
6. Afterwards, set the next development version (`"0.2.0.dev0"`) and start
   a new *Unreleased* section in the changelog.

Versions follow [semantic versioning](https://semver.org/) as it applies
before 1.0: a minor version (0.2.0) may change the API, a patch (0.1.1)
only fixes bugs.

## Supported versions

Python 3.10 and newer, tested up to 3.15. Python 3.10 reached its end of
life in October 2026, and new NumPy and contourpy releases need 3.11 (pip
picks older ones on 3.10); dropping it would allow `typing.Self` and
exception groups. The dependencies' lower bounds are tested by the
*oldest dependencies* job.
