# Developer guide

How PyGridPlot is put together, so that you can find your way around the
code and extend it. For how to *use* the library, see the
[user guide](../user/index.md).

1. [Architecture](architecture.md): the design principles, the layers and
   the rule that keeps them apart, a map of every module, and what happens
   during a frame.
2. [The grid](grid.md): choosing spacings, generating and painting lines,
   labels and their layout, and logarithmic axes.
3. [Plotting](plotting.md): evaluating user functions robustly, the function
   and curve samplers, finding implicit curves, and building primitives.
4. [Rendering](rendering.md): the renderer protocol, rasterising strokes,
   fills and markers, compositing, text, and where the time goes.
5. [Testing and tools](testing.md): the test suite, property tests, the
   documentation checks, and the visual and benchmark tools.
6. [Extending PyGridPlot](extending.md): recipes for new kinds of plots,
   axes, markers, theme options and backends, and the coding conventions.

Every module, class and function also has a docstring explaining what it
does and why, so the code itself is the detailed reference; these pages
give the overview and the reasoning that spans modules.

## Quick start for contributors

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                                   # ~570 tests, about 20 s, headless
ruff check . && ruff format --check .    # lint and formatting
mypy                                     # strict type checking of src/
python scripts/build_api_docs.py         # after changing a public docstring
```

All four checks must pass before a change is done; [Testing and
tools](testing.md#the-checklist) has the full checklist.
