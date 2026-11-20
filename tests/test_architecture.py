"""Architecture rules and smoke tests for the examples and scripts.

The core must stay pure (stdlib + NumPy + itself). That is what lets it be
tested without a display and reused by future backends, so it is enforced
here by inspecting every import statement in ``pygridplot/core``.
"""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "pygridplot" / "core"
ALLOWED_THIRD_PARTY = {"numpy"}


def imported_modules(path: Path) -> set[str]:
    """Return the top-level names of absolute imports in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_modules_only_import_stdlib_numpy_and_core(path: Path) -> None:
    for name in imported_modules(path):
        top = name.split(".")[0]
        if top == "pygridplot":
            assert name.startswith("pygridplot.core"), f"{path.name} imports {name}"
        else:
            allowed = top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY
            assert allowed, f"{path.name} imports {name}"


def test_examples_have_a_main_function() -> None:
    for path in sorted((ROOT / "examples").glob("*.py")):
        namespace = runpy.run_path(str(path), run_name="not_main")
        assert callable(namespace.get("main")), path.name


def test_zoom_sweep_runs_headless(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "examples" / "zoom_sweep.py"), run_name="x")
    namespace["main"](["--save-frames", str(tmp_path), "--count", "3", "--every", "5"])
    assert len(list(tmp_path.glob("frame_*.png"))) == 3


def test_render_gallery_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = ROOT / "scripts" / "render_gallery.py"
    monkeypatch.setattr(
        sys, "argv", ["render_gallery", "--out", str(tmp_path), "--only", "01_default"]
    )
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "01_default.png").exists()
    assert (tmp_path / "_sheet.png").exists()
