"""Tests for the package as installed: its version and its pygame check."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pygridplot as pgp

SRC = Path(__file__).resolve().parents[1] / "src"


def test_the_version_is_a_release_or_development_version() -> None:
    # PEP 440: 0.1.0, 0.1.0.dev0, 0.2.0rc1, ...
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", pgp.__version__)


def test_the_original_pygame_is_refused_with_advice(tmp_path: Path) -> None:
    # A stand-in for the original pygame: no IS_CE marker.
    fake = tmp_path / "pygame"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        textwrap.dedent(
            """
            class version:
                ver = "2.6.1"
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(SRC)])}
    result = subprocess.run(
        [sys.executable, "-c", "import pygridplot"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "needs pygame-ce" in result.stderr
    assert "original pygame 2.6.1" in result.stderr
    assert "pip uninstall pygame" in result.stderr
