"""Tests that keep the documentation honest.

* The API reference must match what ``scripts/build_api_docs.py`` generates
  from the docstrings.
* Every runnable example in the Markdown documentation must run. Runnable
  examples are fenced as ```` ```python ````; signature listings and other
  fragments are fenced as ```` ```py ```` and are not run. The examples of one
  page run in order in a shared namespace, like a doctest session, with
  ``np``, ``math``, ``pygame``, ``pgp``, a display ``screen`` (also called
  ``surface``) and a ``grid`` already defined. Main loops end after a few
  frames: the event queue is replaced by one that quits.
* Every relative link and in-page anchor must resolve (in ``AGENTS.md`` too,
  a context file for AI agents kept outside the repository, when present,
  so that it cannot point at moved pages).
"""

from __future__ import annotations

import itertools
import math
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest

import pygridplot as pgp

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
PAGES = sorted(
    [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    + ([AGENTS] if AGENTS.exists() else [])
)
RUNNABLE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def page_id(path: Path) -> str:
    """Return a short test id for a page."""
    return str(path.relative_to(ROOT))


def test_api_reference_is_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    script = ROOT / "scripts" / "build_api_docs.py"
    monkeypatch.setattr(sys, "argv", ["build_api_docs", "--check"])
    runpy.run_path(str(script), run_name="__main__")  # exits if out of date


@pytest.mark.usefixtures("pygame_display")
@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_examples_run(
    page: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocks = RUNNABLE.findall(page.read_text(encoding="utf-8"))
    if not blocks:
        pytest.skip("no runnable examples")
    monkeypatch.chdir(tmp_path)  # examples may save files (screenshots)
    frames = itertools.count()
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda *a, **k: (
            [pygame.event.Event(pygame.QUIT)] if next(frames) % 3 == 2 else []
        ),
    )
    screen = pygame.display.set_mode((320, 240))
    namespace: dict[str, object] = {
        "np": np,
        "math": math,
        "pygame": pygame,
        "pgp": pgp,
        "screen": screen,
        "surface": screen,
        "grid": pgp.Grid(),
    }
    for number, code in enumerate(blocks, 1):
        try:
            exec(compile(code, f"{page_id(page)} example {number}", "exec"), namespace)
        except Exception as error:  # pragma: no cover - the message is the point
            pytest.fail(
                f"example {number} of {page_id(page)} failed: {error!r}\n{code}"
            )
        if not pygame.display.get_init():  # an example ended pygame: restart it
            pygame.init()
            namespace["screen"] = namespace["surface"] = pygame.display.set_mode(
                (320, 240)
            )


def anchors(text: str) -> set[str]:
    """Return the GitHub-style anchors of a page's headings."""
    result = set()
    for heading in re.findall(r"^#+ (.+)$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).replace(
            " ", "-"
        )
        result.add(slug)
    return result


@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_links_resolve(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)  # skip code
    for target, fragment in re.findall(r"\]\(([^)#\s]*)(?:#([^)\s]*))?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        destination = (page.parent / target).resolve() if target else page
        assert destination.exists(), f"{page_id(page)} links to missing {target}"
        if fragment and destination.suffix == ".md":
            assert fragment in anchors(destination.read_text(encoding="utf-8")), (
                f"{page_id(page)} links to missing anchor {target}#{fragment}"
            )
