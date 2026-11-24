"""Tests for :class:`pygridplot.core.handles.checkbox.Checkbox`."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.handles.checkbox import Checkbox


def test_values_are_bools_starting_unticked() -> None:
    box = Checkbox("show")
    assert box.value is False
    box.value = True
    assert box.value is True
    assert Checkbox("x", value=np.bool_(True)).value is True
    for bad in (1, 0, "yes", None, 0.0):
        with pytest.raises(TypeError, match="True or False"):
            Checkbox("x", value=bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="True or False"):
            box.value = bad  # type: ignore[assignment]


def test_settings_are_validated() -> None:
    with pytest.raises(TypeError, match="label"):
        Checkbox(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="corner"):
        Checkbox("x", corner="middle")  # type: ignore[arg-type]
    box = Checkbox("x", corner="top-right", color="red", enabled=False)
    assert (box.corner, box.color, box.enabled) == ("top-right", "red", False)
    with pytest.raises(ValueError, match="corner"):
        box.corner = "left"  # type: ignore[assignment]


def test_changed_is_stable_through_a_frame() -> None:
    box = Checkbox("x")
    box.begin_frame()
    assert not box.changed
    box.value = True
    assert not box.changed  # until the next frame starts
    box.begin_frame()
    assert box.changed
    box.begin_frame()
    assert not box.changed
    box.value = False
    box.value = True  # and back: no change by the next frame's start
    box.begin_frame()
    assert box.changed  # it did change in between, as a slider moved and back
    box.value = True  # the same value
    box.begin_frame()
    assert not box.changed


def test_hovering_and_removing() -> None:
    removed: list[Checkbox] = []
    box = Checkbox("x", on_remove=removed.append)
    box.set_interaction(hovered=True)
    assert box.hovered
    box.remove()
    box.remove()
    assert removed == [box]  # called back once
    assert not box.hovered
    assert repr(box) == "Checkbox('x', False)"
