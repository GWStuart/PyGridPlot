"""End-to-end tests of saving pictures (:meth:`Grid.save`) and recording videos.

Videos need ffmpeg, which is not a dependency: those tests are skipped
where it cannot be found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pytest

import pygridplot as pgp
from pygridplot import recording

pytestmark = pytest.mark.usefixtures("pygame_display")

SIZE = (320, 240)


def ffmpeg_or_skip() -> str:
    """The ffmpeg program, or skip the test."""
    try:
        return pgp.find_ffmpeg()
    except RuntimeError:
        pytest.skip("ffmpeg is not installed")


def scene(grid: pgp.Grid, surface: pygame.Surface) -> None:
    """Draw a frame with every kind of thing: plots, text, a legend, controls."""
    grid.draw(surface, x_title="x")
    grid.fill_between(surface, np.sin, 0, label="area")
    grid.plot(surface, np.cos, dash="dashed", label="cos x")
    grid.points(surface, [(1, 1), (-1, 0.5)])
    grid.vector_field(surface, lambda x, y: (-y, x), spacing=50)
    grid.text(surface, "label", (1, -1))
    grid.legend(surface)
    grid.color_bar(surface, "viridis", (0, 1), corner="top-left")
    grid.draw_controls(surface)


def make(**kwargs: object) -> tuple[pgp.Grid, pygame.Surface]:
    """A plane on a surface of its own, with a slider and a checkbox."""
    surface = pygame.Surface(SIZE)
    kwargs.setdefault("x_range", (-4, 4))
    grid = pgp.Grid(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]
    grid.slider("a", (0, 1), value=0.5, play="loop")
    grid.checkbox("show", value=True)
    return grid, surface


def pixels(path: Path) -> np.ndarray:
    """A picture file's colours, ``(width, height, 3)``."""
    return pygame.surfarray.array3d(pygame.image.load(str(path))).astype(int)


# --- Pictures -----------------------------------------------------------------


def test_a_picture_at_scale_1_is_the_screen_exactly(tmp_path: Path) -> None:
    grid, surface = make()
    scene(grid, surface)
    grid.save(tmp_path / "plane.png")
    assert np.array_equal(
        pixels(tmp_path / "plane.png"), pygame.surfarray.array3d(surface)
    )


def test_a_plane_inside_a_window_is_saved_alone(tmp_path: Path) -> None:
    window = pygame.Surface((400, 300))
    window.fill((255, 0, 0))
    grid = pgp.Grid(pygame.Rect(30, 20, *SIZE), x_range=(-4, 4))
    scene(grid, window)
    grid.save(tmp_path / "plane.png")
    saved = pixels(tmp_path / "plane.png")
    assert saved.shape[:2] == SIZE
    shown = pygame.surfarray.array3d(window)[30:350, 20:260].astype(int)
    # Moving coordinates by whole pixels can round their last bits.
    assert np.abs(saved - shown).max() <= 1


def test_pictures_scale_with_everything_in_proportion(tmp_path: Path) -> None:
    grid, surface = make()
    scene(grid, surface)
    grid.save(tmp_path / "big.png", scale=2)
    grid.save(tmp_path / "small.jpg", scale=0.5)
    big = pygame.image.load(str(tmp_path / "big.png"))
    assert big.get_size() == (640, 480)
    assert pygame.image.load(str(tmp_path / "small.jpg")).get_size() == (160, 120)
    # Shrunk back, the big picture is the screen, but for the antialiasing.
    shrunk = pygame.surfarray.array3d(pygame.transform.smoothscale(big, SIZE))
    shown = pygame.surfarray.array3d(surface)
    assert np.mean(np.abs(shrunk.astype(int) - shown.astype(int))) < 4


def test_planes_without_a_background_save_transparent_pngs(tmp_path: Path) -> None:
    clear = replace(pgp.LIGHT, background=None)
    grid, surface = make(theme=clear)
    grid.draw(surface)
    grid.plot(surface, np.sin)
    grid.save(tmp_path / "clear.png")
    grid.save(tmp_path / "flat.jpg")
    alpha = pygame.surfarray.array_alpha(pygame.image.load(str(tmp_path / "clear.png")))
    assert alpha.min() == 0  # transparent between the lines
    assert alpha.max() == 255  # opaque on them
    assert tmp_path.joinpath("flat.jpg").stat().st_size > 0


def test_saving_needs_a_drawn_frame_and_a_picture_type(tmp_path: Path) -> None:
    grid, surface = make()
    with pytest.raises(ValueError, match="nothing was drawn"):
        grid.save(tmp_path / "early.png")
    grid.draw(surface)
    with pytest.raises(ValueError, match=r"\.png"):
        grid.save(tmp_path / "plane.svg")
    with pytest.raises(ValueError, match="scale"):
        grid.save(tmp_path / "plane.png", scale=0)
    with pytest.raises(ValueError, match="too large"):
        grid.save(tmp_path / "plane.png", scale=100)
    grid.save(tmp_path / "new" / "folder" / "plane.png")  # folders are made
    assert (tmp_path / "new" / "folder" / "plane.png").exists()


def test_saving_leaves_the_screen_as_it_was(tmp_path: Path) -> None:
    grid, surface = make()
    scene(grid, surface)
    before = pygame.surfarray.array3d(surface).copy()
    grid.save(tmp_path / "a.png", scale=3)
    other = pygame.Surface(SIZE)
    scene(grid, other)  # the next frame draws the same, from the screen's caches
    assert np.array_equal(pygame.surfarray.array3d(other), before)


def test_unchanged_primitives_are_scaled_once() -> None:
    grid, surface = make()
    grid.draw(surface)
    grid.plot(surface, np.sin)
    grid._snapshot(2.0, opaque=True)
    replayer = grid._replayer
    assert replayer is not None
    first = {key: copy for key, (_, copy) in replayer._previous.items()}
    grid.draw(surface)
    grid.plot(surface, np.sin)  # reused by the plot cache: the same objects
    grid._snapshot(2.0, opaque=True)
    again = {key: copy for key, (_, copy) in replayer._previous.items()}
    shared = [key for key in first if key in again and again[key] is first[key]]
    assert shared  # the curve's scaled stroke, at least, is the same object


# --- Numbered pictures --------------------------------------------------------


def test_recording_pictures_numbers_them(tmp_path: Path) -> None:
    grid, surface = make()
    with pgp.Recorder(tmp_path / "frames" / "wave.png", fps=10) as video:
        for _ in range(3):
            scene(grid, surface)
            video.add(grid)
        assert video.frames == 3
        assert video.time == pytest.approx(0.3)
    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert names == ["wave_00000.png", "wave_00001.png", "wave_00002.png"]
    assert video.closed


def test_a_recording_keeps_the_videos_time(tmp_path: Path) -> None:
    grid, surface = make()
    t = grid.slider("t", (0, 10), play="loop", duration=10)
    t.playing = True
    values = []
    with pgp.Recorder(tmp_path / "t.png", fps=25, scale=0.5) as video:
        for _ in range(5):
            grid.draw(surface)
            grid.draw_controls(surface)
            values.append(t.value)
            video.add(grid)
    # Each frame is 1/25 s of video, however long it took to draw.
    assert np.diff(values) == pytest.approx([0.04] * 4)
    assert grid._play_clock is None  # pygame's clock again


def test_surfaces_are_recorded_as_they_are_and_sizes_kept(tmp_path: Path) -> None:
    window = pygame.Surface((100, 80))
    with pgp.Recorder(tmp_path / "w.png", scale=0.5) as video:
        window.fill((10, 200, 30))
        video.add(window)
        video.add(pygame.Surface((300, 50)))  # resized to the first frame's size
    sizes = {
        pygame.image.load(str(path)).get_size() for path in tmp_path.glob("w_*.png")
    }
    assert sizes == {(50, 40)}
    first = pixels(tmp_path / "w_00000.png")
    assert tuple(first[25, 20]) == (10, 200, 30)


def test_recorders_check_what_they_are_given(tmp_path: Path) -> None:
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="fps"):
            pgp.Recorder(tmp_path / "a.png", fps=bad)
    with pytest.raises(TypeError, match="fps"):
        pgp.Recorder(tmp_path / "a.png", fps="60")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale"):
        pgp.Recorder(tmp_path / "a.png", scale=-2)
    video = pgp.Recorder(tmp_path / "a.png")
    with pytest.raises(TypeError, match=r"Grid or a pygame\.Surface"):
        video.add("frame")  # type: ignore[arg-type]
    grid, _ = make()
    with pytest.raises(ValueError, match="nothing was drawn"):
        video.add(grid)
    video.close()
    video.close()  # twice is fine
    with pytest.raises(ValueError, match="closed"):
        video.add(pygame.Surface((4, 4)))
    assert "fps=60.0" in repr(video)


# --- Videos -------------------------------------------------------------------


def frame_count(path: Path) -> int:
    """The number of frames in a video, by ffprobe (next to ffmpeg)."""
    program = shutil.which("ffprobe")
    if program is None:
        pytest.skip("ffprobe is not installed")
    result = subprocess.run(
        [
            program,
            *("-v", "error", "-count_frames", "-select_streams", "v:0"),
            *("-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


@pytest.mark.parametrize("name", ["wave.mp4", "wave.webm", "wave.gif"])
def test_videos_are_written_by_ffmpeg(tmp_path: Path, name: str) -> None:
    ffmpeg_or_skip()
    grid = pgp.Grid(pygame.Rect(0, 0, 161, 101))  # odd sides: padded to even
    surface = pygame.Surface((161, 101))
    with pgp.Recorder(tmp_path / name, fps=20) as video:
        for k in range(8):
            grid.draw(surface)
            grid.plot(surface, lambda x, k=k: np.sin(x - k / 4))
            video.add(grid)
    assert (tmp_path / name).stat().st_size > 0
    assert frame_count(tmp_path / name) == 8


def test_a_recorder_given_no_frames_writes_nothing(tmp_path: Path) -> None:
    ffmpeg_or_skip()
    with pgp.Recorder(tmp_path / "empty.mp4"):
        pass
    assert not (tmp_path / "empty.mp4").exists()


def test_without_ffmpeg_videos_explain_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recording.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # not installed
    with pytest.raises(RuntimeError, match="pip install imageio-ffmpeg"):
        pgp.Recorder(tmp_path / "a.mp4")
    with pytest.raises(RuntimeError, match="not found at"):
        pgp.find_ffmpeg(tmp_path / "nowhere" / "ffmpeg")
    pgp.Recorder(tmp_path / "a.png").close()  # pictures need no ffmpeg


@pytest.mark.skipif(sys.platform == "win32", reason="the fake ffmpeg is a shell script")
def test_ffmpegs_own_message_is_shown_when_it_fails(tmp_path: Path) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho 'Unknown encoder libx264' >&2\nexit 1\n")
    fake.chmod(0o755)
    video = pgp.Recorder(tmp_path / "a.mp4", ffmpeg=fake)
    surface = pygame.Surface((40, 30))

    def record() -> None:
        for _ in range(200):  # until the pipe breaks, or the end
            video.add(surface)
        video.close()

    with pytest.raises(RuntimeError, match="Unknown encoder libx264") as error:
        record()
    assert "imageio-ffmpeg" in str(error.value)  # what to do about it
    assert video.closed
    assert os.path.exists(fake)
