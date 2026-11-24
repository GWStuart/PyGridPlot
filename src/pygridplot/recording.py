"""Recording a plane, or a whole window, into a video or numbered pictures.

A :class:`Recorder` is given each frame as it is drawn::

    with pgp.Recorder("wave.mp4", fps=60) as video:
        while running:
            ...                     # events, drawing
            video.add(grid)         # this frame of the plane
            pygame.display.flip()

The file's extension decides what is written:

* **a video** (``.mp4``, ``.webm``, ``.gif``, ``.mov``, ``.mkv``, ...),
  encoded by `ffmpeg <https://ffmpeg.org>`_, a free program run alongside
  Python (see :func:`find_ffmpeg` for where it is looked for). Frames are
  piped to it as they come, so nothing piles up in memory;
* **numbered pictures** (``.png``, ``.jpg``, ``.bmp``, ``.tga``):
  ``frames/wave.png`` becomes ``frames/wave_00000.png``,
  ``frames/wave_00001.png``, ..., which need nothing beyond pygame (and
  ffmpeg can make a video of them later).

A plane is drawn again for each frame at the recorder's ``scale`` (see
:meth:`pygridplot.Grid.save`): ``scale=0.5`` for a quick, small draft,
``scale=2`` for a sharp final version, from the same program. A plane
recorded keeps the video's time: its playing sliders move on by exactly
``1 / fps`` seconds a frame, whatever time the frame took to draw and
encode, so the video plays at the right speed even when recording slows
the program down. For animations of your own, use :attr:`Recorder.time`
in place of the clock.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO, Final, cast

import pygame

from pygridplot.core.scaling import check_scale
from pygridplot.grid import Grid

PICTURE_TYPES: Final = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
"""Extensions recorded as numbered pictures instead of a video."""

_EVEN: Final = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
"""An ffmpeg filter making the frame's sides even, as most video codecs need."""

_H264: Final = (
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
)
"""H.264, nearly lossless (CRF 18), in the pixel format every player reads."""

_SETTINGS: Final[dict[str, tuple[str, ...]]] = {
    ".mp4": ("-vf", _EVEN, *_H264, "-movflags", "+faststart"),
    ".m4v": ("-vf", _EVEN, *_H264, "-movflags", "+faststart"),
    ".mov": ("-vf", _EVEN, *_H264, "-movflags", "+faststart"),
    ".mkv": ("-vf", _EVEN, *_H264),
    ".webm": (
        "-vf",
        _EVEN,
        *("-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p"),
    ),
    # A palette per frame: one for the whole video would hold every frame
    # in memory until the end.
    ".gif": (
        "-filter_complex",
        "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1",
        "-loop",
        "0",
    ),
}
"""ffmpeg's settings for the common video types (others take ffmpeg's defaults)."""

SETUP_HELP: Final = (
    "Recording a video needs ffmpeg, a free program, and it was not found. "
    "Install it (Windows: winget install ffmpeg; macOS: brew install ffmpeg; "
    "Debian or Ubuntu: sudo apt install ffmpeg), or install a copy just for "
    "Python with: pip install imageio-ffmpeg. Or give its path: "
    "pgp.Recorder(..., ffmpeg='/path/to/ffmpeg'). To record numbered pictures "
    "instead, which need nothing extra, record to a .png path. See the user "
    "guide's page Saving pictures and videos."
)
"""What to do when ffmpeg is missing."""


def find_ffmpeg(path: str | os.PathLike[str] | None = None) -> str:
    """Return the ffmpeg program to record videos with.

    It is looked for, in order: at ``path`` if given; as ``ffmpeg`` on the
    system's ``PATH`` (installed with a package manager, or by hand); and in
    the `imageio-ffmpeg <https://pypi.org/project/imageio-ffmpeg/>`_
    package, which brings a copy of its own (``pip install imageio-ffmpeg``),
    if it is installed.

    Args:
        path: The program, if you know where it is.

    Returns:
        The program's path.

    Raises:
        RuntimeError: If it cannot be found (the message says how to install
            it).
    """
    if path is not None:
        program = shutil.which(os.fspath(path))
        if program is None:
            msg = f"ffmpeg was not found at {os.fspath(path)!r}. {SETUP_HELP}"
            raise RuntimeError(msg)
        return program
    program = shutil.which("ffmpeg")
    if program is not None:
        return program
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # not installed, or its copy is missing
        raise RuntimeError(SETUP_HELP) from None


class Recorder:
    """Records frames into a video, through ffmpeg, or into numbered pictures.

    Create it before the frames, give it each frame with :meth:`add`, and
    close it at the end, best with a ``with`` block, which closes it however
    the block ends. The first frame sets the size of the video; later frames
    of another size (a window resized) are scaled to it.

    Args:
        path: The file to write. Its extension says what: a video
            (``.mp4``, ``.webm``, ``.gif``, ``.mov``, ``.mkv``, or any other
            type ffmpeg writes) or numbered pictures (``.png``, ``.jpg``,
            ``.bmp``, ``.tga``: ``frames/wave.png`` writes
            ``frames/wave_00000.png``, ...). Missing folders are created.
        fps: Frames per second of the video (and the time a frame moves the
            recorded planes' playing sliders on).
        scale: How many times as many pixels along each side a plane is
            drawn with (``0.5`` for a quick draft, ``2`` for a sharp video).
            A surface given to :meth:`add` is resized instead (smoothly, so
            it is blurred when made bigger).
        ffmpeg: Where the ffmpeg program is, if it is not found by itself
            (see :func:`find_ffmpeg`).

    Raises:
        TypeError: If ``fps`` or ``scale`` is not a number.
        ValueError: If ``fps`` or ``scale`` is not positive and finite.
        RuntimeError: If the file is a video and ffmpeg cannot be found.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fps: float = 60.0,
        scale: float = 1.0,
        ffmpeg: str | os.PathLike[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._fps = _check_fps(fps)
        self._scale = check_scale(scale)
        self._pictures = self._path.suffix.lower() in PICTURE_TYPES
        self._program = None if self._pictures else find_ffmpeg(ffmpeg)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._frames = 0
        self._size: tuple[int, int] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._errors: IO[bytes] | None = None
        self._planes: list[Grid] = []
        self._closed = False

    @property
    def path(self) -> Path:
        """The file being written (for pictures, the name they are numbered from)."""
        return self._path

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def scale(self) -> float:
        """How many times as many pixels along each side planes are drawn with."""
        return self._scale

    @property
    def frames(self) -> int:
        """How many frames have been added."""
        return self._frames

    @property
    def time(self) -> float:
        """The video's time at the next frame, in seconds: ``frames / fps``.

        Use it to animate your own things while recording, in place of the
        clock, so they move by exactly one frame's time per frame.
        """
        return self._frames / self._fps

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def add(self, source: Grid | pygame.Surface) -> None:
        """Add a frame: a plane as drawn in this frame, or a surface's pixels.

        Call it once per frame, after drawing it. A plane (a
        :class:`~pygridplot.Grid`) is drawn again at the recorder's scale,
        the plane alone, as :meth:`~pygridplot.Grid.save` does; from its
        first frame on, its playing sliders keep the video's time (one frame
        is ``1 / fps`` seconds). A surface, such as the window
        (``pygame.display.get_surface()``), is recorded as it is, with
        whatever you drew on it.

        Args:
            source: The plane, or a surface.

        Raises:
            ValueError: If the recorder is closed, or nothing was drawn on
                the plane in this frame.
            TypeError: If ``source`` is neither a plane nor a surface.
            RuntimeError: If ffmpeg stopped with an error (its message is
                included).
        """
        if self._closed:
            msg = "this recorder is closed"
            raise ValueError(msg)
        if not isinstance(cast(object, source), (Grid, pygame.Surface)):
            msg = f"add a Grid or a pygame.Surface, got {type(source).__name__}"
            raise TypeError(msg)
        if isinstance(source, Grid):
            picture = source._snapshot(self._scale, opaque=self._opaque)
            if not any(plane is source for plane in self._planes):
                self._planes.append(source)
                source._use_clock(self._clock)
        else:
            picture = source
            if self._scale != 1.0:
                size = (
                    max(1, round(source.get_width() * self._scale)),
                    max(1, round(source.get_height() * self._scale)),
                )
                picture = pygame.transform.smoothscale(source, size)
        if self._size is None:
            self._size = picture.get_size()
        elif picture.get_size() != self._size:
            picture = pygame.transform.smoothscale(picture, self._size)
        if self._pictures:
            self._save_picture(picture)
        else:
            self._write(picture)
        self._frames += 1

    def close(self) -> None:
        """Finish the video (or the pictures), and give the planes their clock back.

        Closing twice does nothing. A recorder given no frames writes no
        file.

        Raises:
            RuntimeError: If ffmpeg stopped with an error (its message is
                included).
        """
        if self._closed:
            return
        self._closed = True
        for plane in self._planes:
            plane._use_clock(None)
        self._planes.clear()
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except BrokenPipeError:
            pass
        code = process.wait()
        if code != 0:
            self._fail(f"ffmpeg stopped with an error (exit code {code})")
        self._close_errors()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if kind is None:
            self.close()
            return
        with contextlib.suppress(Exception):  # keep what was recorded, hide nothing
            self.close()

    def __repr__(self) -> str:
        """Return a short description: the file, the rate and the frames so far."""
        return (
            f"Recorder({str(self._path)!r}, fps={self._fps!r}, frames={self._frames})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @property
    def _opaque(self) -> bool:
        """Whether frames must be opaque (all but PNG pictures, which keep alpha)."""
        return not (self._pictures and self._path.suffix.lower() == ".png")

    def _clock(self) -> float:
        """The video's time at the next frame, in milliseconds (for planes)."""
        return self._frames * 1000.0 / self._fps

    def _save_picture(self, picture: pygame.Surface) -> None:
        """Save a frame as the next numbered picture."""
        name = f"{self._path.stem}_{self._frames:05d}{self._path.suffix}"
        pygame.image.save(picture, str(self._path.with_name(name)))

    def _write(self, picture: pygame.Surface) -> None:
        """Pipe a frame to ffmpeg, starting it at the first frame."""
        if self._process is None:
            self._start()
        process = self._process
        assert process is not None
        assert process.stdin is not None
        try:
            process.stdin.write(pygame.image.tobytes(picture, "RGB"))
        except (BrokenPipeError, OSError):
            process.wait()
            self._process = None
            self._fail("ffmpeg stopped while recording")

    def _start(self) -> None:
        """Start ffmpeg, reading frames of the recorded size from its input."""
        assert self._size is not None
        assert self._program is not None
        width, height = self._size
        command = [
            self._program,
            *("-hide_banner", "-loglevel", "error", "-y"),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}"),
            *("-r", f"{self._fps:g}", "-i", "-"),
            *_SETTINGS.get(self._path.suffix.lower(), ("-vf", _EVEN)),
            str(self._path),
        ]
        # Kept open while ffmpeg runs (it writes its messages there), closed
        # when it ends.
        self._errors = tempfile.TemporaryFile()  # noqa: SIM115
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._errors,
        )

    def _fail(self, what: str) -> None:
        """Raise an error with ffmpeg's own message, and stop recording."""
        message = ""
        if self._errors is not None:
            self._errors.seek(0)
            message = self._errors.read().decode(errors="replace").strip()
        self._close_errors()
        self._closed = True
        for plane in self._planes:
            plane._use_clock(None)
        self._planes.clear()
        hint = ""
        if "Unknown encoder" in message or "Encoder not found" in message:
            hint = (
                " This ffmpeg cannot write that kind of video; pip install "
                "imageio-ffmpeg brings a complete one."
            )
        msg = f"{what}: {message or 'no message'}.{hint}"
        raise RuntimeError(msg)

    def _close_errors(self) -> None:
        """Close the file ffmpeg's messages were kept in."""
        if self._errors is not None:
            self._errors.close()
            self._errors = None


def _check_fps(fps: object) -> float:
    """Validate a frame rate.

    Raises:
        TypeError: If it is not a number.
        ValueError: If it is not positive and finite.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        msg = f"fps must be a number, got {fps!r}"
        raise TypeError(msg)
    value = float(fps)
    if not (math.isfinite(value) and value > 0):
        msg = f"fps must be positive and finite, got {fps!r}"
        raise ValueError(msg)
    return value
