"""Choosing the major grid spacing for an axis from the current zoom level.

The ideal major spacing in world units is ``raw = target_px / scale``. It is
rounded onto the 1-2-5 ladder (see :mod:`pygridplot.core.nice`). The hard
part is what happens as the zoom crosses the point where one ladder step
gives way to the next. Two strategies are supported:

``"fade"`` (default)
    A pure function of the scale. In a narrow band around each switching
    point both spacings are drawn, cross-faded by a smooth weight. Zooming
    through the band blends one grid into the next without any sudden jump.

``"snap"``
    Only one spacing is ever drawn and it switches instantly. A
    :class:`SnapTracker` adds hysteresis so the grid does not flicker back and
    forth when the zoom sits right on a switching point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pygridplot.core.nice import NiceStep, SpacingRule, level_boundary, select_level

TransitionMode: TypeAlias = Literal["fade", "snap"]
"""How the grid moves between ladder steps while zooming (see module docs)."""

MAX_FADE_BAND: Final = 0.14
"""Largest allowed fade half-width in ``log10`` units.

Consecutive switching points are at least ``log10(2) ≈ 0.30`` apart, so a
half-width below ``0.15`` guarantees that two fade bands never overlap.
"""


@dataclass(frozen=True, slots=True)
class SpacingChoice:
    """The major spacing for one axis, possibly part-way between two steps.

    Attributes:
        fine: The finer of the two spacings. Always present.
        coarse: The coarser spacing being faded in, or ``None`` if only
            ``fine`` is drawn.
        blend: The opacity of ``coarse`` in ``[0, 1]``. ``fine`` is drawn with
            opacity ``1 - blend``. Always ``0`` when ``coarse`` is ``None``.
    """

    fine: NiceStep
    coarse: NiceStep | None = None
    blend: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the blend weight is consistent with ``coarse``."""
        if not 0.0 <= self.blend <= 1.0:
            msg = f"blend must be within [0, 1], got {self.blend}"
            raise ValueError(msg)
        if self.coarse is None and self.blend != 0.0:
            msg = "blend must be 0 when there is no coarse step"
            raise ValueError(msg)

    @property
    def dominant(self) -> NiceStep:
        """Whichever of the two steps currently has the larger weight."""
        if self.coarse is not None and self.blend >= 0.5:
            return self.coarse
        return self.fine

    def weighted_steps(self) -> tuple[tuple[NiceStep, float], ...]:
        """Return each step that is drawn at all, paired with its opacity.

        Returns:
            One or two ``(step, weight)`` pairs whose weights sum to 1. Steps
            with zero weight are left out.
        """
        if self.coarse is None or self.blend == 0.0:
            return ((self.fine, 1.0),)
        if self.blend == 1.0:
            return ((self.coarse, 1.0),)
        return ((self.fine, 1.0 - self.blend), (self.coarse, self.blend))


def raw_spacing(scale: float, target_px: float) -> float:
    """Return the world spacing that would appear exactly ``target_px`` apart.

    Args:
        scale: Pixels per world unit along the axis.
        target_px: Desired on-screen distance between major lines.

    Returns:
        ``target_px / scale`` in world units.
    """
    return target_px / scale


def smoothstep(t: float) -> float:
    """Return the cubic smoothstep of ``t`` clamped to ``[0, 1]``.

    It starts and ends with zero slope, so fades begin and end gently.

    Args:
        t: Input value.

    Returns:
        ``3t² - 2t³`` for ``t`` in ``[0, 1]``, and 0 or 1 outside that range.
    """
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def choose_faded(
    scale: float, target_px: float, rule: SpacingRule, band: float
) -> SpacingChoice:
    """Choose the spacing for ``"fade"`` transitions (a pure function of zoom).

    Let ``u = log10(raw)`` and let ``b`` be the switching point between ladder
    levels ``L`` and ``L + 1``. When ``|u - b| < band`` both levels are drawn,
    with the coarser one weighted ``smoothstep((u - (b - band)) / (2 band))``.
    Outside the bands only the selected level is drawn.

    Args:
        scale: Pixels per world unit along the axis.
        target_px: Desired on-screen distance between major lines.
        rule: The ladder rounding rule.
        band: Half-width of each fade band in ``log10`` units. ``0`` gives
            hard switching.

    Returns:
        The (possibly blended) spacing.

    Raises:
        ValueError: If ``band`` is negative or larger than
            :data:`MAX_FADE_BAND`.
    """
    if not 0.0 <= band <= MAX_FADE_BAND:
        msg = f"fade band must be within [0, {MAX_FADE_BAND}], got {band}"
        raise ValueError(msg)
    raw = raw_spacing(scale, target_px)
    level = select_level(raw, rule)
    if band == 0.0:
        return SpacingChoice(NiceStep(level))
    u = math.log10(raw)
    upper = level_boundary(level, rule)
    lower = level_boundary(level - 1, rule)
    if u > upper - band:  # approaching the next coarser level
        weight = smoothstep((u - (upper - band)) / (2.0 * band))
        return _blend(level, weight)
    if u <= lower + band:  # just left the next finer level
        weight = smoothstep((u - (lower - band)) / (2.0 * band))
        return _blend(level - 1, weight)
    return SpacingChoice(NiceStep(level))


def _blend(fine_level: int, weight: float) -> SpacingChoice:
    """Build a blended choice, collapsing it when the weight is exactly 0 or 1.

    Args:
        fine_level: The finer ladder level of the pair.
        weight: Opacity of the coarser level.

    Returns:
        The corresponding :class:`SpacingChoice`.
    """
    if weight <= 0.0:
        return SpacingChoice(NiceStep(fine_level))
    if weight >= 1.0:
        return SpacingChoice(NiceStep(fine_level + 1))
    return SpacingChoice(NiceStep(fine_level), NiceStep(fine_level + 1), weight)


class SnapTracker:
    """Chooses spacings for ``"snap"`` transitions, with hysteresis.

    The tracker remembers the level it chose last time. It only switches to a
    different level once the raw spacing has gone past the switching point by
    more than ``hysteresis`` (in ``log10`` units). Holding the zoom right on a
    boundary therefore never makes the grid flicker.

    One tracker is needed per axis per plane.
    """

    def __init__(self) -> None:
        self._level: int | None = None

    @property
    def level(self) -> int | None:
        """The level chosen by the last call to :meth:`choose`, if any."""
        return self._level

    def reset(self) -> None:
        """Forget the previous choice so the next one is made from scratch."""
        self._level = None

    def choose(
        self, scale: float, target_px: float, rule: SpacingRule, hysteresis: float
    ) -> SpacingChoice:
        """Choose the spacing for the current scale.

        Args:
            scale: Pixels per world unit along the axis.
            target_px: Desired on-screen distance between major lines.
            rule: The ladder rounding rule.
            hysteresis: How far past a switching point (in ``log10`` units)
                the zoom must go before the level changes.

        Returns:
            A non-blended :class:`SpacingChoice`.
        """
        raw = raw_spacing(scale, target_px)
        current = self._level
        if current is not None:
            u = math.log10(raw)
            low = level_boundary(current - 1, rule) - hysteresis
            high = level_boundary(current, rule) + hysteresis
            if low < u <= high:
                return SpacingChoice(NiceStep(current))
        self._level = select_level(raw, rule)
        return SpacingChoice(NiceStep(self._level))
