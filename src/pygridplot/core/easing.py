"""Smooth motion towards moving targets with a critically damped spring.

Eased zooming and panning need a value (for example the log of the zoom
level) to follow a *target* that jumps every time the user scrolls a wheel
notch. Simple exponential smoothing jumps straight to full speed after
every notch and then slows down, so a notched wheel produces a visible
pulse per notch. A **critically damped spring** instead accelerates
smoothly and keeps its velocity continuous. A stream of notches then turns
into an even glide, and the spring still settles quickly without
overshooting.

The spring is advanced with the exact closed-form solution of its
differential equation, so it is stable for any time step (even a stalled
frame) and independent of the frame rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

SETTLE_FRACTION: Final = 3.89
"""``t / time_constant`` after which a single step has covered 90% of the way."""


@dataclass(slots=True)
class Spring:
    """A one-dimensional critically damped spring chasing a target.

    The spring only tracks the *offset* between the current value and its
    target, plus a velocity. The caller owns the actual value and applies the
    deltas returned by :meth:`advance`.

    Attributes:
        offset: Current value minus target value.
        velocity: Rate of change of the current value, per second.
    """

    offset: float = 0.0
    velocity: float = 0.0

    @property
    def active(self) -> bool:
        """Whether the spring is still moving or away from its target."""
        return self.offset != 0.0 or self.velocity != 0.0

    def push(self, amount: float) -> None:
        """Move the target by ``amount`` (the value will follow smoothly).

        Args:
            amount: Change of the target value.
        """
        self.offset -= amount

    def stop(self) -> None:
        """Stop immediately, discarding any remaining motion."""
        self.offset = 0.0
        self.velocity = 0.0

    def advance(
        self,
        dt: float,
        time_constant: float,
        *,
        settle_offset: float,
        settle_velocity: float,
    ) -> float:
        """Advance the spring by ``dt`` seconds.

        Args:
            dt: Elapsed time in seconds (non-negative).
            time_constant: ``1 / omega`` in seconds. A single jump of the target
                is 90% complete after about ``3.9 * time_constant``.
            settle_offset: Below this offset (together with a small velocity)
                the spring snaps onto its target and stops.
            settle_velocity: Velocity below which the spring may settle.

        Returns:
            How much the current value changed. The caller adds this to the
            value it is easing.
        """
        if not self.active:
            return 0.0
        if time_constant <= 0.0:
            delta = -self.offset
            self.stop()
            return delta
        omega = 1.0 / time_constant
        decay = math.exp(-omega * dt)
        bias = self.velocity + omega * self.offset
        new_offset = (self.offset + bias * dt) * decay
        new_velocity = (self.velocity - omega * bias * dt) * decay
        if abs(new_offset) < settle_offset and abs(new_velocity) < settle_velocity:
            delta = -self.offset  # land exactly on the target
            self.stop()
            return delta
        delta = new_offset - self.offset
        self.offset = new_offset
        self.velocity = new_velocity
        return delta
