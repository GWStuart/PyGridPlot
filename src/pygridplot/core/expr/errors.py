"""The error raised for expressions that cannot be read or compiled."""

from __future__ import annotations


class ExpressionError(ValueError):
    """An expression is not valid: a syntax error, an unknown name, a bad call.

    The message says what is wrong; when the expression's text and the
    position of the problem are known, :func:`str` also shows the text with
    a caret under the problem::

        unknown name 'sinx' (did you mean sin(x)?)
          y = sinx + 1
              ^

    Args:
        message: What is wrong.
        text: The expression's text, if known.
        position: Index of the problem in ``text``, if known.

    Attributes:
        message: What is wrong.
        text: The expression's text (empty if unknown).
        position: Index of the problem in ``text`` (``None`` if unknown).
    """

    def __init__(
        self, message: str, text: str = "", position: int | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.text = text
        self.position = position

    def __str__(self) -> str:
        """Return the message, with the text and a caret if they are known."""
        if not self.text or self.position is None:
            return self.message
        caret = " " * min(max(self.position, 0), len(self.text)) + "^"
        return f"{self.message}\n  {self.text}\n  {caret}"
