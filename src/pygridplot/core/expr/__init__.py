"""A small, safe language for maths expressions, compiled to NumPy.

Expressions such as ``"a sin(x - P.x) + 1"`` or ``"x^2 + y^2"`` are parsed
into syntax trees (:mod:`.parser`, never with :func:`eval`) and compiled
into functions of their variables (:mod:`.compiler`) built from NumPy
ufuncs, so they run on whole arrays, give NaN where undefined, and can be
differentiated exactly.

* :mod:`~pygridplot.core.expr.lexer`: text to tokens.
* :mod:`~pygridplot.core.expr.parser`: tokens to syntax trees; the grammar.
* :mod:`~pygridplot.core.expr.builtins`: the built-in constants and
  functions.
* :mod:`~pygridplot.core.expr.compiler`: syntax trees to functions, with
  names bound to values, functions and points.
* :mod:`~pygridplot.core.expr.function`: :func:`expression`, text to a
  function in one call.
* :mod:`~pygridplot.core.expr.errors`: :class:`ExpressionError`.

This subpackage builds on :mod:`pygridplot.core.plot` (for exact
derivatives, ``f'(x)``); nothing in ``core`` depends on it.
"""

from pygridplot.core.expr.builtins import CONSTANTS, FUNCTIONS, RESERVED
from pygridplot.core.expr.compiler import (
    Binding,
    CompiledExpression,
    ConstantBinding,
    FunctionBinding,
    PointBinding,
    ValueBinding,
    compile_expression,
)
from pygridplot.core.expr.errors import ExpressionError
from pygridplot.core.expr.function import expression
from pygridplot.core.expr.parser import Equation, parse, parse_equation, references

__all__ = [
    "CONSTANTS",
    "FUNCTIONS",
    "RESERVED",
    "Binding",
    "CompiledExpression",
    "ConstantBinding",
    "Equation",
    "ExpressionError",
    "FunctionBinding",
    "PointBinding",
    "ValueBinding",
    "compile_expression",
    "expression",
    "parse",
    "parse_equation",
    "references",
]
