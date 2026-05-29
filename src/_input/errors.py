"""Exception types for the input layer.

Defined here (rather than in ``read_param``) so the registry validators
can ``raise ParameterFileError`` without creating a
``registry → read_param → registry`` import cycle.  ``read_param``
re-exports the name for back-compat.
"""
from __future__ import annotations


class ParameterFileError(Exception):
    """Raised when a parameter file has formatting or validation errors."""
