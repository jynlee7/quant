"""Pure decision arithmetic shared across switching, sizing and intents."""

from fractions import Fraction


def exact_decimal(value: float) -> Fraction:
    """Exact decimal value, independent of the ambient decimal context."""
    return Fraction(repr(float(value)))
