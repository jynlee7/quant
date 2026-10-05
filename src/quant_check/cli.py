"""`quant-check`: the build gate that runs before every commit (ARCHITECTURE §13)."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError("red:B0")
