"""Canonical JSON: the one byte form behind every hash and replay comparison (ARCHITECTURE §4.3).

Keys are sorted, floats use the shortest text that round-trips, and NaN and infinity are refused.
Times are UTC, written with a Z. A value without exactly one form (a naive datetime, a set, a
non-string key, a float32) is an error rather than a guess.
"""

from __future__ import annotations

from typing import Any


def canonical_dumps(value: Any) -> bytes:
    raise NotImplementedError("red:B1")


def canonical_loads(data: bytes | str) -> Any:
    raise NotImplementedError("red:B1")


def sha256_hex(value: Any) -> str:
    """SHA-256 of the canonical bytes, in hex."""
    raise NotImplementedError("red:B1")
