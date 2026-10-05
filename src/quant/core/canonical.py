"""Canonical JSON: the one byte form behind every hash and replay comparison (ARCHITECTURE §4.3).

Keys are sorted, floats use the shortest text that round-trips, and NaN and infinity are refused.
Times are UTC, written with a Z. A value without exactly one form (a naive datetime, a set, a
non-string key, a float32) is an error rather than a guess.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel

from quant.core.errors import CanonicalJSONError


def _float(value: float, path: str) -> float:
    if not math.isfinite(value):
        raise CanonicalJSONError(f"{path}: {value!r} has no JSON form")
    return value


def _normalize(value: Any, path: str) -> Any:
    """`value` as plain JSON types, or an error naming where it has no single form."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _float(float(value), path)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if value.dtype != np.float64:
            raise CanonicalJSONError(f"{path}: {value.dtype} isn't float64")
        return _float(float(value), path)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or (
            value.dtype.kind == "f" and value.dtype != np.float64
        ):
            raise CanonicalJSONError(f"{path}: arrays of {value.dtype} have no canonical form")
        if value.ndim == 0:
            return _normalize(value[()], path)
        return [_normalize(item, f"{path}[{i}]") for i, item in enumerate(value.tolist())]
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalJSONError(f"{path}: naive datetime {value.isoformat()}")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _normalize(value.value, path)
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"), path)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{path}: key {key!r} isn't a string")
            out[key] = _normalize(item, f"{path}.{key}")
        return out
    if isinstance(value, list | tuple):
        return [_normalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise CanonicalJSONError(f"{path}: {type(value).__name__} has no canonical JSON form")


def canonical_dumps(value: Any) -> bytes:
    text = json.dumps(
        _normalize(value, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError(f"a string can't be encoded as UTF-8: {exc}") from exc


def _refuse_constant(name: str) -> Any:
    raise CanonicalJSONError(f"{name} has no JSON form")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in pairs:
        if key in out:
            raise CanonicalJSONError(f"duplicate key {key!r}")
        out[key] = item
    return out


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise CanonicalJSONError(f"{text} overflows float64")
    return value


def canonical_loads(data: bytes | str) -> Any:
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalJSONError(f"not UTF-8: {exc}") from exc
    try:
        return json.loads(
            data,
            parse_float=_finite_float,
            parse_constant=_refuse_constant,
            object_pairs_hook=_unique_keys,
        )
    except json.JSONDecodeError as exc:
        raise CanonicalJSONError(f"not JSON: {exc}") from exc


def sha256_hex(value: Any) -> str:
    """SHA-256 of the canonical bytes, in hex."""
    return hashlib.sha256(canonical_dumps(value)).hexdigest()
