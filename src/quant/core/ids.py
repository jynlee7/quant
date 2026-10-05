"""Deterministic ids, derived from what they identify, so replays reproduce them."""

from __future__ import annotations

from datetime import datetime

from quant.core.clock import ensure_utc


def decision_id(tau: datetime) -> str:
    """`d20190702T1446Z` for the decision at 2019-07-02 14:46 UTC."""
    tau = ensure_utc(tau)
    if tau.second or tau.microsecond:
        raise ValueError(f"decision times fall on whole minutes, not {tau.isoformat()}")
    return f"d{tau:%Y%m%dT%H%M}Z"
