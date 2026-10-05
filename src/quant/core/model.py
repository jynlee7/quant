"""The base for every schema: config files, playbooks, proposals from L1, records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Unknown keys rejected, no type coercion, no NaN, and immutable once built."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )
