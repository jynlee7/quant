"""The base for every schema: config files, playbooks, proposals from L1, records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StrictModel(BaseModel):
    """Unknown keys rejected, no type coercion, no NaN, and immutable once built."""

    def __init__(self, **data: Any) -> None:
        raise NotImplementedError("red:B1")

    @classmethod
    def model_validate(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("red:B1")
