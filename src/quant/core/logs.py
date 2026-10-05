"""Structured logs that never carry a secret (I6).

Every line passes a redactor that replaces known secret values (read from the environment),
fields named like secrets, and strings shaped like keys. Standard-library loggers, such as
httpx's, which prints request URLs, go through the same redactor.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

REDACTED = "[REDACTED]"


def configure_logging(
    *, level: str = "INFO", stream: io.TextIOBase | None = None, secrets: Iterable[str] = ()
) -> None:
    raise NotImplementedError("red:B1")
